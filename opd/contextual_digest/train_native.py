import argparse
from dataclasses import asdict
from datetime import timedelta
import gc
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import threading
import time
import traceback

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer

from .backend import HFBackend
from .alignment import require_baseline_recipe
from .batch_journal import BatchJournal
from .batched_rollout import independent_batch
from .native_rollout import NativeRelayRollout
from .protocols import batched_handoff_implementation
from .config import TrainConfig
from .data import OnlineTrainingStream, load_questions
from .journal import RolloutJournal, model_fingerprint
from .snapshot import verify_snapshot, publish_update
from .train import emit, save_checkpoint, PROJECT
from .update import MetaHandoffUpdater
from .vllm_client import VLLMClient


DEFAULT_EXECUTION = {'micro_batch_size': 64, 'native_speculative_tokens': 4, 'vllm_max_num_seqs': 1024,
                     'vllm_memory_fraction': 0.45, 'enforce_eager': False,
                     'independent_memory_fraction': 0.12, 'independent_max_num_seqs': 16,
                     'graph_max_num_seqs': 16}


def validate_execution(execution):
    if set(execution) != set(DEFAULT_EXECUTION):
        raise ValueError('Unknown or missing native execution settings')
    for key in ('micro_batch_size', 'vllm_max_num_seqs', 'independent_max_num_seqs', 'graph_max_num_seqs'):
        if not isinstance(execution[key], int) or execution[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if execution['native_speculative_tokens'] != 4:
        raise ValueError('Native speculation must match the original Relay run (4 tokens)')
    if execution['micro_batch_size'] > execution['vllm_max_num_seqs']:
        raise ValueError('Microbatch exceeds native inference sequence slots')
    if any(not 0 < execution[k] < 1 for k in ('vllm_memory_fraction', 'independent_memory_fraction')):
        raise ValueError('Invalid native engine memory fractions')
    if not isinstance(execution['enforce_eager'], bool):
        raise ValueError('Invalid native eager setting')


def sync_student(client, student, directory):
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / 'student_transfer.pt'
    weights = {name: tensor.detach().to(device='cpu', dtype=torch.bfloat16)
               for name, tensor in student.state_dict().items()}
    torch.save(weights, target)
    sha = hashlib.sha256(target.read_bytes()).hexdigest()
    result = client.call('load_weights', path=str(target), sha256=sha)
    if len(result) != 1 or result[0]['file_sha256'] != sha or result[0]['verified_input_tensors'] != len(weights):
        raise ValueError('Inference Student weight verification is incomplete')
    target.unlink()
    return result[0]


def release_training_cuda_cache(devices):


    gc.collect()
    records = []
    for device in dict.fromkeys(devices):
        with torch.cuda.device(device):
            torch.cuda.synchronize()
            before = dict(allocated=torch.cuda.memory_allocated(),
                          reserved=torch.cuda.memory_reserved(),
                          free=torch.cuda.mem_get_info()[0])
            torch.cuda.empty_cache()
            after = dict(allocated=torch.cuda.memory_allocated(),
                         reserved=torch.cuda.memory_reserved(),
                         free=torch.cuda.mem_get_info()[0])
        records.append(dict(device=device, before=before, after=after))
    return records


def run(config, execution, *, resume=None, recover_generation=False, stop_after_step=None):
    world, rank = int(os.environ.get('WORLD_SIZE', '1')), int(os.environ.get('RANK', '0'))
    config.validate(world)
    validate_execution(execution)
    baseline_alignment = require_baseline_recipe(config, execution)
    if stop_after_step is not None and (type(stop_after_step) is not int or stop_after_step < 1):
        raise ValueError('stop_after_step must be a positive absolute committed step')
    builder_class, graph_system, wire_format = batched_handoff_implementation(config.handoff.protocol)
    if resume and recover_generation:
        raise ValueError('Choose checkpoint or initial-generation recovery')
    output = Path(config.output_dir)
    if not output.is_absolute():
        output = PROJECT / output
    sources = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).parent.glob('*.py'))}
    torch.cuda.set_device(config.student_devices[rank])
    if world > 1:
        dist.init_process_group('nccl', device_id=torch.device(f'cuda:{config.student_devices[rank]}'),
                                timeout=timedelta(hours=4))
    if rank == 0:
        if resume:
            snapshot = verify_snapshot(resume, world)
            if snapshot['source_sha256'] != sources:
                raise ValueError('Checkpoint source differs from this execution implementation')
            if json.loads((resume / 'execution.json').read_text()) != execution:
                raise ValueError('Checkpoint execution settings changed')
        if recover_generation:
            prior = json.loads((output / 'run_manifest.json').read_text())
            if prior['config'] != config.to_dict() or prior['execution'] != execution or prior['generation_source_sha256'] != sources:
                raise ValueError('Initial generation recovery needs the same method and execution')
            if list(output.glob('metrics_rank_*.jsonl')) or (output / 'latest_recovery.txt').exists():
                raise ValueError('Recover updated Student from its full checkpoint')
        if (output / 'run_manifest.json').exists() and not resume and not recover_generation:
            raise FileExistsError('A new run needs a new output directory')
        output.mkdir(parents=True, exist_ok=True)
        manifest = {'method': 'CD-OPD', 'backend': 'original_vllm_relay_with_async_handoff_graph',
                    'shared_baseline_parameters': baseline_alignment,
                    'student_forward_precision': {'master_parameters': 'float32', 'forward_parameters': 'bfloat16',
                                                  'buffers': 'float32', 'gradient_reduce': 'float32'},
                    'native_sampling_seed_policy': 'engine seed = replica rank; no per-request seed override',
                    'config': config.to_dict(), 'execution': execution, 'world_size': world,
                    'teacher_frozen': True, 'meta_gradient_scope': 'all trainable student parameters',
                    'task_gradient_only_for_meta': True, 'wire_format': wire_format,
                    'mandatory_summary_policy': ({'per_unfinished_takeover': True, 'student_resumes_after_takeover_limit': True,
                        'reserved_before_each_native_call': config.handoff.max_summary_tokens,
                        'total_response_limit_unchanged': True, 'teacher_eos_ends_response': True,
                        'sampling_budget_allocation_changed': True}
                        if config.handoff.protocol == 'contextual_digest' else None),
                    'generation_source_sha256': sources, 'resume': str(resume) if resume else None,
                    'run_control': {'stop_after_step': stop_after_step},
                    'sampling_note': 'Same temperature/top-p/Relay rule; backend rounding and request RNG differ from HF serial realizations.'}
        (output / 'run_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    if world > 1:
        dist.barrier()
    torch.manual_seed(config.seed)
    tokenizer = AutoTokenizer.from_pretrained(config.student_model, local_files_only=True)
    teacher_tokenizer = AutoTokenizer.from_pretrained(config.teacher_model, local_files_only=True)
    if tokenizer.get_vocab() != teacher_tokenizer.get_vocab():
        raise ValueError('Public Student/Teacher token vocabularies differ')
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    student_path = str(resume / 'student') if resume else config.student_model
    student = AutoModelForCausalLM.from_pretrained(student_path, dtype=torch.float32, local_files_only=True,
                                                  attn_implementation='sdpa').to(f'cuda:{config.student_devices[rank]}')
    teacher = AutoModelForCausalLM.from_pretrained(config.teacher_model, dtype=torch.bfloat16, local_files_only=True,
                                                  attn_implementation='sdpa').to(f'cuda:{config.teacher_devices[rank]}')
    teacher.requires_grad_(False)
    teacher.eval()
    if getattr(student.config, 'attention_dropout', 0) or any(isinstance(m, torch.nn.Dropout) and m.p for m in student.modules()):
        raise ValueError('One shared Student snapshot requires dropout=0')
    if config.gradient_checkpointing:
        student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    student_backend = HFBackend(student, score_chunk_tokens=config.score_chunk_tokens,
                                forward_parameter_dtype=torch.bfloat16)
    teacher_backend = HFBackend(teacher, score_chunk_tokens=config.score_chunk_tokens)
    optimizer = torch.optim.AdamW(student.parameters(), lr=config.lr, betas=tuple(config.adam_betas), weight_decay=config.weight_decay)
    updater = MetaHandoffUpdater(student_backend, teacher_backend, optimizer, config, world > 1)
    questions, skipped = load_questions(config.train_data, tokenizer, config.relay.max_prompt_tokens)
    stream = OnlineTrainingStream(questions, config.handoff_batch_size,
                                  config.meta.task_batch_size if config.meta.enabled else 0, config.seed, config.epochs)
    student_generator = torch.Generator(device=student_backend.device).manual_seed(config.seed + rank)
    teacher_generator = torch.Generator(device=teacher_backend.device).manual_seed(config.seed + 10000 + rank)
    if resume:
        saved, current = TrainConfig.load(resume / 'config.json').to_dict(), config.to_dict()
        for key in ('output_dir', 'save_every', 'max_steps'):
            saved.pop(key)
            current.pop(key)
        if saved != current:
            raise ValueError('Checkpoint training parameters differ')
        optimizer.load_state_dict(torch.load(resume / 'optimizer.pt', map_location=student_backend.device, weights_only=False))
        state = torch.load(resume / f'rank_{rank}.pt', map_location='cpu', weights_only=False)
        snapshot = json.loads((resume / 'snapshot_manifest.json').read_text())
        if model_fingerprint(student) != snapshot['student_sha256'] or state['step'] != snapshot['step']:
            raise ValueError('Checkpoint Student/rank state mismatch')
        stream.load_state_dict(state['stream'])
        student_generator.set_state(state['student_rng'])
        teacher_generator.set_state(state['teacher_rng'])
        torch.set_rng_state(state['torch_rng'])
        torch.cuda.set_rng_state_all(state['cuda_rng'])
        if state.get('payload'):
            publish_update(output, rank, state['payload']['metrics'], state['payload']['audit'])
        emit('resume_verified', rank=rank, step=state['step'], student_sha256=snapshot['student_sha256'])
    if stop_after_step is not None and stream.step >= stop_after_step:
        raise ValueError('Requested stop step is already committed; choose a later step or omit the control')
    module, name = config.reward_function.split(':', 1)
    reward = getattr(importlib.import_module(module), name)
    os.environ['VERL_OPD_TOKENIZER_PATH'] = config.student_model
    from opd.patches.vllm.speculative_decode import _get_reflection_token_ids, _get_paragraph_boundary_token_ids
    reflection_ids = set(_get_reflection_token_ids(torch.device('cpu')).tolist())
    paragraph_ids = set(_get_paragraph_boundary_token_ids(torch.device('cpu')).tolist())
    if not reflection_ids:
        raise ValueError('Relay reflection markers missing')
    progress = {'rank': rank, 'step': stream.step, 'phase': 'engine_start', 'item': 0}
    heartbeat_stop = threading.Event()
    def heartbeat():
        while not heartbeat_stop.wait(30):
            emit('heartbeat', **dict(progress))
    threading.Thread(target=heartbeat, daemon=True).start()
    clients = []
    failed = False
    try:
        emit('cuda_cache_released', rank=rank, step=stream.step, phase='engine_start',
             devices=release_training_cuda_cache([config.student_devices[rank], config.teacher_devices[rank]]))
        native_client = VLLMClient(config.teacher_model, config.student_model, config.student_devices[rank],
                                   output / 'runtime/native', max_model_len=config.handoff.graph_max_model_len,
                                   max_num_seqs=execution['vllm_max_num_seqs'],
                                   memory_fraction=execution['vllm_memory_fraction'],
                                   enforce_eager=execution['enforce_eager'],
                                   worker_module='opd.contextual_digest.native_worker',
                                   worker_options={'draft_model': student_path, 'native_relay': config.to_dict()['relay'],
                                                   'engine_seed': rank})
        clients.append(native_client)
        student_client = VLLMClient(student_path, config.student_model, config.student_devices[rank],
                                   output / 'runtime/independent',
                                   max_model_len=config.relay.max_prompt_tokens + config.meta.task_max_response_tokens,
                                   max_num_seqs=execution['independent_max_num_seqs'],
                                   memory_fraction=execution['independent_memory_fraction'],
                                   enforce_eager=execution['enforce_eager'])
        clients.append(student_client)
        teacher_client = VLLMClient(config.teacher_model, config.student_model, config.teacher_devices[rank],
                                   output / 'runtime/graph', max_model_len=config.handoff.graph_max_model_len,
                                   graph_constraints=True, max_num_seqs=execution['graph_max_num_seqs'],
                                   memory_fraction=execution['vllm_memory_fraction'], enforce_eager=True)
        clients.append(teacher_client)
        builder = builder_class(teacher_client, tokenizer, config.handoff) if config.handoff.enabled else None
        rollout = NativeRelayRollout(native_client, tokenizer, config.relay, builder,
                                    graph_batch_size=execution['graph_max_num_seqs'], student_client=student_client)
        emit('ready', rank=rank, training_questions=len(questions), filtered_prompts=skipped, world_size=world,
             micro_batch_size=execution['micro_batch_size'], execution=execution)
        sleeping, last_saved = False, stream.step if resume else None
        controlled_stop = False
        while config.max_steps is None or stream.step < config.max_steps:
            try:
                handoff_questions, task_questions = next(stream)
            except StopIteration:
                break
            step, started = stream.step, time.perf_counter()
            progress.update(step=step, phase='student_sync')
            if sleeping:
                emit('cuda_cache_released', rank=rank, step=step, phase='before_inference_wake',
                     devices=release_training_cuda_cache([config.student_devices[rank], config.teacher_devices[rank]]))
                for client in clients:
                    client.wake()
            verification = sync_student(student_client, student, output / 'runtime' / f'rank_{rank}')
            emit('student_inference_weights_verified', rank=rank, step=step, **verification)
            native_verification = sync_student(native_client, student, output / 'runtime' / f'native_rank_{rank}')
            if native_verification['file_sha256'] != verification['file_sha256']:
                raise ValueError('Independent and native Student snapshots differ')
            emit('native_draft_weights_verified', rank=rank, step=step, **native_verification)
            rollout.timings = {}
            emit('rollout_start', rank=rank, step=step)
            identity_config = config.to_dict()
            for key in ('output_dir', 'save_every', 'max_steps'):
                identity_config.pop(key)
            identity = {'student_sha256': model_fingerprint(student), 'config': identity_config, 'execution': execution,
                        'graph_extraction_instructions': graph_system if builder else None,
                        'generation_source_sha256': sources, 'corpus': stream.fingerprint,
                        'handoff_ids': [q.id for q in handoff_questions], 'independent_ids': [q.id for q in task_questions]}
            journal = BatchJournal(RolloutJournal(output, step, rank, identity, (student_generator, teacher_generator)))
            handoff = []
            local = handoff_questions[rank::world]
            micro = execution['micro_batch_size']
            offset = 0
            while offset < len(local):
                group = local[offset:offset + micro]
                progress.update(phase='handoff', item=offset + 1, total=len(local))
                cached = journal.load_prefix('handoff', offset, [q.id for q in group])
                if cached:
                    traces, _ = cached
                    group = group[:len(traces)]
                else:
                    try:
                        traces = rollout.generate_many(group, student_generator, teacher_generator)
                    except Exception:
                        if builder and builder.last_attempt:
                            (output / f'failed_graph_step_{step}_rank_{rank}.json').write_text(json.dumps(builder.last_attempt, ensure_ascii=False, indent=2))
                        raise
                    journal.save('handoff', offset, traces)
                handoff.extend(traces)
                for index, trace in enumerate(traces, offset):
                    emit('handoff_rollout', rank=rank, step=step, item=index, response_tokens=len(trace.response_ids),
                         summary_segments=len(trace.summary_spans), stop_reason=trace.stop_reason, cached=bool(cached))
                offset += len(group)
            progress.update(phase='independent', item=0, total=len(task_questions[rank::world]) * config.meta.task_samples)
            task_local = task_questions[rank::world]
            tasks = []
            independent_started = time.perf_counter()
            if config.meta.enabled:
                cached = journal.load('independent', 0, [q.id for q in task_local for _ in range(config.meta.task_samples)])
                if cached:
                    independent, rewards = cached
                else:
                    independent = independent_batch(student_client, tokenizer, task_local, config.meta.task_samples,
                                                     config.meta.task_max_response_tokens, student_generator)
                    by_id = {q.id: q for q in task_local}
                    rewards = []
                    for trace in independent:
                        question = by_id[trace.question_id]
                        score = reward(data_source=question.data_source, solution_str=tokenizer.decode(trace.response_ids, skip_special_tokens=True),
                                       ground_truth=question.ground_truth)
                        rewards.append(float(score['score'] if isinstance(score, dict) else score))
                    journal.save('independent', 0, independent, rewards)
                tasks = list(zip(independent, rewards, strict=True))
                for index, (trace, score) in enumerate(tasks):
                    emit('independent_rollout', rank=rank, step=step, sample=index, score=score, response_tokens=len(trace.response_ids))
            rollout.timings['independent_seconds'] = time.perf_counter() - independent_started
            progress.update(phase='engine_sleep')
            for client in clients:
                client.sleep()
            sleeping = True
            progress.update(phase='meta_update')
            emit('meta_update_start', rank=rank, step=step)
            update_started = time.perf_counter()
            metrics, segment_audits = updater.update(handoff, tasks)
            metrics.update(step=step, rank=rank, wall_seconds=time.perf_counter() - started,
                           meta_update_seconds=time.perf_counter() - update_started, rollout_timing=dict(rollout.timings))
            audit = {'step': step, 'snapshot_step': step - 1, 'handoff_question_ids': [q.id for q in handoff_questions],
                     'task_question_ids': [q.id for q in task_questions], 'handoff_trajectories': [asdict(t) for t in handoff],
                     'task_trajectories': [{'trajectory': asdict(t), 'reward': r} for t, r in tasks],
                     'meta_segments': segment_audits, 'execution': execution}
            controlled_stop = stop_after_step is not None and step >= stop_after_step
            regular = step % config.save_every == 0 or step == config.max_steps or controlled_stop
            progress.update(phase='checkpoint')
            save_checkpoint(output, step, student, tokenizer, optimizer, stream, (student_generator, teacher_generator),
                            config, rank, world, recovery=not regular, payload={'metrics': metrics, 'audit': audit}, execution=execution)
            publish_update(output, rank, metrics, audit)
            emit('update_complete', **metrics)
            if regular:
                last_saved = step
            if controlled_stop:
                break
        if stream.step and last_saved != stream.step:
            save_checkpoint(output, stream.step, student, tokenizer, optimizer, stream, (student_generator, teacher_generator),
                            config, rank, world, execution=execution)
        emit('run_paused' if controlled_stop else 'run_complete', rank=rank, completed_steps=stream.step,
             stop_after_step=stop_after_step)
    except BaseException as exc:
        failed = True
        emit('run_failed', rank=rank, step=stream.step, error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        heartbeat_stop.set()
        for client in reversed(clients):
            try:
                client.close()
            except Exception as exc:
                emit('engine_cleanup_error', rank=rank, error=str(exc))
        if failed and world > 1:


            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(1)
        if world > 1 and dist.is_initialized():
            dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--check-config', action='store_true')
    parser.add_argument('--execution-config', type=Path)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--recover-generation', action='store_true')
    parser.add_argument('--max-steps', type=int)
    parser.add_argument('--stop-after-step', type=int,
                        help='Save and exit after this absolute update; keep the full training recipe unchanged')
    args = parser.parse_args()
    config = TrainConfig.load(args.config)
    if args.max_steps is not None:
        config.max_steps = args.max_steps
    execution = json.loads(args.execution_config.read_text()) if args.execution_config else dict(DEFAULT_EXECUTION)
    if args.check_config:
        config.validate(len(config.student_devices))
        validate_execution(execution)
        require_baseline_recipe(config, execution)
        print(json.dumps(config.to_dict(), indent=2))
        return
    run(config, execution, resume=args.resume, recover_generation=args.recover_generation,
        stop_after_step=args.stop_after_step)


if __name__ == '__main__':
    main()

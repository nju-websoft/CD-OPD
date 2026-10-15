from __future__ import annotations

import argparse
import faulthandler
import hashlib
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import importlib
import json
import os
from pathlib import Path
import signal
import shutil
import threading
import time

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer

from .backend import HFBackend
from .digest import ContextualDigestBuilder, DIGEST_SYSTEM
from .config import TrainConfig
from .data import OnlineTrainingStream, load_questions
from .journal import RolloutJournal, model_fingerprint
from .rollout import RelayRollout, independent_rollout
from .update import MetaHandoffUpdater
from .snapshot import seal_snapshot, verify_snapshot, publish_update


PROJECT = Path(__file__).resolve().parents[2]


def emit(event, **values):
    print(json.dumps({"utc": datetime.now(timezone.utc).isoformat(), "event": event, **values},
                     ensure_ascii=False), flush=True)


def save_checkpoint(output, step, student, tokenizer, optimizer, stream, generators, config, rank, world,
                    *, recovery=False, payload=None, execution=None):
    student_sha256 = model_fingerprint(student) if hasattr(student, "state_dict") else None
    if world > 1:
        hashes = [None] * world
        dist.all_gather_object(hashes, student_sha256)
        if len(set(hashes)) != 1:
            raise ValueError("Student replicas differ before checkpoint commit")
    target = output / ("recovery" if recovery else "checkpoints") / f"step_{step}"
    temporary = target.with_name(target.name + ".incomplete")
    if rank == 0:
        if target.exists() or temporary.exists():
            raise FileExistsError(f"Checkpoint output already exists: {target}")
        temporary.mkdir(parents=True)
        student.save_pretrained(temporary / "student", safe_serialization=True)
        tokenizer.save_pretrained(temporary / "student")
        torch.save(optimizer.state_dict(), temporary / "optimizer.pt")
        (temporary / "config.json").write_text(json.dumps(config.to_dict(), indent=2) + "\n")
        if execution is not None:
            (temporary / "execution.json").write_text(json.dumps(execution, indent=2) + "\n")
    if world > 1:
        dist.barrier()
    torch.save({
        "step": step, "stream": stream.state_dict(),
        "student_rng": generators[0].get_state(), "teacher_rng": generators[1].get_state(),
        "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
        "payload": payload,
        "student_sha256": student_sha256,
    }, temporary / f"rank_{rank}.pt")
    if world > 1:
        dist.barrier()
    if rank == 0:
        seal_snapshot(temporary, {
            "step": step, "world_size": world,
            "student_sha256": student_sha256,
            "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in sorted(Path(__file__).parent.glob("*.py"))},
        })
        temporary.replace(target)
        if not recovery:
            (output / "latest_checkpoint.txt").write_text(str(target) + "\n")
        pointer = output / "latest_recovery.incomplete"
        pointer.write_text(str(target) + "\n")
        pointer.replace(output / "latest_recovery.txt")
        archive = os.environ.get("META_HANDOFF_HF_ARCHIVE")
        if archive and not recovery:
            from .checkpoints import apply_storage_policy
            apply_storage_policy(output, target, archive, int(os.environ.get("META_HANDOFF_KEEP_RESUMABLE", "2")))


        for old in (output / "recovery").glob("step_*"):
            if old != target and old.name[5:].isdigit() and (old / "config.json").is_file():
                shutil.rmtree(old)
        emit("checkpoint_saved", rank=rank, step=step, path=str(target), recovery=recovery)
    if world > 1:
        dist.barrier()


def run(config: TrainConfig, resume: Path | None = None, recover_generation: bool = False,
        replay_journal: Path | None = None, bootstrap_journal: Path | None = None):


    faulthandler.register(signal.SIGUSR1)
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    config.validate(world)
    generation_sources = ("backend.py", "constrained.py", "graph.py", "rollout.py", "data.py", "digest.py")
    generation_sha256 = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                         for name in generation_sources}
    if sum((bool(resume), recover_generation, bool(replay_journal), bool(bootstrap_journal))) > 1:
        raise ValueError("Select only one recovery mode")
    from .recovery import JournalReplay
    history = replay_journal or bootstrap_journal
    replay = JournalReplay(history, config, world) if history else None
    output = Path(config.output_dir)
    if not output.is_absolute():
        output = PROJECT / output
    torch.cuda.set_device(config.student_devices[rank])
    if world > 1:


        dist.init_process_group("nccl", device_id=torch.device(f"cuda:{config.student_devices[rank]}"),
                                timeout=timedelta(hours=4))
    if rank == 0:
        if resume:
            saved_snapshot = verify_snapshot(resume, world)
            current_sources = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in sorted(Path(__file__).parent.glob("*.py"))}
            if saved_snapshot.get("source_sha256") != current_sources:
                raise ValueError("Checkpoint method source differs; start a separate experiment for changed code")
        if bootstrap_journal and (any(output.glob("metrics_rank_*.jsonl"))
                                  or (output / "latest_checkpoint.txt").exists()
                                  or (output / "latest_recovery.txt").exists()
                                  or (output / "audit").exists()):
            raise FileExistsError("Bootstrap is a NEW optimization run; preserve and separate prior update records first")
        if recover_generation:
            prior = json.loads((output / "run_manifest.json").read_text())
            if prior["config"] != config.to_dict():
                raise ValueError("Generation recovery requires the exact same configuration")
            if any(output.glob("metrics_rank_*.jsonl")) or (output / "latest_checkpoint.txt").exists():
                raise ValueError("Student already updated; recover from a saved checkpoint")
        if (output / "run_manifest.json").exists() and resume is None and not recover_generation and not replay:
            raise FileExistsError("Output already contains a run; select a new output or an explicit checkpoint resume")
        output.mkdir(parents=True, exist_ok=True)
        manifest = {
            "method": "CD-OPD", "backend": "hf_exact_reference", "config": config.to_dict(),
            "teacher_frozen": True, "meta_gradient_scope": "all trainable student parameters",
            "meta_weight_normalization": "global summary-token-weighted mean=1",
            "opd_normalization": "original global valid-response-token mean",
            "task_gradient_only_for_meta": True, "world_size": world,
            "resume": str(resume) if resume else None,
            "recover_generation": recover_generation,
            "replay_journal": str(replay_journal) if replay_journal else None,
            "bootstrap_journal": str(bootstrap_journal) if bootstrap_journal else None,
            "bootstrap_semantics": ("NEW optimization run: reuse only the first batch from the exact initial Student; "
                                    "do not claim historical post-update snapshot recovery") if bootstrap_journal else None,
            "generation_source_sha256": generation_sha256,
            "checkpoint_archive": os.environ.get("META_HANDOFF_HF_ARCHIVE"),
            "keep_resumable": os.environ.get("META_HANDOFF_KEEP_RESUMABLE"),
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }
        (output / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    if world > 1:
        dist.barrier()
    torch.manual_seed(config.seed)
    tokenizer = AutoTokenizer.from_pretrained(config.student_model, local_files_only=True)
    teacher_tokenizer = AutoTokenizer.from_pretrained(config.teacher_model, local_files_only=True)
    if tokenizer.get_vocab() != teacher_tokenizer.get_vocab() or tokenizer.eos_token_id != teacher_tokenizer.eos_token_id:
        raise ValueError("Relay's shared emitted-token scoring requires identical teacher/student vocabulary")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    student_path = str(resume / "student") if resume else config.student_model
    student = AutoModelForCausalLM.from_pretrained(
        student_path, dtype=torch.float32, local_files_only=True, attn_implementation="sdpa",
    ).to(f"cuda:{config.student_devices[rank]}")
    teacher = AutoModelForCausalLM.from_pretrained(
        config.teacher_model, dtype=torch.bfloat16, local_files_only=True, attn_implementation="sdpa",
    ).to(f"cuda:{config.teacher_devices[rank]}")
    teacher.requires_grad_(False)
    teacher.eval()


    for module in student.modules():
        if isinstance(module, torch.nn.Dropout) and module.p != 0:
            raise ValueError("The exact reference requires student dropout=0")
    if getattr(student.config, "attention_dropout", 0) != 0:
        raise ValueError("The exact reference requires attention_dropout=0")
    if config.gradient_checkpointing:
        student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    student_backend = HFBackend(student, score_chunk_tokens=config.score_chunk_tokens,
                                forward_parameter_dtype=torch.bfloat16)
    teacher_backend = HFBackend(teacher, score_chunk_tokens=config.score_chunk_tokens)
    optimizer = torch.optim.AdamW(
        student.parameters(), lr=config.lr, betas=tuple(config.adam_betas), weight_decay=config.weight_decay,
    )
    updater = MetaHandoffUpdater(student_backend, teacher_backend, optimizer, config, world > 1)
    builder = ContextualDigestBuilder(teacher_backend, tokenizer, config.handoff)
    graph_system = DIGEST_SYSTEM
    os.environ["VERL_OPD_TOKENIZER_PATH"] = config.student_model

    from opd.patches.vllm.speculative_decode import _get_reflection_token_ids, _get_paragraph_boundary_token_ids
    reflection_ids = set(_get_reflection_token_ids(torch.device("cpu")).tolist())
    paragraph_ids = set(_get_paragraph_boundary_token_ids(torch.device("cpu")).tolist())
    if not reflection_ids:
        raise RuntimeError("Relay reflection markers are missing")
    rollout = RelayRollout(student_backend, teacher_backend, tokenizer, config.relay,
                           reflection_ids, paragraph_ids, builder)
    questions, skipped = load_questions(config.train_data, tokenizer, config.relay.max_prompt_tokens)
    stream = OnlineTrainingStream(
        questions, config.handoff_batch_size,
        config.meta.task_batch_size if config.meta.enabled else 0, config.seed, config.epochs,
    )
    student_generator = torch.Generator(device=student_backend.device).manual_seed(config.seed + rank)
    teacher_generator = torch.Generator(device=teacher_backend.device).manual_seed(config.seed + 10000 + rank)
    if resume:
        saved_config = TrainConfig.load(resume / "config.json").to_dict()
        current_config = config.to_dict()
        for key in ("output_dir", "max_steps", "save_every"):
            saved_config.pop(key)
            current_config.pop(key)
        if saved_config != current_config:
            raise ValueError("Resume changes the method, model, data, topology or optimization configuration")
        optimizer.load_state_dict(torch.load(resume / "optimizer.pt", map_location=student_backend.device, weights_only=False))
        state = torch.load(resume / f"rank_{rank}.pt", map_location="cpu", weights_only=False)
        snapshot = json.loads((resume / "snapshot_manifest.json").read_text())
        if model_fingerprint(student) != snapshot["student_sha256"]:
            raise ValueError("Reloaded Student is not identical to its saved checkpoint")
        if state["step"] != snapshot["step"]:
            raise ValueError("Rank state and checkpoint step disagree")
        stream.load_state_dict(state["stream"])
        student_generator.set_state(state["student_rng"])
        teacher_generator.set_state(state["teacher_rng"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        if state.get("payload"):
            publish_update(output, rank, state["payload"]["metrics"], state["payload"]["audit"],
                           replay=state["payload"].get("replay", False))
        emit("resume_verified", rank=rank, step=state["step"], student_sha256=snapshot["student_sha256"])
    module, name = config.reward_function.split(":", 1)
    reward = getattr(importlib.import_module(module), name)
    emit("ready", rank=rank, training_questions=len(questions), filtered_prompts=skipped,
         reflection_ids=sorted(reflection_ids), world_size=world)
    progress = {"rank": rank, "step": stream.step, "phase": "ready", "item": 0}
    heartbeat_stop = threading.Event()

    def heartbeat():
        while not heartbeat_stop.wait(30):
            emit("heartbeat", **dict(progress))

    threading.Thread(target=heartbeat, name="meta-progress", daemon=True).start()
    metrics_file = output / f"metrics_rank_{rank}.jsonl"
    last_saved = stream.step if resume else None
    while config.max_steps is None or stream.step < config.max_steps:
        try:
            handoff_questions, task_questions = next(stream)
        except StopIteration:
            break
        step = stream.step
        started = time.monotonic()

        emit("rollout_start", step=step, rank=rank)
        progress.update(step=step, phase="handoff", total=len(handoff_questions[rank::world]))
        identity_config = config.to_dict()
        for key in ("output_dir", "save_every", "max_steps"):
            identity_config.pop(key)
        identity = {
            "student_sha256": model_fingerprint(student), "config": identity_config,
            "graph_extraction_instructions": graph_system if builder else None,
            "generation_source_sha256": generation_sha256,
            "corpus": stream.fingerprint,
            "handoff_ids": [q.id for q in handoff_questions],
            "independent_ids": [q.id for q in task_questions],
        }
        replay_limit = 1 if bootstrap_journal else replay.steps if replay else 0
        replaying = replay is not None and step <= replay_limit
        if replaying:
            identity = replay.identity(step, rank, identity)
        journal = RolloutJournal(output, step, rank, identity, (student_generator, teacher_generator))
        handoff = []
        for index, question in enumerate(handoff_questions[rank::world]):
            progress.update(item=index + 1)
            try:
                cached = journal.load("handoff", index, question.id)
                if cached:
                    trace, _ = cached
                else:
                    if replaying:
                        raise FileNotFoundError("Completed update is missing a required handoff journal item")
                    trace = rollout.generate(question.id, question.text, question.prompt_ids,
                                             student_generator, teacher_generator)
                    journal.save("handoff", index, trace)
            except Exception:
                if builder and builder.last_attempt:
                    (output / f"failed_graph_step_{step}_rank_{rank}.json").write_text(
                        json.dumps(builder.last_attempt, ensure_ascii=False, indent=2) + "\n")
                raise
            handoff.append(trace)
            emit("handoff_rollout", step=step, rank=rank, item=index, response_tokens=len(trace.response_ids),
                 summary_segments=len(trace.summary_spans), stop_reason=trace.stop_reason, cached=bool(cached))
        tasks = []
        progress.update(phase="independent", item=0,
                        total=len(task_questions[rank::world]) * config.meta.task_samples)
        for question in task_questions[rank::world]:
            for sample in range(config.meta.task_samples):
                progress.update(item=len(tasks) + 1)
                cached = journal.load("independent", len(tasks), question.id)
                if cached:
                    trace, score = cached
                else:
                    if replaying:
                        raise FileNotFoundError("Completed update is missing a required independent journal item")
                    trace = independent_rollout(student_backend, tokenizer, question.id, question.prompt_ids,
                                                config.meta.task_max_response_tokens, student_generator)
                    score = reward(
                        data_source=question.data_source,
                        solution_str=tokenizer.decode(trace.response_ids, skip_special_tokens=True),
                        ground_truth=question.ground_truth,
                    )
                    if isinstance(score, dict):
                        score = score["score"]
                    journal.save("independent", len(tasks), trace, float(score))
                tasks.append((trace, float(score)))
                emit("independent_rollout", step=step, rank=rank, sample=sample, score=float(score),
                     response_tokens=len(trace.response_ids))
        emit("meta_update_start", step=step, rank=rank)
        progress.update(phase="meta_update")
        metrics, segment_audits = updater.update(handoff, tasks)
        metrics.update(step=step, rank=rank, wall_seconds=time.monotonic() - started)
        if replaying and not bootstrap_journal:
            verified = replay.verify_update(step, rank, metrics, model_fingerprint(student))
            emit("replay_snapshot_verified", step=step, rank=rank, student_sha256=verified)
            metrics["replayed"] = True
        if replaying and bootstrap_journal:
            actual_hash = model_fingerprint(student)
            hashes = [None] * world
            if world > 1:
                dist.all_gather_object(hashes, actual_hash)
            else:
                hashes[0] = actual_hash
            if len(set(hashes)) != 1:
                raise ValueError("Bootstrapped Student replicas differ after their first update")
            emit("bootstrap_update_complete", step=step, rank=rank, student_sha256=actual_hash,
                 historical_post_update_sha256=replay.plan["identities"]["2"][str(rank)]["student_sha256"],
                 historical_metrics=replay.plan["metrics"][str(rank)]["1"], current_metrics=metrics)
        audit = {
            "step": step, "snapshot_step": step - 1,
            "handoff_question_ids": [q.id for q in handoff_questions],
            "task_question_ids": [q.id for q in task_questions],
            "handoff_trajectories": [asdict(trace) for trace in handoff],
            "task_trajectories": [{"trajectory": asdict(trace), "reward": score} for trace, score in tasks],
            "meta_segments": segment_audits,
        }
        regular = step % config.save_every == 0 or step == config.max_steps
        progress.update(phase="checkpoint" if regular else "recovery_checkpoint")
        payload = {"metrics": metrics, "audit": audit, "replay": bool(replaying and not bootstrap_journal)}


        save_checkpoint(output, step, student, tokenizer, optimizer, stream,
                        (student_generator, teacher_generator), config, rank, world,
                        recovery=not regular, payload=payload)
        publish_update(output, rank, metrics, audit, replay=payload["replay"])
        emit("update_complete", **metrics)
        if regular:
            last_saved = step
        if replaying and step == replay_limit:
            emit("bootstrap_complete" if bootstrap_journal else "replay_complete", rank=rank, step=step)
        if step == config.max_steps:
            break
    completed_steps = stream.step
    if completed_steps and last_saved != completed_steps:
        save_checkpoint(output, completed_steps, student, tokenizer, optimizer, stream,
                        (student_generator, teacher_generator), config, rank, world)
    emit("run_complete", rank=rank, completed_steps=completed_steps)
    heartbeat_stop.set()
    if world > 1:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--check-config", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--recover-generation", action="store_true",
                        help="Recover interrupted generation before the first Student update")
    parser.add_argument("--replay-journal", type=Path,
                        help="Replay completed updates with strict historical Student hash and metric verification")
    parser.add_argument("--bootstrap-journal", type=Path,
                        help="NEW run from the exact initial Student and cached first batch; regenerate later batches")
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args()
    config = TrainConfig.load(args.config)
    if args.max_steps is not None:
        config.max_steps = args.max_steps
    if args.check_config:
        config.validate(len(config.student_devices))
        print(json.dumps(config.to_dict(), ensure_ascii=False, indent=2))
        return
    run(config, args.resume, args.recover_generation, args.replay_journal, args.bootstrap_journal)


if __name__ == "__main__":
    main()

import argparse
import json
import os
from pathlib import Path
import threading
import time
import traceback
from multiprocessing.connection import Client


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--socket', required=True)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    parent = os.getppid()

    def monitor():
        while True:
            time.sleep(2)
            if os.getppid() != parent:
                os._exit(1)

    threading.Thread(target=monitor, daemon=True).start()
    connection = Client(args.socket, family='AF_UNIX', authkey=bytes.fromhex(os.environ.pop('META_VLLM_AUTH')))
    try:
        config = json.loads(Path(args.config).read_text())
        relay = config['native_relay']


        os.environ.update(VERL_OPD_ROLLOUT_MODE='relay', VERL_OPD_COLLECT_DRAFT_PROBS='1',
                          VERL_OPD_TOKENIZER_PATH=config['tokenizer'],
                          VERL_OPD_SOURCE_METRICS='1', RELAY_OPD_EXPORT_STUDENT_ACTION='0',
                          RELAY_OPD_TRIGGER_TOPK=str(relay['trigger_topk']),
                          RELAY_OPD_MAX_TAKEOVERS='1',
                          RELAY_OPD_PARAGRAPHS_PER_TAKEOVER=str(relay['paragraphs_per_takeover']),
                          RELAY_OPD_MAX_TAKEOVER_TOKENS=str(relay['max_takeover_tokens'] - 1))
        from opd.patches.vllm import speculative_decode as patch
        from .native_protocol import NativeTraceStore
        store = NativeTraceStore()


        os.environ['VERL_OPD_MASK_IPC_SOCKET'] = 'native-in-process-callback'
        patch._send_mask_chunk_ipc = store.record
        patch._send_stop_event_ipc = store.stop
        patch.apply_patches()
        from .native_replay import install_prefix_replay
        replays = {}
        def reset_private_state(request_id):
            store.stops.pop(request_id, None)
            for state in (patch._TAKEOVER_TOKENS_REMAINING, patch._TAKEOVER_PARAGRAPHS_REMAINING,
                          patch._COMPLETED_TAKEOVERS):
                state.pop(request_id, None)
        install_prefix_replay(replays, patch._strip_vllm_suffix, reset_private_state)
        from vllm import LLM, SamplingParams
        llm = LLM(model=config['model'], tokenizer=config['tokenizer'], dtype='bfloat16',
                  seed=config['engine_seed'],
                  speculative_config={'method': 'draft_model', 'model': config['draft_model'],
                                      'num_speculative_tokens': 4, 'draft_tensor_parallel_size': 1},
                  max_model_len=config['max_model_len'], max_num_seqs=config['max_num_seqs'],
                  gpu_memory_utilization=config['memory_fraction'], max_num_batched_tokens=8192,
                  enable_prefix_caching=True, enable_sleep_mode=True,
                  enforce_eager=config['enforce_eager'], distributed_executor_backend='uni',
                  generation_config='vllm', disable_log_stats=True,
                  worker_extension_cls='opd.contextual_digest.native_weights.NativeWeightReloadExtension')
        engine = llm.llm_engine
        pending = {}
        serial = 0
        connection.send({'ok': True, 'result': {'ready': True, 'backend': 'original_vllm_relay',
                                              'total_takeover_budget': relay['max_takeovers'],
                                              'call_return_budget': 1, 'num_speculative_tokens': 4}})
        while True:
            message = connection.recv()
            command = message['command']
            if command == 'close':
                connection.send({'ok': True})
                break
            if command == 'relay_add':
                accepted = []
                for request in message['requests']:
                    serial += 1
                    request_id = f'native-{serial}'
                    prefix = request['prompt_ids']
                    replay = prefix[-1] if request.get('resume_handoff') else None
                    if replay is not None and len(prefix) < 2:
                        raise ValueError('Native handoff needs a nonempty replay prefill')


                    prompt = prefix[:-1] if replay is not None else prefix
                    params = SamplingParams(max_tokens=request['max_tokens'] + int(replay is not None) + 1,
                                            temperature=request.get('temperature', 1.0), top_p=1.0, top_k=-1,
                                            seed=request.get('seed'))
                    if replay is not None:
                        replays[request_id] = replay
                    engine.add_request(request_id, {'prompt_token_ids': prompt}, params)
                    pending[request_id] = {**request, 'replay_token': replay}
                    accepted.append(request['key'])
                result = {'accepted': accepted, 'pending': len(pending)}
            elif command == 'relay_advance':
                finished = []
                for _ in range(message.get('max_steps', 8)):
                    if not pending:
                        break
                    for generation in engine.step():
                        if not generation.finished:
                            continue
                        request_id = generation.request_id
                        request = pending.pop(request_id)
                        output = generation.outputs[0]
                        result = store.consume(request_id, output.token_ids,
                                               limit=request['max_tokens'], replay_token=request['replay_token'],
                                               finish_reason=output.finish_reason)
                        finished.append({'key': request['key'], **result})
                        replays.pop(request_id, None)
                        for state in (patch._TAKEOVER_TOKENS_REMAINING, patch._TAKEOVER_PARAGRAPHS_REMAINING,
                                      patch._COMPLETED_TAKEOVERS):
                            state.pop(request_id, None)
                    if finished:
                        break
                result = {'finished': finished, 'pending': len(pending)}
            elif command == 'relay_finish':
                if pending or engine.has_unfinished_requests():
                    raise RuntimeError('Native rollout has unfinished requests')
                if store.tokens or store.stops:
                    raise RuntimeError('Native rollout has unconsumed provenance')
                result = patch.pop_opd_stats()
            elif command in ('sleep', 'wake', 'load_weights'):
                if pending:
                    raise RuntimeError('Cannot mutate native engine while requests are active')
                if command == 'sleep':
                    llm.sleep(level=1)
                    result = None
                elif command == 'wake':
                    llm.wake_up()
                    result = None
                else:
                    result = llm.collective_rpc('load_handoff_weights', args=(message['path'], message['sha256']))
                    llm.reset_prefix_cache()
            else:
                raise ValueError(f'Unknown native command: {command}')
            connection.send({'ok': True, 'result': result})
    except BaseException:
        error = traceback.format_exc()
        print(error, flush=True)
        try:
            connection.send({'ok': False, 'error': error})
        except (EOFError, BrokenPipeError):
            pass
        raise
    finally:
        connection.close()


if __name__ == '__main__':
    main()

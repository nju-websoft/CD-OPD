import argparse
import json
import os
from pathlib import Path
import traceback
import threading
import time
from multiprocessing.connection import Client


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--socket', required=True)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    parent_pid = os.getppid()

    def monitor_parent():
        while True:
            time.sleep(2)
            if os.getppid() != parent_pid:
                print(json.dumps({'event': 'inference_parent_exited', 'original_parent': parent_pid,
                                  'current_parent': os.getppid()}), flush=True)
                os._exit(1)

    threading.Thread(target=monitor_parent, daemon=True).start()
    connection = Client(args.socket, family='AF_UNIX', authkey=bytes.fromhex(os.environ.pop('META_VLLM_AUTH')))
    try:
        from vllm import LLM, SamplingParams
        config = json.loads(Path(args.config).read_text())
        llm = LLM(model=config['model'], tokenizer=config['tokenizer'], dtype='bfloat16',
                  max_model_len=config['max_model_len'], max_num_seqs=config['max_num_seqs'],
                  gpu_memory_utilization=config['memory_fraction'], max_num_batched_tokens=8192,
                  enable_prefix_caching=True, enable_sleep_mode=True,
                  enforce_eager=config['enforce_eager'], distributed_executor_backend='uni',
                  generation_config='vllm', disable_log_stats=True,
                  logits_processors=['opd.contextual_digest.vllm_graph:VLLMGraphProcessor']
                  if config['graph_constraints'] else None,
                  worker_extension_cls='opd.contextual_digest.vllm_weights.WeightReloadExtension')
        connection.send({'ok': True, 'result': {'ready': True}})
        while True:
            message = connection.recv()
            command = message['command']
            if command == 'close':
                connection.send({'ok': True})
                break
            if command == 'generate':
                requests = message['requests']
                params = [SamplingParams(
                    max_tokens=r['max_tokens'], temperature=r.get('temperature', 1.0), top_p=1.0, top_k=-1,
                    seed=r.get('seed'), logprobs=r.get('logprobs'),
                    extra_args={**({'meta_graph_schema': r['graph_schema']} if r.get('graph_schema') else {}),
                                **({'relay_paragraph_stop': r['relay_paragraph_stop']}
                                   if r.get('relay_paragraph_stop') is not None else {})} or None,
                ) for r in requests]
                generated = llm.generate([{'prompt_token_ids': r['prompt_ids']} for r in requests], params,
                                         use_tqdm=False)
                result = []
                for request, generation in zip(requests, generated, strict=True):
                    output = generation.outputs[0]
                    topk = request.get('logprobs')
                    ids, reason = list(output.token_ids), output.finish_reason
                    rule = request.get('relay_paragraph_stop')
                    if rule is not None:
                        count, boundaries = 0, set(rule['token_ids'])
                        for index, token in enumerate(ids):
                            count += int(token in boundaries)
                            if count >= rule['count']:


                                ids, reason = ids[:index + 1], 'relay_paragraph_boundary'
                                break
                    result.append({'token_ids': ids, 'finish_reason': reason,
                                   'decoded_tokens': len(output.token_ids),
                                   'topk': [[token for token, value in item.items()
                                             if value.rank is not None and value.rank <= topk]
                                            for item in output.logprobs[:len(ids)]] if topk else None})
            elif command == 'sleep':
                llm.sleep(level=1)
                result = None
            elif command == 'wake':
                llm.wake_up()
                result = None
            elif command == 'load_weights':
                result = llm.collective_rpc('load_handoff_weights', args=(message['path'], message['sha256']))
                llm.reset_prefix_cache()
            else:
                raise ValueError(f'Unknown inference command: {command}')
            connection.send({'ok': True, 'result': result})
    except BaseException:
        error = traceback.format_exc()
        print(error, flush=True)
        try:
            connection.send({'ok': False, 'error': error})
        except (BrokenPipeError, EOFError):
            pass
        raise
    finally:
        connection.close()


if __name__ == '__main__':
    main()

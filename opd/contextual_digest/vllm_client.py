import json
import os
from pathlib import Path
import secrets
import signal
import subprocess
import sys
import time
from multiprocessing.connection import Listener

from .compile_cache import inference_cache_environment


class VLLMClient:
    def __init__(self, model, tokenizer, gpu, runtime, *, max_model_len, max_num_seqs=16,
                 memory_fraction=0.45, graph_constraints=False, enforce_eager=True,
                 worker_module='opd.contextual_digest.vllm_worker', worker_options=None):
        runtime = Path(runtime).resolve()
        runtime.mkdir(parents=True, exist_ok=True)
        self.address = f'/tmp/meta-vllm-{os.getpid()}-{gpu}-{secrets.token_hex(4)}.sock'
        authkey = secrets.token_bytes(32)
        listener = Listener(self.address, family='AF_UNIX', authkey=authkey)
        env = {k: v for k, v in os.environ.items()
               if k not in {'RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT',
                            'GROUP_RANK', 'ROLE_RANK', 'ROLE_WORLD_SIZE'}
               and not k.startswith(('TORCHELASTIC_', 'NCCL_'))}
        env.update(CUDA_VISIBLE_DEVICES=str(gpu), META_VLLM_AUTH=authkey.hex(),
                   META_GRAPH_TOKENIZER_PATH=tokenizer, VLLM_ENABLE_V1_MULTIPROCESSING='0',
                   VLLM_USE_FLASHINFER_SAMPLER='0', VLLM_ALLREDUCE_USE_FLASHINFER='0',
                   TOKENIZERS_PARALLELISM='false', VLLM_WORKER_MULTIPROC_METHOD='spawn')
        config = dict(model=model, tokenizer=tokenizer, max_model_len=max_model_len,
                      max_num_seqs=max_num_seqs, memory_fraction=memory_fraction,
                      graph_constraints=graph_constraints, enforce_eager=enforce_eager)
        if worker_options:
            config.update(worker_options)
        cache_environment, cache_record = inference_cache_environment(config, worker_module, env)
        env.update(cache_environment)
        (runtime / f'engine_gpu_{gpu}.cache.json').write_text(json.dumps(cache_record, indent=2) + '\n')
        config_path = runtime / f'engine_gpu_{gpu}.json'
        config_path.write_text(json.dumps(config, indent=2) + '\n')
        self.log_path = runtime / f'engine_gpu_{gpu}.log'
        with self.log_path.open('ab') as log:
            self.process = subprocess.Popen([sys.executable, '-m', worker_module,
                                             '--socket', self.address, '--config', str(config_path)],
                                            env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                            start_new_session=True)


        listener._listener._socket.settimeout(1)
        deadline = time.monotonic() + 60
        try:
            while True:
                try:
                    self.connection = listener.accept()
                    break
                except TimeoutError:
                    if self.process.poll() is not None or time.monotonic() > deadline:
                        raise RuntimeError(f'Inference worker failed to connect; see {self.log_path}')
        finally:
            listener.close()
        self._receive(timeout=900)

    def _receive(self, timeout=900):
        deadline = time.monotonic() + timeout
        while not self.connection.poll(1):
            if self.process.poll() is not None or time.monotonic() > deadline:
                raise self._worker_failure('stopped or timed out')
        try:
            response = self.connection.recv()
        except (EOFError, ConnectionResetError) as error:
            raise self._worker_failure(type(error).__name__) from error
        if not response.get('ok'):
            raise RuntimeError(response.get('error', 'Inference worker failed'))
        return response.get('result')

    def _worker_failure(self, reason):
        try:
            code = self.process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            code = None
        signame = signal.Signals(-code).name if code is not None and code < 0 else None
        record = {'event': 'inference_worker_failure', 'reason': reason,
                  'worker_pid': self.process.pid, 'returncode': code, 'signal': signame,
                  'log': str(self.log_path)}
        self.log_path.with_suffix('.failure.json').write_text(json.dumps(record, indent=2) + '\n')
        return RuntimeError(f'Inference worker failure: {json.dumps(record)}')

    def call(self, command, **kwargs):
        self.connection.send({'command': command, **kwargs})
        return self._receive()

    def generate(self, requests):
        return self.call('generate', requests=requests) if requests else []

    def sleep(self):
        return self.call('sleep')

    def wake(self):
        return self.call('wake')

    def close(self):
        try:
            if self.process.poll() is None:
                self.call('close')
        finally:
            if self.process.poll() is None:
                os.killpg(self.process.pid, signal.SIGTERM)
            self.process.wait(timeout=30)
            if hasattr(self, 'connection'):
                self.connection.close()
            Path(self.address).unlink(missing_ok=True)

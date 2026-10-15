import hashlib
import json
from pathlib import Path


def inference_cache_environment(config, worker_module, environment):


    identity = {'version': 1, 'worker_module': worker_module,
                'engine_config': {k: v for k, v in config.items() if k != 'engine_seed'},
                'model_configs': {}}
    for key in ('model', 'draft_model', 'tokenizer'):
        if config.get(key):
            path = Path(config[key]) / 'config.json'
            identity['model_configs'][key] = hashlib.sha256(path.read_bytes()).hexdigest()
    encoded = json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()
    namespace = hashlib.sha256(encoded).hexdigest()
    cache_root = Path(environment.get('VLLM_CACHE_ROOT', str(Path.home() / '.cache/vllm')))
    root = cache_root / 'meta_handoff_engines' / namespace
    overrides = {'VLLM_CACHE_ROOT': str(root / 'vllm'),
                 'TRITON_CACHE_DIR': str(root / 'triton'),
                 'TORCHINDUCTOR_CACHE_DIR': str(root / 'inductor')}
    return overrides, {'namespace': namespace, 'identity': identity, 'environment': overrides}

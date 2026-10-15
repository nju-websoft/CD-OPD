import hashlib
from pathlib import Path

import torch


class WeightReloadExtension:
    def load_handoff_weights(self, path, expected_sha256):
        path = Path(path)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected_sha256:
            raise ValueError('Student weight transfer checksum mismatch')
        weights = torch.load(path, map_location='cpu', weights_only=True)
        loaded = self.model_runner.model.load_weights(weights.items())
        parameters = dict(self.model_runner.model.named_parameters(remove_duplicate=False))
        checked = 0
        for name, expected in weights.items():
            target_name, offset = name, 0
            for component in ('q', 'k', 'v'):
                suffix = f'.self_attn.{component}_proj.weight'
                if name.endswith(suffix):
                    prefix = name[:-len(suffix)] + '.self_attn.'
                    target_name = prefix + 'qkv_proj.weight'
                    if component in ('k', 'v'):
                        offset += weights[prefix + 'q_proj.weight'].shape[0]
                    if component == 'v':
                        offset += weights[prefix + 'k_proj.weight'].shape[0]
            for component in ('gate', 'up'):
                suffix = f'.mlp.{component}_proj.weight'
                if name.endswith(suffix):
                    prefix = name[:-len(suffix)] + '.mlp.'
                    target_name = prefix + 'gate_up_proj.weight'
                    if component == 'up':
                        offset = weights[prefix + 'gate_proj.weight'].shape[0]
            if target_name not in parameters:
                raise ValueError(f'Cannot verify transferred Student parameter {name}')
            actual_parameter = parameters[target_name]
            actual_view = actual_parameter[offset:offset + expected.shape[0]]
            if actual_view.shape != expected.shape or not torch.equal(
                actual_view, expected.to(device=actual_view.device, dtype=actual_view.dtype)
            ):
                raise ValueError(f'Student weight reload verification failed: {name}')
            checked += 1
        torch.cuda.synchronize()
        return {'file_sha256': actual, 'input_tensors': len(weights), 'loaded_parameters': len(loaded),
                'verified_input_tensors': checked}

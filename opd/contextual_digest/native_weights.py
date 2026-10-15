from types import SimpleNamespace

from .vllm_weights import WeightReloadExtension


class NativeWeightReloadExtension:
    def load_handoff_weights(self, path, expected_sha256):
        runner = self.model_runner
        drafter = getattr(getattr(runner, 'drafter', None), 'model', None)
        if drafter is None or drafter is runner.model:
            raise ValueError('Native Relay must expose a separate Student draft model')
        proxy = SimpleNamespace(model_runner=SimpleNamespace(model=drafter))
        result = WeightReloadExtension.load_handoff_weights(proxy, path, expected_sha256)
        result['reload_target'] = 'student_draft_model'
        return result

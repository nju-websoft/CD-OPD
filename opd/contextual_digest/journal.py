from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import torch

from .graph import SummarySpan
from .rollout import Trajectory


def model_fingerprint(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
        raw = tensor.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy()
        digest.update(memoryview(raw).cast("B"))
    return digest.hexdigest()


class RolloutJournal:
    def __init__(self, output, step, rank, identity, generators):
        serialized = json.dumps(identity, ensure_ascii=False, sort_keys=True)
        self.identity = hashlib.sha256(serialized.encode()).hexdigest()
        self.directory = Path(output) / "rollout_journal" / f"step_{step}" / f"rank_{rank}" / self.identity[:16]
        self.directory.mkdir(parents=True, exist_ok=True)
        record = self.directory / "identity.json"
        if record.exists() and json.loads(record.read_text()) != json.loads(serialized):
            raise ValueError("Rollout journal snapshot identity mismatch")
        record.write_text(serialized + "\n")
        self.generators = generators

    def _path(self, kind, index):
        if kind not in {"handoff", "independent"} or index < 0:
            raise ValueError("Invalid rollout journal item")
        return self.directory / f"{kind}_{index:05d}.pt"

    def save(self, kind, index, trace, reward=None):
        trace.validate()
        path = self._path(kind, index)
        if path.exists():
            raise FileExistsError("A completed rollout journal item cannot be overwritten")
        value = {
            "identity": self.identity, "trace": asdict(trace), "reward": reward,
            "generator_states": [g.get_state() for g in self.generators],
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        temporary = path.with_suffix(".incomplete")
        torch.save(value, temporary)
        temporary.replace(path)

    def load(self, kind, index, question_id):
        path = self._path(kind, index)
        if not path.exists():
            return None
        value = torch.load(path, map_location="cpu", weights_only=False)
        if value["identity"] != self.identity or value["trace"]["question_id"] != question_id:
            raise ValueError("Cached rollout belongs to a different snapshot or question")
        raw = dict(value["trace"])
        raw["summary_spans"] = [SummarySpan(**span) for span in raw["summary_spans"]]
        trace = Trajectory(**raw)
        trace.validate()
        for generator, state in zip(self.generators, value["generator_states"], strict=True):
            generator.set_state(state)
        torch.set_rng_state(value["torch_rng"])
        if value["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(value["cuda_rng"])
        return trace, value["reward"]

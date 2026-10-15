import json
from pathlib import Path


class JournalReplay:
    def __init__(self, path, config, world):
        self.path = Path(path)
        self.plan = json.loads(self.path.read_text())
        if self.plan["config"] != config.to_dict() or self.plan["world_size"] != world:
            raise ValueError("Journal replay requires the exact original configuration and topology")
        self.steps = self.plan["completed_steps"]
        if not isinstance(self.steps, int) or self.steps < 1:
            raise ValueError("Replay needs at least one completed update")
        for step in range(1, self.steps + 2):
            if set(self.plan["identities"][str(step)]) != {str(r) for r in range(world)}:
                raise ValueError("Replay needs both ranks' snapshot evidence through the next step")

    def identity(self, step, rank, actual):
        actual = json.loads(json.dumps(actual))
        expected = self.plan["identities"][str(step)][str(rank)]


        for key in expected.keys() | actual.keys():
            if key != "generation_source_sha256" and actual.get(key) != expected.get(key):
                raise ValueError(f"Replay step {step} rank {rank} differs in {key}")
        return expected

    def verify_update(self, step, rank, metrics, student_sha256):
        expected_hash = self.plan["identities"][str(step + 1)][str(rank)]["student_sha256"]
        if student_sha256 != expected_hash:
            raise ValueError(f"Replayed Student after step {step} is not bitwise identical; refusing stale rollouts")
        expected = self.plan["metrics"][str(rank)][str(step)]
        for key, value in expected.items():
            if key not in {"wall_seconds", "replayed"} and metrics.get(key) != value:
                raise ValueError(f"Replay metrics differ at step {step}, {key}")
        return expected_hash

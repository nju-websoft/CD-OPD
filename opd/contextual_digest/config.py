from dataclasses import asdict, dataclass, field
from pathlib import Path
import json
import math


@dataclass
class RelayConfig:
    trigger_topk: int = 5
    max_takeovers: int = 2
    paragraphs_per_takeover: int = 3
    max_takeover_tokens: int = 256
    max_prompt_tokens: int = 2048
    max_response_tokens: int = 16384
    temperature: float = 1.0
    top_p: float = 1.0


@dataclass
class HandoffConfig:
    enabled: bool = True
    protocol: str = 'contextual_digest'
    dependency_graph: bool = True
    graph_max_new_tokens: int = 2048
    max_summary_tokens: int = 1024
    max_nodes: int = 32
    graph_max_model_len: int = 34817


@dataclass
class MetaConfig:
    enabled: bool = True
    task_batch_size: int = 8
    task_samples: int = 2
    task_max_response_tokens: int = 16384

    weight_strength: float = 1.0


@dataclass
class TrainConfig:
    student_model: str = "models/Qwen3-0.6B"
    teacher_model: str = "models/Qwen3-4B-Instruct-2507"
    train_data: str = "data/train.parquet"
    output_dir: str = "runs/cd_opd_0.6b"
    student_devices: list[int] = field(default_factory=lambda: [0, 1])
    teacher_devices: list[int] = field(default_factory=lambda: [2, 3])
    handoff_batch_size: int = 128
    seed: int = 42
    epochs: int = 1
    max_steps: int | None = None
    save_every: int = 5
    lr: float = 1e-6
    weight_decay: float = 0.01
    adam_betas: tuple[float, float] = (0.9, 0.999)
    clip_ratio: float = 0.2
    max_grad_norm: float = 1.0
    score_chunk_tokens: int = 128
    gradient_checkpointing: bool = True
    reward_function: str = "opd.reward.math_reward:compute_score"
    relay: RelayConfig = field(default_factory=RelayConfig)
    handoff: HandoffConfig = field(default_factory=HandoffConfig)
    meta: MetaConfig = field(default_factory=MetaConfig)

    def validate(self, world_size=1):
        if not self.handoff.enabled or not self.handoff.dependency_graph or not self.meta.enabled:
            raise ValueError('This release contains the complete CD-OPD method only')
        if self.meta.weight_strength != 1.0:
            raise ValueError('The complete paper method uses unmodified normalized meta weights')
        strength = self.meta.weight_strength
        if (isinstance(strength, bool) or not isinstance(strength, (int, float))
                or not math.isfinite(strength) or not 0 <= strength <= 1):
            raise ValueError('Meta weight_strength must be finite and between 0 and 1')
        if self.handoff.protocol not in {'contextual_digest'}:
            raise ValueError('Unknown handoff protocol')
        if self.meta.task_samples < 2:
            raise ValueError("Online meta distillation requires at least two independent samples per question")
        if self.meta.enabled and not self.handoff.enabled:
            raise ValueError("Meta reweighting requires handoff summaries")
        if self.handoff_batch_size < world_size or self.meta.task_batch_size < world_size:
            raise ValueError("Both training batches must provide at least one question per rank")
        if len(self.student_devices) != world_size or len(self.teacher_devices) != world_size:
            raise ValueError("Provide one student and one frozen-teacher device per process")
        if len(set(self.student_devices + self.teacher_devices)) != 2 * world_size:
            raise ValueError("Student and teacher devices must be distinct")
        if self.relay.top_p != 1.0 or self.relay.temperature != 1.0:
            raise ValueError("The reference preserves Relay's temperature=1 and top_p=1")
        if self.relay.trigger_topk < 1 or self.relay.max_takeovers < 1 or self.relay.paragraphs_per_takeover < 0:
            raise ValueError("Invalid Relay trigger/return settings")
        positive = [self.relay.max_takeover_tokens, self.relay.max_prompt_tokens, self.relay.max_response_tokens,
                    self.handoff.graph_max_new_tokens, self.handoff.max_summary_tokens, self.handoff.max_nodes,
                    self.meta.task_max_response_tokens, self.handoff_batch_size, self.meta.task_batch_size,
                    self.lr, self.max_grad_norm, self.score_chunk_tokens, self.epochs, self.save_every]
        if any(x <= 0 for x in positive) or (self.max_steps is not None and self.max_steps < 1):
            raise ValueError("Token, batch, update and optimization budgets must be positive")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def load(cls, path):
        import os
        values = json.loads(Path(path).read_text())
        for name in ('student_model', 'teacher_model', 'train_data', 'output_dir'):
            if name in values:
                values[name] = os.path.expanduser(os.path.expandvars(values[name]))
                if '$' in values[name]:
                    raise ValueError(f'Unresolved environment variable in {name}')
        for name, kind in (("relay", RelayConfig), ("handoff", HandoffConfig), ("meta", MetaConfig)):
            values[name] = kind(**values.get(name, {}))
        return cls(**values)

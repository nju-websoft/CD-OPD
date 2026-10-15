from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import math

import torch
import torch.distributed as dist

from .graph import SummarySpan


def relative_advantages(rewards: torch.Tensor) -> torch.Tensor:

    if rewards.ndim != 2 or rewards.shape[1] < 2 or not torch.isfinite(rewards).all():
        raise ValueError("Task rewards must be finite [questions, samples>=2]")
    rewards = rewards.detach().to(torch.float64)
    others = (rewards.sum(dim=1, keepdim=True) - rewards) / (rewards.shape[1] - 1)
    advantages = rewards - others
    ties = (rewards == rewards[:, :1]).all(dim=1, keepdim=True)
    return torch.where(ties, torch.zeros_like(advantages), advantages).detach()


def task_policy_loss(log_probs: torch.Tensor, advantage: float | torch.Tensor) -> torch.Tensor:

    if log_probs.ndim != 1 or log_probs.numel() == 0:
        raise ValueError("Task response must have at least one token")
    advantage = torch.as_tensor(advantage, device=log_probs.device, dtype=log_probs.dtype).detach()
    return -advantage * log_probs.mean()


def gradient_dot(first: Sequence[torch.Tensor | None], second: Sequence[torch.Tensor | None]) -> torch.Tensor:

    if len(first) != len(second):
        raise ValueError("Gradient parameter lists differ")
    reference = next((g for g in (*first, *second) if g is not None), None)
    if reference is None:
        return torch.tensor(0.0, dtype=torch.float64)
    result = torch.zeros((), device=reference.device, dtype=torch.float64)
    for left, right in zip(first, second, strict=True):
        if left is None or right is None:
            continue
        if left.shape != right.shape or left.device != right.device:
            raise ValueError("Gradient tensors are not aligned")
        a, b = left.detach().reshape(-1), right.detach().reshape(-1)
        for start in range(0, a.numel(), 1 << 20):
            result += torch.dot(a[start:start + (1 << 20)].double(), b[start:start + (1 << 20)].double())
    if not torch.isfinite(result):
        raise FloatingPointError("Nonfinite gradient alignment")
    return result.detach()


def segment_alignments(token_losses, spans: Sequence[SummarySpan], parameters, task_gradients) -> torch.Tensor:
    validate_spans(spans, len(token_losses))
    scores = []
    for span in spans:
        segment_loss = token_losses[span.start:span.end].mean()
        gradients = torch.autograd.grad(segment_loss, parameters, retain_graph=True, allow_unused=True)
        scores.append(gradient_dot(gradients, task_gradients).clamp_min(0))
        del gradients
    return torch.stack(scores) if scores else token_losses.new_empty((0,), dtype=torch.float64)


def validate_spans(spans: Sequence[SummarySpan], response_length: int):
    end = 0
    for span in spans:
        if not 0 <= span.start < span.end <= response_length or span.start < end:
            raise ValueError("Summary spans must be nonempty, disjoint and within the actual response")
        end = span.end


@dataclass(frozen=True)
class WeightResult:
    weights: torch.Tensor
    summary_tokens: float
    weighted_alignment: float
    fallback: bool
    raw_weights: torch.Tensor


def normalize_weights(alignments: torch.Tensor, lengths: torch.Tensor, distributed: bool = False,
                      *, strength: float = 1.0) -> WeightResult:


    if (isinstance(strength, bool) or not isinstance(strength, (int, float))
            or not math.isfinite(strength) or not 0 <= strength <= 1):
        raise ValueError('Meta weight strength must be finite and between 0 and 1')
    if alignments.ndim != 1 or alignments.shape != lengths.shape:
        raise ValueError("Alignment and length vectors must have matching one-dimensional shapes")
    a = alignments.detach().double()
    n = lengths.detach().to(device=a.device, dtype=torch.float64)
    if not torch.isfinite(a).all() or not torch.isfinite(n).all() or (a < 0).any() or (n <= 0).any():
        raise ValueError("Expected finite nonnegative alignments and positive lengths")
    totals = torch.stack((n.sum(), (n * a).sum()))
    if distributed:
        if not dist.is_initialized():
            raise RuntimeError("Distributed weight normalization requires an initialized process group")
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    count, denominator = totals.unbind()
    fallback = bool(denominator == 0)
    raw_weights = torch.ones_like(a) if fallback else count * a / denominator
    if not torch.isfinite(raw_weights).all():
        raise FloatingPointError("Nonfinite meta weights")


    weights = raw_weights if strength == 1 else 1 + strength * (raw_weights - 1)
    return WeightResult(weights.detach(), float(count), float(denominator), fallback, raw_weights.detach())


def token_weights(response_length: int, spans: Sequence[SummarySpan], weights: torch.Tensor, *, device=None, dtype=torch.float32):
    validate_spans(spans, response_length)
    if len(spans) != weights.numel():
        raise ValueError("Each summary segment needs exactly one meta weight")
    if not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Meta weights must be finite and nonnegative")
    result = torch.ones(response_length, device=device or weights.device, dtype=dtype)
    for span, weight in zip(spans, weights.detach(), strict=True):
        result[span.start:span.end] = weight
    return result.detach()


def weighted_token_sum(losses: torch.Tensor, spans: Sequence[SummarySpan], weights: torch.Tensor) -> torch.Tensor:

    return (losses * token_weights(len(losses), spans, weights, device=losses.device, dtype=losses.dtype)).sum()

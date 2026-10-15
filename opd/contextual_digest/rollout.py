from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable

import torch

from .config import RelayConfig
from .graph import CompiledSummary, SummaryBudgetExceeded, SummarySpan


@dataclass
class Trajectory:
    question_id: str
    prompt_ids: list[int]
    response_ids: list[int] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    summary_spans: list[SummarySpan] = field(default_factory=list)
    handoffs: list[dict] = field(default_factory=list)
    stop_reason: str = ""

    def validate(self):
        from .meta import validate_spans
        if not self.prompt_ids or len(self.response_ids) != len(self.sources):
            raise ValueError("Invalid trajectory token/source alignment")
        validate_spans(self.summary_spans, len(self.response_ids))
        covered = set()
        for span in self.summary_spans:
            covered.update(range(span.start, span.end))
        if covered != {i for i, source in enumerate(self.sources) if source == "summary"}:
            raise ValueError("Every actual summary token must belong to exactly one segment")


def relay_trigger(student_logits, teacher_logits, sampled_student_token, reflection_ids, topk):
    teacher_token = int(teacher_logits.argmax())
    student_topk = student_logits.topk(min(topk, student_logits.numel())).indices
    triggered = (
        teacher_token in reflection_ids
        and not any(int(token) in reflection_ids for token in student_topk)
    )
    return triggered, teacher_token


def sample_token(logits, generator, temperature=1.0):
    return int(torch.multinomial(torch.softmax(logits.float() / temperature, dim=-1), 1, generator=generator))


class RelayRollout:
    def __init__(self, student, teacher, tokenizer, config: RelayConfig,
                 reflection_ids: set[int], paragraph_ids: set[int], summary_builder: Callable | None = None):
        self.student, self.teacher, self.tokenizer = student, teacher, tokenizer
        self.config = config
        self.reflection_ids, self.paragraph_ids = reflection_ids, paragraph_ids
        self.summary_builder = summary_builder
        self.paper_digest = bool(getattr(summary_builder, 'paper_digest', False))
        self.summary_reserve = summary_builder.config.max_summary_tokens if self.paper_digest else 0
        if self.paper_digest:
            summary_builder.validate_relay(config)
        eos = tokenizer.eos_token_id
        self.eos = set(eos if isinstance(eos, (list, tuple)) else [eos])

    def generate(self, question_id, question_text, prompt_ids, student_generator, teacher_generator):
        cfg = self.config
        if len(prompt_ids) > cfg.max_prompt_tokens:
            raise ValueError("Prompt exceeds the configured Relay limit")
        trace = Trajectory(question_id, list(prompt_ids))
        self.student.clear_cache()
        self.teacher.clear_cache()
        in_teacher, completed, remaining_tokens, remaining_paragraphs = False, 0, 0, 0
        guidance_start = None

        def append(token, source):
            trace.response_ids.append(token)
            trace.sources.append(source)

        def hand_back():
            nonlocal completed, in_teacher
            completed += 1
            in_teacher = False
            event = {"takeover": completed, "guidance_start": guidance_start,
                     "guidance_end": len(trace.response_ids), "summary_status": "not_requested"}
            trace.handoffs.append(event)
            remaining = cfg.max_response_tokens - len(trace.response_ids)
            if remaining <= 1:
                event["summary_status"] = "insufficient_response_budget"
                return
            if self.summary_builder is None:
                return
            prefix = self.tokenizer.decode(trace.response_ids[:guidance_start], skip_special_tokens=False)
            guidance = self.tokenizer.decode(trace.response_ids[guidance_start:], skip_special_tokens=False)
            try:
                extra = {'guidance_token_ids': list(trace.response_ids[guidance_start:])} if self.paper_digest else {}
                summary, audit = self.summary_builder(
                    question_text, prefix, guidance, len(trace.response_ids), remaining - 1,
                    **extra,
                )
            except SummaryBudgetExceeded as exc:


                event.update(summary_status="budget_error", reason=str(exc))
                raise
            event.update(audit)
            event["summary_status"] = "inserted" if summary.token_ids else "no_grounded_frontier"
            for token in summary.token_ids:
                append(token, "summary")
            trace.summary_spans.extend(
                replace(s, node_id=f"h{completed}:{s.node_id}") for s in summary.spans
            )

        while len(trace.response_ids) < cfg.max_response_tokens and not trace.stop_reason:
            prefix = trace.prompt_ids + trace.response_ids
            if in_teacher:
                token = sample_token(self.teacher.next_logits(prefix), teacher_generator, cfg.temperature)
                append(token, "teacher")
                remaining_tokens -= 1
                remaining_paragraphs -= int(token in self.paragraph_ids)
                if token in self.eos:
                    trace.stop_reason = "eos"
                elif remaining_tokens <= 0 or remaining_paragraphs <= 0:
                    hand_back()
            else:
                student_logits = self.student.next_logits(prefix)
                draft = sample_token(student_logits, student_generator, cfg.temperature)
                can_take_over = completed < cfg.max_takeovers and (
                    cfg.max_response_tokens - len(trace.response_ids) > self.summary_reserve + 1)
                trigger, token = False, None
                if can_take_over:
                    teacher_logits = self.teacher.next_logits(prefix).to(student_logits.device)
                    trigger, token = relay_trigger(student_logits, teacher_logits, draft, self.reflection_ids, cfg.trigger_topk)
                if trigger and can_take_over:
                    guidance_start = len(trace.response_ids)
                    append(token, "teacher")
                    remaining_tokens = min(cfg.max_takeover_tokens - 1,
                        cfg.max_response_tokens - len(trace.response_ids) - self.summary_reserve - 1)
                    remaining_paragraphs = cfg.paragraphs_per_takeover - int(token in self.paragraph_ids)
                    in_teacher = remaining_paragraphs > 0 and remaining_tokens > 0
                    if token in self.eos:
                        trace.stop_reason = "eos"
                    elif not in_teacher:
                        hand_back()
                else:
                    append(draft, "student")
                    if draft in self.eos:
                        trace.stop_reason = "eos"
        if not trace.stop_reason:
            trace.stop_reason = "response_limit"
        self.student.clear_cache()
        self.teacher.clear_cache()
        trace.validate()
        if self.paper_digest:
            from .digest import validate_digest_trajectory
            validate_digest_trajectory(trace, cfg.max_response_tokens, self.summary_reserve)
        return trace


def independent_rollout(student, tokenizer, question_id, prompt_ids, max_tokens, generator):

    trace = Trajectory(question_id, list(prompt_ids))
    eos = tokenizer.eos_token_id
    eos = set(eos if isinstance(eos, (list, tuple)) else [eos])
    student.clear_cache()
    for _ in range(max_tokens):
        token = sample_token(student.next_logits(trace.prompt_ids + trace.response_ids), generator)
        trace.response_ids.append(token)
        trace.sources.append("student")
        if token in eos:
            trace.stop_reason = "eos"
            break
    trace.stop_reason = trace.stop_reason or "response_limit"
    student.clear_cache()
    trace.validate()
    return trace

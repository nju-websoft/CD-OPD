from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import time

from .batched_rollout import random_seed
from .digest import validate_digest_trajectory
from .rollout import Trajectory


class NativeRelayRollout:
    def __init__(self, native_client, tokenizer, config, summary_builder, *, graph_batch_size=16,
                 student_client=None):
        self.native, self.tokenizer, self.config = native_client, tokenizer, config
        self.summary_builder, self.student_client = summary_builder, student_client
        if student_client is None or summary_builder is None:
            raise ValueError('CD-OPD requires Student inference and Contextual Digest clients')
        if graph_batch_size < 1:
            raise ValueError('Graph batch size must be positive')
        eos = tokenizer.eos_token_id
        self.eos = set(eos if isinstance(eos, (list, tuple)) else [eos])
        self.graph_batch_size = graph_batch_size
        self.summary_reserve = summary_builder.config.max_summary_tokens
        summary_builder.validate_relay(config)
        self.timings = {}

    def _add_time(self, key, seconds):
        self.timings[key] = self.timings.get(key, 0.0) + seconds

    def generate_many(self, questions, student_generator, teacher_generator):
        cfg = self.config
        if any(not q.prompt_ids or len(q.prompt_ids) > cfg.max_prompt_tokens for q in questions):
            raise ValueError('Invalid native Relay question prefix')
        traces = [Trajectory(q.id, list(q.prompt_ids)) for q in questions]
        counts = [0] * len(traces)
        waiting, targets = [], []
        future, pending = None, 0
        student_tail = set()

        def submit(indices, resume):
            nonlocal pending
            requests = []
            for index in indices:
                trace = traces[index]
                remaining = cfg.max_response_tokens - len(trace.response_ids) - self.summary_reserve - 1
                if counts[index] >= cfg.max_takeovers or remaining <= 0:
                    student_tail.add(index)
                    continue
                requests.append(dict(key=index, prompt_ids=trace.prompt_ids + trace.response_ids,
                    max_tokens=remaining, temperature=cfg.temperature, resume_handoff=resume))
            if requests:
                started = time.perf_counter()
                result = self.native.call('relay_add', requests=requests)
                self._add_time('native_dispatch_seconds', time.perf_counter() - started)
                if result['accepted'] != [r['key'] for r in requests]:
                    raise RuntimeError('Native request dispatch lost or reordered a request')
                pending = result['pending']

        def build(items):
            requests = []
            for index, event, _ in items:
                trace = traces[index]
                start, end = event['guidance_start'], event['guidance_end']
                requests.append(dict(question=questions[index].text,
                    prefix=self.tokenizer.decode(trace.response_ids[:start], skip_special_tokens=False),
                    guidance=self.tokenizer.decode(trace.response_ids[start:end], skip_special_tokens=False),
                    guidance_token_ids=trace.response_ids[start:end], response_offset=end,
                    remaining_tokens=self.summary_reserve))
            started = time.perf_counter()
            result = self.summary_builder.build_many(requests)
            return result, time.perf_counter() - started

        submit(list(range(len(traces))), False)
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix='digest_graph') as executor:
            while pending or waiting or future:
                if future and (future.done() or not pending):
                    summaries, seconds = future.result()
                    self._add_time('graph_handoff_seconds', seconds)
                    resumed = []
                    for (index, event, _), (digest, audit) in zip(targets, summaries, strict=True):
                        trace = traces[index]
                        if len(trace.response_ids) != event['guidance_end']:
                            raise RuntimeError('Student advanced before its Digest was constructed')
                        if not digest.token_ids or len(digest.token_ids) > self.summary_reserve:
                            raise RuntimeError('Digest is empty or exceeds its reserved budget')
                        event.update(audit, summary_status='inserted')
                        trace.response_ids.extend(digest.token_ids)
                        trace.sources.extend(['summary'] * len(digest.token_ids))
                        trace.summary_spans.extend(replace(span, node_id=f"h{event['takeover']}:{span.node_id}")
                                                   for span in digest.spans)
                        event['summary_end'] = len(trace.response_ids)
                        if len(trace.response_ids) > cfg.max_response_tokens:
                            raise RuntimeError('Digest exceeded the response budget')
                        resumed.append(index)
                    future, targets = None, []
                    submit(resumed, True)
                if waiting and future is None and (len(waiting) >= self.graph_batch_size or not pending
                        or time.perf_counter() - waiting[0][2] >= 2.0):
                    targets, waiting = waiting[:self.graph_batch_size], waiting[self.graph_batch_size:]
                    future = executor.submit(build, targets)
                if not pending:
                    continue
                started = time.perf_counter()
                result = self.native.call('relay_advance', max_steps=8)
                self._add_time('native_relay_seconds', time.perf_counter() - started)
                pending = result['pending']
                for output in result['finished']:
                    index = output['key']
                    trace = traces[index]
                    before = len(trace.response_ids)
                    ids, sources = output['token_ids'], output['sources']
                    if trace.stop_reason or not ids or len(ids) != len(sources):
                        raise RuntimeError('Invalid native completion or token provenance')
                    trace.response_ids.extend(ids)
                    trace.sources.extend(sources)
                    if len(trace.response_ids) > cfg.max_response_tokens:
                        raise RuntimeError('Native response exceeded its token budget')
                    teachers = [j for j, source in enumerate(sources) if source == 'teacher']
                    ended = ids[-1] in self.eos
                    if not teachers:
                        if output['returned']:
                            raise RuntimeError('Native handback has no Teacher provenance')
                        if ended:
                            trace.stop_reason = 'eos'
                        else:
                            student_tail.add(index)
                        continue
                    beginning = teachers[0]
                    if teachers != list(range(beginning, len(ids))) or len(teachers) > cfg.max_takeover_tokens:
                        raise RuntimeError('Teacher guidance is not a bounded contiguous suffix')
                    counts[index] += 1
                    event = dict(takeover=counts[index], guidance_start=before + beginning,
                        guidance_end=len(trace.response_ids), summary_status='teacher_eos' if ended else 'not_requested',
                        generation_backend='native_vllm_relay', summary_reserved_tokens=self.summary_reserve)
                    trace.handoffs.append(event)
                    if ended:
                        trace.stop_reason = 'eos'
                    else:
                        waiting.append((index, event, time.perf_counter()))
        self.native.call('relay_finish')
        active = [i for i in sorted(student_tail) if len(traces[i].response_ids) < cfg.max_response_tokens]
        outputs = self.student_client.generate([dict(
            prompt_ids=traces[i].prompt_ids + traces[i].response_ids,
            max_tokens=cfg.max_response_tokens - len(traces[i].response_ids),
            temperature=cfg.temperature, seed=random_seed(student_generator)) for i in active]) if active else []
        for index, output in zip(active, outputs, strict=True):
            ids = output['token_ids']
            if not ids or len(ids) > cfg.max_response_tokens - len(traces[index].response_ids):
                raise RuntimeError('Invalid Student continuation after teacher takeover')
            traces[index].response_ids.extend(ids)
            traces[index].sources.extend(['student'] * len(ids))
            traces[index].stop_reason = 'eos' if ids[-1] in self.eos else 'response_limit'
        for index in student_tail:
            traces[index].stop_reason = traces[index].stop_reason or 'response_limit'
        for trace in traces:
            if not trace.stop_reason:
                raise RuntimeError('Native trajectory was not completed')
            validate_digest_trajectory(trace, cfg.max_response_tokens, self.summary_reserve)
        return traces

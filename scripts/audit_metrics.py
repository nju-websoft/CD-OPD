import argparse
from collections import Counter
from dataclasses import replace
import json
import math
from pathlib import Path

from transformers import AutoTokenizer
from opd.contextual_digest.config import TrainConfig
from opd.contextual_digest.data import load_questions
from opd.contextual_digest.digest import parse_digest_graph, compile_digest, validate_digest_trajectory
from opd.contextual_digest.graph import SummarySpan
from opd.contextual_digest.rollout import Trajectory


def audit(run, config):
    tokenizer = AutoTokenizer.from_pretrained(config.student_model, local_files_only=True)
    questions, _ = load_questions(config.train_data, tokenizer, config.relay.max_prompt_tokens)
    questions = {q.id: q for q in questions}
    paths = sorted(run.glob('metrics_rank_*.jsonl'))
    if len(paths) != len(config.student_devices):
        raise ValueError('Missing rank metrics')
    metrics = [{row['step']: row for row in map(json.loads, p.read_text().splitlines())} for p in paths]
    steps = set(metrics[0])
    if any(set(m) != steps for m in metrics) or not steps:
        raise ValueError('Missing or inconsistent completed updates')
    reports = []
    for step in sorted(steps):
        traces, tasks, segments = [], [], []
        for rank in range(len(paths)):
            metric = metrics[rank][step]
            assert metric['student_optimizer_steps'] == 1 and metric['teacher_optimizer_steps'] == 0
            assert metric['teacher_parameters_unchanged']
            row = json.loads((run / f'audit/step_{step}/rank_{rank}.json').read_text())
            traces.extend(row['handoff_trajectories'])
            tasks.extend(row['task_trajectories'])
            segments.extend(s for item in row['meta_segments'] for s in item['segments'])
        assert len(traces) == config.handoff_batch_size
        assert len(tasks) == config.meta.task_batch_size * config.meta.task_samples
        task_counts = Counter(t['trajectory']['question_id'] for t in tasks)
        assert len(task_counts) == config.meta.task_batch_size
        assert set(task_counts.values()) == {config.meta.task_samples}
        assert not {t['question_id'] for t in traces} & set(task_counts)
        for task in tasks:
            assert task['reward'] in (0, 1)
            assert all(s == 'student' for s in task['trajectory']['sources'])
            assert not task['trajectory']['summary_spans'] and not task['trajectory']['handoffs']
        total_tokens, digest_tokens = 0, 0
        for raw in traces:
            trace = Trajectory(**{**raw, 'summary_spans': [SummarySpan(**s) for s in raw['summary_spans']]})
            validate_digest_trajectory(trace, config.relay.max_response_tokens, config.handoff.max_summary_tokens)
            total_tokens += len(trace.response_ids)
            digest_tokens += sum(s.end-s.start for s in trace.summary_spans)
            for event in trace.handoffs:
                if event['summary_status'] == 'teacher_eos':
                    continue
                start, end = event['guidance_start'], event['guidance_end']
                request = dict(question=questions[trace.question_id].text,
                    prefix=tokenizer.decode(trace.response_ids[:start], skip_special_tokens=False),
                    guidance=tokenizer.decode(trace.response_ids[start:end], skip_special_tokens=False),
                    guidance_token_ids=trace.response_ids[start:end], response_offset=end)
                nodes, order = parse_digest_graph(event['raw_graph'], request, config.handoff.max_nodes)
                digest = compile_digest(nodes, order, tokenizer, request, event['summary_budget'])
                assert list(digest.token_ids) == trace.response_ids[end:end+len(digest.token_ids)]
                actual = [s for s in trace.summary_spans if s.node_id.startswith(f"h{event['takeover']}:")]
                assert actual == [replace(s, node_id=f"h{event['takeover']}:{s.node_id}") for s in digest.spans]
        lengths = [s['end']-s['start'] for s in segments]
        scores = [s['alignment'] for s in segments]
        assert all(n > 0 for n in lengths)
        assert all(math.isfinite(a) and a >= 0 for a in scores)
        denominator = sum(n*a for n, a in zip(lengths, scores, strict=True))
        count = sum(lengths)
        expected = [count*a/denominator if denominator else 1.0 for a in scores]
        assert all(math.isclose(s['weight'], w, rel_tol=1e-10, abs_tol=1e-10)
                   for s, w in zip(segments, expected, strict=True))
        assert count == digest_tokens
        assert math.isclose(sum(n*s['weight'] for n,s in zip(lengths,segments,strict=True)), count, rel_tol=1e-10, abs_tol=1e-10)
        assert total_tokens == metrics[0][step]['response_tokens']
        reports.append(dict(step=step, response_tokens=total_tokens, digest_tokens=digest_tokens,
                            digest_segments=len(segments), zero_alignment_fallback=denominator == 0))
    return dict(verified_steps=len(reports), updates=reports)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = audit(args.run, TrainConfig.load(args.config))
    content = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(content + '\n')
    print(content)


if __name__ == '__main__':
    main()

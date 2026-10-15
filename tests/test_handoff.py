from copy import deepcopy
import json
import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from opd.contextual_digest.backend import HFBackend
from opd.contextual_digest.config import TrainConfig, RelayConfig
from opd.contextual_digest.data import OnlineTrainingStream, TrainingQuestion
from opd.contextual_digest.graph import SummarySpan, CompiledSummary
from opd.contextual_digest.losses import relay_token_losses
from opd.contextual_digest.meta import relative_advantages, gradient_dot, segment_alignments, normalize_weights, token_weights, weighted_token_sum
from opd.contextual_digest.rollout import Trajectory, RelayRollout, relay_trigger, independent_rollout
from opd.contextual_digest.update import MetaHandoffUpdater


class CharacterTokenizer:
    eos_token_id = 0

    def encode(self, text, **kwargs):
        return [ord(c) for c in text]

    def decode(self, tokens, **kwargs):
        return "".join(chr(t) for t in tokens)


def test_relative_advantages_ties_and_minimum_samples():
    values = relative_advantages(torch.tensor([[1., 0., 0.], [0.3, 0.3, 0.3]]))
    torch.testing.assert_close(values, torch.tensor([[1., -.5, -.5], [0., 0., 0.]], dtype=torch.float64))
    with pytest.raises(ValueError):
        relative_advantages(torch.ones(2, 1))


def test_full_parameter_alignment_length_normalization_and_detachment():
    p = torch.nn.Parameter(torch.tensor([1., 2.], dtype=torch.float64))
    q = torch.nn.Parameter(torch.tensor([3.], dtype=torch.float64))

    losses = torch.stack((2*p[0]+q[0], 2*p[0]+q[0], -p[0]-q[0], 3*p[1]+2*q[0]))
    spans = [SummarySpan("a", 0, 2, ""), SummarySpan("b", 2, 3, ""), SummarySpan("c", 3, 4, "")]
    task = (torch.tensor([1., 2.], dtype=torch.float64), torch.tensor([4.], dtype=torch.float64))
    a = segment_alignments(losses, spans, (p, q), task)
    torch.testing.assert_close(a, torch.tensor([6., 0., 14.], dtype=torch.float64))
    result = normalize_weights(a, torch.tensor([2., 1., 1.]))
    torch.testing.assert_close(result.weights, 4*a/26)
    assert not result.weights.requires_grad
    assert float((result.weights * torch.tensor([2., 1., 1.])).sum()) == pytest.approx(4)
    assert p.grad is None and q.grad is None
    assert float(gradient_dot(task, task)) == 21
    assert normalize_weights(torch.zeros(2), torch.tensor([2., 7.])).weights.tolist() == [1., 1.]


def test_non_summary_tokens_and_original_denominator_are_unchanged():
    losses = torch.arange(1., 7., requires_grad=True)
    spans = [SummarySpan("a", 1, 3, ""), SummarySpan("b", 4, 5, "")]
    weights = torch.tensor([0., 3.], requires_grad=True)
    result = weighted_token_sum(losses, spans, weights) / 6
    result.backward()
    torch.testing.assert_close(losses.grad, torch.tensor([1., 0., 0., 1., 3., 1.])/6)
    assert weights.grad is None
    with pytest.raises(ValueError):
        token_weights(6, [SummarySpan("a", 1, 4, ""), SummarySpan("b", 3, 5, "")], weights)


def tiny_qwen():
    return Qwen3ForCausalLM(Qwen3Config(
        vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        max_position_embeddings=128, attention_dropout=0., tie_word_embeddings=True,
    )).eval()


def test_causal_scorer_and_gradients_match_public_hf_forward():
    torch.manual_seed(4)
    model = tiny_qwen()
    backend = HFBackend(model, sample_bfloat16=False, score_chunk_tokens=2)
    prompt, response = [1, 2, 3], [4, 5, 6, 7]
    ids = torch.tensor([prompt + response])
    expected = model(ids, use_cache=False).logits[0, len(prompt)-1:-1].float().log_softmax(-1)
    expected = expected.gather(-1, torch.tensor(response)[:, None]).squeeze(-1)
    actual = backend.log_probs(prompt, response, grad=True)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    params = tuple(model.parameters())
    grad_actual = torch.autograd.grad(actual.sum(), params)
    grad_expected = torch.autograd.grad(expected.sum(), params)
    for a, b in zip(grad_actual, grad_expected):
        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)

    changed = backend.log_probs(prompt, [4, 5, 8, 9])
    torch.testing.assert_close(changed[:2], actual.detach()[:2])

    private = backend.log_probs([10, 11, 12], response)
    assert not torch.allclose(actual.detach(), private)
    backend.clear_cache()
    a = backend.next_logits(prompt)
    b = backend.next_logits(prompt + response[:1])
    with torch.no_grad():
        torch.testing.assert_close(a, model(torch.tensor([prompt])).logits[0, -1])
        torch.testing.assert_close(b, model(torch.tensor([prompt + response[:1]])).logits[0, -1])


def test_relay_clipping_and_advantage_stop_gradient():
    new = torch.tensor([-1., -1., -1., -1.], requires_grad=True)
    old = torch.tensor([-2., -2., 0., 0.])
    teacher = torch.tensor([0., -2., 0., -2.], requires_grad=True)
    result = relay_token_losses(new, old, teacher)

    expected = torch.tensor([-2.4, 0., 0., 1.6])
    torch.testing.assert_close(result, expected)
    result.sum().backward()
    assert teacher.grad is None
    torch.testing.assert_close(new.grad, torch.zeros(4))


class FixedBackend:
    def __init__(self, mapping, default):
        self.mapping, self.default, self.seen = mapping, default, []
        self.device = torch.device("cpu")

    def clear_cache(self):
        pass

    def next_logits(self, prefix):
        self.seen.append(list(prefix))
        result = torch.full((8,), -1000.)
        result[self.mapping.get(tuple(prefix), self.default)] = 0.
        return result


def test_trigger_uses_teacher_argmax_outside_student_topk():
    s, t = torch.tensor([10., 8., 1.]), torch.tensor([0., 0., 10.])
    assert relay_trigger(s, t, 0, {2}, 2) == (True, 2)
    assert relay_trigger(s, t, 0, {2}, 3) == (False, 2)
    assert relay_trigger(s, t, 2, {2}, 2) == (True, 2)
    assert relay_trigger(s, t, 0, {1}, 2) == (False, 2)


def test_online_batches_disjoint_resume_exact_and_no_fixed_holdout():
    questions = [TrainingQuestion(str(i), str(i), [i+1], str(i), "math") for i in range(20)]
    stream = OnlineTrainingStream(questions, 4, 3, epochs=2)
    first_h, first_t = next(stream)
    state = stream.state_dict()
    clone = OnlineTrainingStream(questions, 4, 3, epochs=2)
    clone.load_state_dict(state)
    assert next(stream) == next(clone)
    all_h, all_t = {q.id for q in first_h}, {q.id for q in first_t}
    for h, t in clone:
        assert not {q.id for q in h} & {q.id for q in t}
        all_h.update(q.id for q in h)
        all_t.update(q.id for q in t)
    assert all_h & all_t
    altered = deepcopy(questions)
    altered[0].ground_truth = "different"
    with pytest.raises(ValueError, match="changed"):
        OnlineTrainingStream(altered, 4, 3, epochs=2).load_state_dict(state)


class TableBackend:

    def __init__(self, values, frozen=False):
        self.model = torch.nn.Linear(1, 3, bias=False, dtype=torch.float64)
        with torch.no_grad():
            self.model.weight[:, 0].copy_(torch.tensor(values))
        self.model.requires_grad_(not frozen)
        self.device = torch.device("cpu")

    def log_probs(self, prompt, response, grad=False):
        with torch.enable_grad() if grad else torch.no_grad():
            return self.model.weight[:, 0].log_softmax(0)[response]


def trace(q, tokens, spans=()):
    sources = ["student"] * len(tokens)
    for span in spans:
        sources[span.start:span.end] = ["summary"] * (span.end-span.start)
    return Trajectory(q, [1], tokens, sources, list(spans))


@pytest.mark.parametrize('strength', [1.0])
def test_online_meta_changes_summary_learning_with_one_update_and_frozen_teacher(strength):
    student, teacher = TableBackend([0., 0., 0.]), TableBackend([1., 1., -2.], frozen=True)
    cfg = TrainConfig()
    cfg.meta.weight_strength = strength
    cfg.max_grad_norm = 100.
    optimizer = torch.optim.SGD(student.model.parameters(), lr=.01)
    update = MetaHandoffUpdater(student, teacher, optimizer, cfg)
    handoff = [trace("handoff", [0, 1, 2], [SummarySpan("a", 0, 1, "a"), SummarySpan("b", 1, 2, "b")])]
    tasks = [(trace("task", [0]), 1.), (trace("task", [1]), 0.)]
    before_student = student.model.weight.detach().clone()
    before_teacher = teacher.model.weight.detach().clone()


    lp = student.log_probs([1], [0, 1, 2], grad=True)
    teacher_lp = teacher.log_probs([1], [0, 1, 2])
    oracle_loss = (relay_token_losses(lp, lp.detach(), teacher_lp)
                   * torch.tensor([1 + strength, 1 - strength, 1.], dtype=torch.float64)).mean()
    oracle_gradient = torch.autograd.grad(oracle_loss, (student.model.weight,))[0]
    calls = []
    original = optimizer.step
    def step(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    optimizer.step = step
    metrics, audits = update.update(handoff, tasks)
    assert calls == [1]
    assert [s["weight"] for s in audits[0]["segments"]] == pytest.approx([1 + strength, 1 - strength])
    assert [s['raw_weight'] for s in audits[0]['segments']] == pytest.approx([2., 0.])
    assert metrics['meta_weight_strength'] == strength
    assert metrics['summary_weight_rms_from_one'] == pytest.approx(strength)
    assert metrics['summary_raw_weight_rms_from_one'] == pytest.approx(1.)
    assert metrics["task_informative_questions"] == 1
    assert metrics["summary_weighted_token_sum"] == pytest.approx(2)
    torch.testing.assert_close(student.model.weight, before_student - .01*oracle_gradient)
    torch.testing.assert_close(teacher.model.weight, before_teacher)
    assert teacher.model.weight.grad is None and not teacher.model.weight.requires_grad
    assert student.model.weight.grad is None


@pytest.mark.parametrize('strength', [1.0])
def test_tied_online_rewards_fall_back_to_equal_summary_weights(strength):
    student, teacher = TableBackend([0., 0., 0.]), TableBackend([1., 1., -2.], frozen=True)
    cfg = TrainConfig()
    cfg.meta.weight_strength = strength
    updater = MetaHandoffUpdater(student, teacher, torch.optim.SGD(student.model.parameters(), lr=.01), cfg)
    h = [trace("h", [0, 1], [SummarySpan("a", 0, 1, ""), SummarySpan("b", 1, 2, "")])]
    tasks = [(trace("t", [0]), 0.), (trace("t", [1]), 0.)]
    metrics, audits = updater.update(h, tasks)
    assert metrics["meta_equal_weight_fallback"] and metrics["task_informative_questions"] == 0
    assert [s["weight"] for s in audits[0]["segments"]] == [1., 1.]
    with pytest.raises(ValueError, match="overlap"):
        updater.update(h, [(trace("h", [0]), 1.), (trace("h", [1]), 0.)])


def distributed_examples(rank):
    if rank == 0:
        h = trace("h0", [0, 1, 2], [SummarySpan("a", 0, 1, ""), SummarySpan("b", 1, 2, "")])
        tasks = [(trace("t0", [0]), 1.), (trace("t0", [1]), 0.)]
    else:
        h = trace("h1", [0, 1, 1, 2], [SummarySpan("a", 0, 2, ""), SummarySpan("b", 2, 3, "")])
        tasks = [(trace("t1", [0]), 1.), (trace("t1", [2]), 0.)]
    return h, tasks


def distributed_worker(rank, init_file, output, strength):
    from pathlib import Path
    import torch.distributed as dist
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method="file://" + init_file, rank=rank, world_size=2)
    try:
        student = TableBackend([0., 0., 0.])
        teacher = TableBackend([1., 1., -2.], frozen=True)
        cfg = TrainConfig()
        cfg.meta.weight_strength = strength
        cfg.max_grad_norm = 100.
        optimizer = torch.optim.SGD(student.model.parameters(), lr=.01)
        h, tasks = distributed_examples(rank)
        metrics, audits = MetaHandoffUpdater(student, teacher, optimizer, cfg, distributed=True).update([h], tasks)
        (Path(output) / f"rank{rank}.json").write_text(json.dumps(dict(
            weight=student.model.weight.detach().tolist(), metrics=metrics, audits=audits,
        )))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize('strength', [1.0])
def test_distributed_update_matches_global_batch_including_meta_weights(tmp_path, strength):
    import torch.multiprocessing as mp
    mp.spawn(distributed_worker, args=(str(tmp_path / "rendezvous"), str(tmp_path), strength), nprocs=2, join=True)
    student, teacher = TableBackend([0., 0., 0.]), TableBackend([1., 1., -2.], frozen=True)
    cfg = TrainConfig()
    cfg.meta.weight_strength = strength
    cfg.max_grad_norm = 100.
    optimizer = torch.optim.SGD(student.model.parameters(), lr=.01)
    h0, t0 = distributed_examples(0)
    h1, t1 = distributed_examples(1)
    metrics, audits = MetaHandoffUpdater(student, teacher, optimizer, cfg).update([h0, h1], t0+t1)
    for rank in range(2):
        result = json.loads((tmp_path / f"rank{rank}.json").read_text())
        torch.testing.assert_close(torch.tensor(result["weight"], dtype=torch.float64), student.model.weight)
        assert result["metrics"]["summary_weighted_token_sum"] == pytest.approx(metrics["summary_weighted_token_sum"])
        assert [s["weight"] for s in result["audits"][0]["segments"]] == pytest.approx(
            [s["weight"] for s in audits[rank]["segments"]])


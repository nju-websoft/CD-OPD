import torch
import pytest

from opd.contextual_digest.graph import SummarySpan
from opd.contextual_digest.journal import RolloutJournal, model_fingerprint
from opd.contextual_digest.rollout import Trajectory


def example():
    return Trajectory("q1", [1, 2], [3, 4, 5], ["student", "summary", "summary"],
                      [SummarySpan("h1:I1", 1, 3, "A condition remains unverified.")])


def test_recovery_restores_trace_reward_and_all_sampling_states(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    generators = [torch.Generator().manual_seed(42), torch.Generator().manual_seed(10042)]
    identity = {"student_sha256": "snapshot-one", "betas": (0.9, 0.999)}
    journal = RolloutJournal(tmp_path, 1, 0, identity, generators)
    trace = example()
    for generator in generators:
        torch.rand(7, generator=generator)
    journal.save("independent", 0, trace, 0.75)
    expected = [torch.rand(6, generator=g) for g in generators] + [torch.rand(6)]
    for generator in generators:
        torch.rand(31, generator=generator)
    torch.rand(31)

    recovered = RolloutJournal(tmp_path, 1, 0, identity, generators)
    restored, reward = recovered.load("independent", 0, "q1")
    assert restored == trace and reward == 0.75
    actual = [torch.rand(6, generator=g) for g in generators] + [torch.rand(6)]
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    with pytest.raises(FileExistsError):
        recovered.save("independent", 0, trace, 0.75)
    with pytest.raises(ValueError, match="different snapshot or question"):
        recovered.load("independent", 0, "a different question")


def test_changed_student_snapshot_cannot_reuse_rollouts(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    model = torch.nn.Linear(2, 2)
    model.register_buffer("scalar", torch.tensor(1))
    before = model_fingerprint(model)
    first = RolloutJournal(tmp_path, 1, 0, {"student_sha256": before}, [])
    first.save("handoff", 0, example())
    with torch.no_grad():
        model.weight[0, 0].add_(0.1)
    after = model_fingerprint(model)
    assert before != after
    second = RolloutJournal(tmp_path, 1, 0, {"student_sha256": after}, [])
    assert second.load("handoff", 0, "q1") is None
    assert first.directory != second.directory


def test_journal_does_not_persist_invalid_token_alignment(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    journal = RolloutJournal(tmp_path, 1, 0, {"student_sha256": "snapshot"}, [])
    trace = example()
    trace.summary_spans.clear()
    with pytest.raises(ValueError, match="Every actual summary token"):
        journal.save("handoff", 0, trace)
    assert not list(journal.directory.glob("*.pt"))

import json
from types import SimpleNamespace

import pytest
import torch

from opd.contextual_digest.recovery import JournalReplay


def test_replay_refuses_stale_snapshots_changed_tasks_and_different_updates(tmp_path):
    config = SimpleNamespace(to_dict=lambda: {"seed": 42})
    first = {"student_sha256": "before", "config": {"seed": 42}, "handoff_ids": ["a"],
             "generation_source_sha256": {"decoder": "old"}}
    plan = {"config": config.to_dict(), "world_size": 1, "completed_steps": 1,
            "identities": {"1": {"0": first}, "2": {"0": {**first, "student_sha256": "after"}}},
            "metrics": {"0": {"1": {"opd_loss": 0.2, "wall_seconds": 30}}}}
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    replay = JournalReplay(path, config, 1)
    assert replay.identity(1, 0, {**first, "generation_source_sha256": {"decoder": "fixed"}}) == first
    for change in ({"student_sha256": "stale"}, {"handoff_ids": ["b"]}):
        with pytest.raises(ValueError, match="differs"):
            replay.identity(1, 0, {**first, **change})
    assert replay.verify_update(1, 0, {"opd_loss": 0.2, "wall_seconds": 1}, "after") == "after"
    with pytest.raises(ValueError, match="bitwise identical"):
        replay.verify_update(1, 0, {"opd_loss": 0.2}, "different")
    with pytest.raises(ValueError, match="metrics differ"):
        replay.verify_update(1, 0, {"opd_loss": 0.3}, "after")


def test_recovery_snapshot_survives_failed_save_and_regular_cadence_is_separate(tmp_path, monkeypatch):
    from opd.contextual_digest.train import save_checkpoint
    monkeypatch.delenv("META_HANDOFF_HF_ARCHIVE", raising=False)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: [])

    class Student:
        def save_pretrained(self, path, **kwargs):
            path.mkdir(parents=True)
            (path / "weights").write_text("complete")

    tokenizer = SimpleNamespace(save_pretrained=lambda path: (path / "tokenizer").write_text("complete"))
    state = SimpleNamespace(state_dict=lambda: {"step": 1})
    config = SimpleNamespace(to_dict=lambda: {"save_every": 5})
    generators = [torch.Generator(), torch.Generator()]

    def save(step, recovery=True, student=None):
        save_checkpoint(tmp_path, step, student or Student(), tokenizer, state, state, generators,
                        config, 0, 1, recovery=recovery)

    save(1)
    assert not (tmp_path / "latest_checkpoint.txt").exists()
    first = (tmp_path / "latest_recovery.txt").read_text()
    broken = SimpleNamespace(save_pretrained=lambda *a, **kw: (_ for _ in ()).throw(IOError("disk failure")))
    with pytest.raises(IOError, match="disk failure"):
        save(2, student=broken)
    assert (tmp_path / "latest_recovery.txt").read_text() == first
    assert (tmp_path / "recovery/step_1/optimizer.pt").is_file()
    save(3)
    assert not (tmp_path / "recovery/step_1").exists()
    save(5, recovery=False)
    assert not (tmp_path / "recovery/step_3").exists()
    assert (tmp_path / "checkpoints/step_5/rank_0.pt").is_file()
    assert (tmp_path / "latest_checkpoint.txt").read_text() == (tmp_path / "latest_recovery.txt").read_text()

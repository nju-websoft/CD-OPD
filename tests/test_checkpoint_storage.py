import json
from pathlib import Path

import pytest

from opd.contextual_digest.checkpoints import apply_storage_policy, archive_student


def checkpoint(output, step):
    path = output / "checkpoints" / f"step_{step}"
    (path / "student").mkdir(parents=True)
    (path / "student/model.safetensors").write_bytes(bytes([step]) * 100)
    (path / "student/config.json").write_text('{"model_type":"test"}')
    (path / "optimizer.pt").write_bytes(b"optimizer")
    (path / "rank_0.pt").write_bytes(b"rank state")
    return path


def test_retention_keeps_all_weights_and_two_complete_resume_states(tmp_path):
    output, archive = tmp_path / "run", tmp_path / "archive"
    saved = []
    for step in (5, 10, 15):
        saved.append(checkpoint(output, step))
        apply_storage_policy(output, saved[-1], archive, keep_resumable=2)
    assert (saved[0] / "student").is_symlink()
    assert not (saved[0] / "optimizer.pt").exists()
    assert not (saved[0] / "rank_0.pt").exists()
    for path, step in zip(saved, (5, 10, 15)):
        assert (path / "student/model.safetensors").read_bytes() == bytes([step]) * 100
        assert (archive / f"step_{step}/archive_manifest.json").exists()
    assert all((path / "optimizer.pt").exists() and (path / "rank_0.pt").exists() for path in saved[1:])
    record = json.loads((output / "checkpoint_storage.json").read_text())
    assert record["resumable_checkpoints"] == [str(p) for p in saved[1:]]


def test_corrupt_archive_cannot_trigger_deletion_of_a_durable_checkpoint(tmp_path):
    output, archive = tmp_path / "run", tmp_path / "archive"
    first = checkpoint(output, 5)
    archive_student(first, archive)
    (archive / "step_5/model.safetensors").write_bytes(b"corrupted")
    second = checkpoint(output, 10)
    with pytest.raises(ValueError, match="checksum"):
        apply_storage_policy(output, second, archive, keep_resumable=1)
    assert (first / "optimizer.pt").exists()
    assert (first / "student/model.safetensors").read_bytes() == bytes([5]) * 100


def test_archive_ownership_prevents_cross_run_reuse(tmp_path):
    archive = tmp_path / "archive"
    archive_student(checkpoint(tmp_path / "run1", 5), archive)
    with pytest.raises(ValueError, match="different run"):
        archive_student(checkpoint(tmp_path / "run2", 5), archive)


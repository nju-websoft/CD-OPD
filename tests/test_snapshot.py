import json

import pytest

from opd.contextual_digest.snapshot import seal_snapshot, verify_snapshot, publish_update


def test_snapshot_checks_all_resume_files_before_loading(tmp_path):
    (tmp_path / "student").mkdir()
    files = {"student/model.safetensors": b"student", "optimizer.pt": b"adam",
             "rank_0.pt": b"rng and stream", "config.json": b"{}"}
    for name, value in files.items():
        (tmp_path / name).write_bytes(value)
    seal_snapshot(tmp_path, {"step": 1, "world_size": 1, "student_sha256": "model hash"})
    assert verify_snapshot(tmp_path, 1)["step"] == 1
    with pytest.raises(ValueError, match="topology"):
        verify_snapshot(tmp_path, 2)
    (tmp_path / "optimizer.pt").write_bytes(b"corrupted adam")
    with pytest.raises(ValueError, match="checksum"):
        verify_snapshot(tmp_path, 1)


def test_resume_can_publish_a_committed_update_once_after_a_log_interruption(tmp_path):
    metrics = {"step": 1, "rank": 0, "opd_loss": 0.2}
    audit = {"step": 1, "source": "committed snapshot"}


    publish_update(tmp_path, 0, metrics, audit)
    publish_update(tmp_path, 0, metrics, audit)
    assert len((tmp_path / "metrics_rank_0.jsonl").read_text().splitlines()) == 1
    assert json.loads((tmp_path / "audit/step_1/rank_0.json").read_text()) == audit
    with pytest.raises(ValueError, match="disagree"):
        publish_update(tmp_path, 0, {**metrics, "opd_loss": 0.3}, audit)
    publish_update(tmp_path, 0, {**metrics, "step": 2}, {**audit, "step": 2})
    with pytest.raises(ValueError, match="newer updates"):
        publish_update(tmp_path, 0, metrics, audit)


@pytest.mark.parametrize("text", ["中文，😀", "private \\udcdd literal", "private \udcdd escape", "high \ud83d escape"])
def test_resume_publication_preserves_private_unicode_after_metrics_were_written(tmp_path, text):
    metrics = {"step": 26, "rank": 0, "opd_loss": 0.3}
    audit = {"step": 26, "graph": {"raw_state": text}, "public_tokens": [1, 2, 3]}


    (tmp_path / "metrics_rank_0.jsonl").write_text(json.dumps(metrics) + "\n")
    directory = tmp_path / "audit/step_26"
    directory.mkdir(parents=True)
    (directory / "rank_0.json.incomplete").write_text("")
    publish_update(tmp_path, 0, metrics, audit)
    publish_update(tmp_path, 0, metrics, audit)
    assert json.loads((directory / "rank_0.json").read_text(encoding="utf-8")) == audit
    assert not (directory / "rank_0.json.incomplete").exists()
    assert (tmp_path / "metrics_rank_0.jsonl").read_text().splitlines() == [json.dumps(metrics)]

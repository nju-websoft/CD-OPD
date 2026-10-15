from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def archive_student(checkpoint, archive_root):
    checkpoint, archive_root = Path(checkpoint), Path(archive_root)
    output = checkpoint.parent.parent.resolve()
    archive_root.mkdir(parents=True, exist_ok=True)
    owner = archive_root / "owner.json"
    identity = {"output": str(output), "method": "Meta-Handoff OPD"}
    if owner.exists():
        if json.loads(owner.read_text()) != identity:
            raise ValueError("Checkpoint archive belongs to a different run")
    elif any(archive_root.iterdir()):
        raise ValueError("Refusing to use a nonempty, unowned checkpoint archive")
    else:
        owner.write_text(json.dumps(identity) + "\n")
    target = archive_root / checkpoint.name
    source = checkpoint / "student"
    if target.exists():
        record = json.loads((target / "archive_manifest.json").read_text())
        if record["checkpoint"] != str(checkpoint.resolve()):
            raise ValueError("Archive checkpoint identity mismatch")

        for relative, expected in record["sha256"].items():
            if file_hash(target / relative) != expected:
                raise ValueError("Archived Student checksum mismatch")
        return target
    temporary = target.with_name(target.name + ".incomplete")
    if temporary.exists():
        raise FileExistsError(f"Incomplete archive needs inspection: {temporary}")
    shutil.copytree(source, temporary, symlinks=False)
    hashes = {}
    for path in source.rglob("*"):
        if path.is_file():
            relative = str(path.relative_to(source))
            expected = file_hash(path)
            if file_hash(temporary / relative) != expected:
                raise IOError("Student archive copy failed checksum verification")
            hashes[relative] = expected
    (temporary / "archive_manifest.json").write_text(json.dumps({
        "checkpoint": str(checkpoint.resolve()), "sha256": hashes,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "volatile_storage": str(archive_root.resolve()).startswith("/dev/shm/"),
    }, indent=2) + "\n")
    temporary.replace(target)
    return target


def apply_storage_policy(output, checkpoint, archive_root, keep_resumable=2):
    if keep_resumable < 1:
        raise ValueError("Retain at least one resumable checkpoint")
    output, checkpoint = Path(output), Path(checkpoint)
    archive_student(checkpoint, archive_root)
    candidates = sorted(
        [p for p in (output / "checkpoints").glob("step_*")
         if p.name[5:].isdigit() and (p / "optimizer.pt").is_file()],
        key=lambda p: int(p.name[5:]),
    )
    for old in candidates[:-keep_resumable]:
        archived = archive_student(old, archive_root)
        if not (old / "storage_retired.json").exists():
            (old / "storage_retired.json").write_text(json.dumps({
                "student_archive": str(archived), "resumable": False,
                "reason": "baseline storage policy: retain two latest complete resume states",
            }) + "\n")
        student = old / "student"
        if not student.is_symlink():
            shutil.rmtree(student)
            student.symlink_to(archived, target_is_directory=True)
        (old / "optimizer.pt").unlink()
        for state in old.glob("rank_*.pt"):
            state.unlink()
    (output / "checkpoint_storage.json").write_text(json.dumps({
        "archive_root": str(Path(archive_root).resolve()),
        "archive_is_volatile": str(Path(archive_root).resolve()).startswith("/dev/shm/"),
        "keep_resumable": keep_resumable,
        "resumable_checkpoints": [str(p) for p in candidates[-keep_resumable:]],
        "latest_checkpoint": str(checkpoint),
    }, indent=2) + "\n")


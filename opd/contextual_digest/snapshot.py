import hashlib
import json
import os
from pathlib import Path


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def seal_snapshot(directory, metadata):
    directory = Path(directory)
    files = {}
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            files[str(path.relative_to(directory))] = file_sha256(path)
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
    record = {**metadata, "format_version": 1, "files_sha256": files}
    target = directory / "snapshot_manifest.json"
    with target.open("x") as handle:
        handle.write(json.dumps(record, indent=2) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    for path in [*(p for p in directory.rglob("*") if p.is_dir()), directory]:
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    return record


def verify_snapshot(directory, world):
    directory = Path(directory)
    record = json.loads((directory / "snapshot_manifest.json").read_text())
    if record["format_version"] != 1 or record["world_size"] != world:
        raise ValueError("Checkpoint format or rank topology differs")
    required = {"optimizer.pt", "config.json", *(f"rank_{r}.pt" for r in range(world))}
    if not required <= set(record["files_sha256"]):
        raise ValueError("Checkpoint is missing optimizer, configuration or rank state")
    for relative, expected in record["files_sha256"].items():
        path = directory / relative
        if not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError("Checkpoint manifest path escapes its directory")
        if not path.is_file() or file_sha256(path) != expected:
            raise ValueError(f"Checkpoint checksum mismatch: {relative}")
    return record


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".incomplete")
    with temporary.open("w") as handle:


        handle.write(json.dumps(value, ensure_ascii=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def publish_update(output, rank, metrics, audit, *, replay=False):

    output = Path(output)
    path = output / (f"replay_metrics_rank_{rank}.jsonl" if replay else f"metrics_rank_{rank}.jsonl")
    rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    prior = [row for row in rows if row["step"] == metrics["step"]]
    if prior and (len(prior) != 1 or prior[0] != metrics):
        raise ValueError("Existing metrics disagree with the committed checkpoint")
    if any(row["step"] > metrics["step"] for row in rows):
        raise ValueError("Output contains newer updates; select its latest committed checkpoint")
    if not prior:
        rows.append(metrics)
        temporary = path.with_name(path.name + ".incomplete")
        with temporary.open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    atomic_json(output / ("replay_audit" if replay else "audit") / f"step_{metrics['step']}" / f"rank_{rank}.json", audit)

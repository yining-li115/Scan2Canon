"""Checksummed, bounded tool stages shared by canonicalization workers."""

from collections.abc import Callable
from pathlib import Path
from time import perf_counter
from typing import Any

from experiments.artifacts import sha256, write_json
from experiments.canonicalization.trace import event
from experiments.orientation.common import checked_file


def execute_stage(
    root: Path, state: dict[str, Any], name: str, execute: Callable[[Path], Path]
) -> Path:
    attempts = state["tools"].setdefault(name, [])
    if attempts and attempts[-1]["status"] == "succeeded":
        record = attempts[-1]
        directory = root / str(record["path"])
        for path, digest in record["files"].items():
            checked_file(directory, path, digest)
        event(root, "stage_reused", stage=name, path=record["path"])
        return directory
    if len(attempts) >= 2:
        raise RuntimeError(f"Tool attempt budget exhausted: {name}")
    directory = root / "work" / f"{name}-{len(attempts) + 1}"
    directory.parent.mkdir(exist_ok=True)
    record = {"status": "started", "path": str(directory.relative_to(root))}
    attempts.append(record)
    state["status"] = "running"
    write_json(root / "state.json", state)
    event(root, "stage_started", stage=name, attempt=len(attempts), path=record["path"])
    started = perf_counter()
    try:
        execute(directory)
    except Exception as error:
        event(
            root,
            "stage_failed",
            stage=name,
            error_type=type(error).__name__,
            elapsed_seconds=perf_counter() - started,
        )
        raise
    record.update(
        status="succeeded",
        files={
            str(p.relative_to(directory)): sha256(p)
            for p in sorted(directory.rglob("*"))
            if p.is_file()
        },
    )
    write_json(root / "state.json", state)
    event(
        root,
        "stage_completed",
        stage=name,
        path=record["path"],
        elapsed_seconds=perf_counter() - started,
        file_count=len(record["files"]),
    )
    return directory

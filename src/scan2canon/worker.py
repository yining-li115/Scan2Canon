"""Single-host worker. The OS releases its exclusive lock after a crash."""

import fcntl
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from scan2canon.domain import Job
from scan2canon.runtime import Runtime


@contextmanager
def worker_lock(root: Path) -> Iterator[None]:
    with (root / "worker.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another worker is already using this data directory") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def run_once(runtime: Runtime) -> Job | None:
    with worker_lock(runtime.root):
        runtime.jobs.recover_interrupted()
        job = runtime.jobs.claim()
        return runtime.harness.run(job) if job else None

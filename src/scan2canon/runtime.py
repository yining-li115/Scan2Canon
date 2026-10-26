"""Composition root: concrete adapters are selected in this module only."""

import os
from dataclasses import dataclass
from pathlib import Path

from scan2canon.domain import Stage
from scan2canon.harness import Harness
from scan2canon.policy import BaselinePolicy
from scan2canon.ports import Tool
from scan2canon.service import ScanService
from scan2canon.storage.artifacts import LocalArtifacts
from scan2canon.storage.jobs import SQLiteJobs
from scan2canon.tools.inspect import InspectCapture


@dataclass(frozen=True)
class Runtime:
    root: Path
    jobs: SQLiteJobs
    artifacts: LocalArtifacts
    service: ScanService
    harness: Harness


def build_runtime(root: Path | None = None, tools: dict[Stage, Tool] | None = None) -> Runtime:
    root = (root or Path(os.environ.get("SCAN2CANON_DATA_DIR", ".scan2canon"))).resolve()
    artifacts = LocalArtifacts(root / "artifacts")
    jobs = SQLiteJobs(root / "jobs.sqlite3")
    registry: dict[Stage, Tool] = {Stage.INSPECT: InspectCapture()}
    if tools:
        registry.update(tools)
    return Runtime(
        root=root,
        jobs=jobs,
        artifacts=artifacts,
        service=ScanService(jobs, artifacts),
        harness=Harness(jobs, artifacts, BaselinePolicy(), registry),
    )

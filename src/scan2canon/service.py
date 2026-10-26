"""Application operations shared by CLI and HTTP entrypoints."""

from pathlib import Path
from uuid import UUID

from scan2canon.domain import Capture, Job
from scan2canon.ports import ArtifactStore, JobRepository


class ScanService:
    def __init__(self, jobs: JobRepository, artifacts: ArtifactStore):
        self.jobs = jobs
        self.artifacts = artifacts

    def import_capture(self, source: Path) -> Capture:
        return self.artifacts.import_capture(source)

    def submit(self, capture_id: UUID, max_attempts: int = 12) -> Job:
        self.artifacts.capture(capture_id)
        job = Job(capture_id=capture_id, max_attempts=max_attempts)
        self.jobs.create(job)
        return job

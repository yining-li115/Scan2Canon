"""Boundaries implemented by storage adapters and compute tools."""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from uuid import UUID

from scan2canon.domain import Capture, Event, Job, Stage, StageResult
from scan2canon.library import Assessment, MatchProposal, ObjectRecord


class ArtifactStore(Protocol):
    def read(self, key: str) -> bytes: ...

    def write(self, job_id: UUID, name: str, content: bytes) -> str: ...

    def exists(self, key: str) -> bool: ...

    def capture(self, capture_id: UUID) -> Capture: ...

    def import_capture(self, source: Path) -> Capture: ...


class JobRepository(Protocol):
    def create(self, job: Job) -> None: ...

    def get(self, job_id: UUID) -> Job: ...

    def save(self, job: Job, event: Event) -> None: ...

    def claim(self) -> Job | None: ...

    def events(self, job_id: UUID) -> list[Event]: ...

    def retry(self, job_id: UUID) -> Job: ...

    def recover_interrupted(self) -> int: ...


@dataclass(frozen=True)
class ToolContext:
    job: Job
    capture: Capture
    artifacts: ArtifactStore


class Tool(Protocol):
    version: str

    def execute(self, context: ToolContext) -> StageResult: ...


@dataclass(frozen=True)
class Decision:
    stage: Stage
    reason: str


class Policy(Protocol):
    def decide(self, job: Job) -> Decision: ...


class ToolUnavailable(RuntimeError):
    """A required adapter is not configured."""


class ReviewRequired(RuntimeError):
    """Evidence requires review before this stage may complete."""


class ObjectLibrary(Protocol):
    """Immutable evidence-backed proposals; no automatic category admission."""

    def ingest(self, item: "ObjectRecord", assessment: "Assessment") -> None: ...

    def objects(self, limit: int = 100, offset: int = 0) -> list["ObjectRecord"]: ...

    def get_object(self, object_id: UUID) -> "ObjectRecord": ...

    def assessments(self, object_id: UUID) -> list["Assessment"]: ...

    def get_assessment(self, assessment_id: UUID) -> "Assessment": ...

    def propose_match(self, proposal: "MatchProposal") -> None: ...

    def matches(self, object_id: UUID) -> list["MatchProposal"]: ...

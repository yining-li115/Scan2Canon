"""Pipeline contracts, independent of HTTP, storage engines and model SDKs."""

import math
from datetime import UTC, datetime
from enum import StrEnum
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


def utc_now() -> datetime:
    return datetime.now(UTC)


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Stage(StrEnum):
    INSPECT = "inspect"
    RECONSTRUCT = "reconstruct"
    MESH = "mesh"
    CANONICALIZE = "canonicalize"
    VALIDATE = "validate"
    EXPORT = "export"


STAGES = tuple(Stage)
REQUIRED_OUTPUT = {
    Stage.INSPECT: "capture_report",
    Stage.RECONSTRUCT: "reconstruction",
    Stage.MESH: "mesh",
    Stage.CANONICALIZE: "pose",
    Stage.VALIDATE: "quality_report",
    Stage.EXPORT: "asset",
}


class Status(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    BLOCKED = "blocked"
    NEEDS_REVIEW = "needs_review"
    FAILED = "failed"
    SUCCEEDED = "succeeded"


class Frame(Model):
    key: str
    sha256: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class Capture(Model):
    id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=utc_now)
    frames: tuple[Frame, ...] = Field(min_length=3, max_length=200)


class StageResult(Model):
    outputs: dict[str, str] = Field(default_factory=dict)
    metrics: dict[str, float] = Field(default_factory=dict)
    # Only the validation stage may authorize export.
    quality_passed: bool | None = None


class Job(Model):
    id: UUID = Field(default_factory=uuid4)
    capture_id: UUID
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    status: Status = Status.QUEUED
    completed: tuple[Stage, ...] = ()
    outputs: dict[str, str] = Field(default_factory=dict)
    attempts: int = Field(default=0, ge=0)
    max_attempts: int = Field(default=12, ge=1, le=100)
    message: str | None = None
    pipeline_version: str = "1"

    @model_validator(mode="after")
    def validate_checkpoint(self) -> "Job":
        if self.completed != STAGES[: len(self.completed)]:
            raise ValueError("Completed stages must be a contiguous pipeline prefix")
        if self.status == Status.SUCCEEDED and self.completed != STAGES:
            raise ValueError("A successful job must have completed every stage")
        if self.attempts > self.max_attempts:
            raise ValueError("Attempts cannot exceed the job budget")
        return self

    @property
    def next_stage(self) -> Stage | None:
        return STAGES[len(self.completed)] if len(self.completed) < len(STAGES) else None


class Event(Model):
    at: datetime = Field(default_factory=utc_now)
    kind: str
    stage: Stage | None = None
    message: str
    tool_version: str | None = None
    duration_seconds: float | None = None
    metrics: dict[str, float] = Field(default_factory=dict)


Vector = tuple[float, float, float]
Matrix = tuple[Vector, Vector, Vector]


class CanonicalPose(Model):
    """Column-vector convention: canonical = rotation @ source + translation.

    No scale is inferred. Axis conversion and semantic canonicalization must be
    documented by convention_id. Symmetry is retained rather than hidden.
    """

    rotation: Matrix
    translation: Vector = (0.0, 0.0, 0.0)
    convention_id: str = Field(min_length=1)
    category: str = Field(min_length=1)
    symmetry: str = Field(min_length=1)
    scale_known: bool = False

    @model_validator(mode="after")
    def validate_rigid_transform(self) -> "CanonicalPose":
        values = [value for row in self.rotation for value in row] + list(self.translation)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Pose values must be finite")
        for i, left in enumerate(self.rotation):
            for j, right in enumerate(self.rotation):
                dot = sum(a * b for a, b in zip(left, right, strict=True))
                if not math.isclose(dot, float(i == j), abs_tol=1e-5):
                    raise ValueError("Rotation must be orthonormal")
        a, b, c = self.rotation
        determinant = (
            a[0] * (b[1] * c[2] - b[2] * c[1])
            - a[1] * (b[0] * c[2] - b[2] * c[0])
            + a[2] * (b[0] * c[1] - b[1] * c[0])
        )
        if not math.isclose(determinant, 1.0, abs_tol=1e-5):
            raise ValueError("Rotation must preserve handedness")
        return self

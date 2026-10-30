"""Instance direction meanings and observed landmarks, without category definitions."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class DirectionHypothesis(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    id: str = Field(min_length=1, max_length=60)
    axis: Literal["up", "front"]
    role: Literal["upright", "functional", "presentation", "structural", "model_reference"]
    description: str = Field(min_length=1, max_length=600)
    from_part: str = Field(min_length=1, max_length=150)
    to_part: str = Field(min_length=1, max_length=150)
    evidence_ids: list[str] = Field(min_length=1, max_length=12)


class DirectionIssue(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    axis: Literal["up", "front"]
    kind: Literal["missing_observation", "competing_definitions", "conflicting_observations"]
    detail: str = Field(min_length=1, max_length=600)
    evidence_ids: list[str] = Field(min_length=1, max_length=12)


class SymmetryHypothesis(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    yaw: Literal["unknown", "none", "discrete", "continuous"]
    basis: Literal["geometry", "appearance", "both"]
    evidence_ids: list[str] = Field(max_length=12)
    explanation: str = Field(min_length=1, max_length=600)
    counterevidence: str = Field(max_length=600)


class ImagePoint(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)


class DirectionMeasurement(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    hypothesis_id: str
    candidate_id: int = Field(ge=0)
    image_id: str
    tail: ImagePoint
    head: ImagePoint
    uncertainty: float = Field(ge=0.005, le=0.25)


class DirectionAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    hypotheses: list[DirectionHypothesis] = Field(max_length=6)
    selected_up: str | None
    selected_front: str | None
    issues: list[DirectionIssue] = Field(max_length=6)
    symmetry: SymmetryHypothesis
    measurements: list[DirectionMeasurement] = Field(max_length=8)

    @model_validator(mode="after")
    def references(self) -> "DirectionAssessment":
        hypotheses = {h.id: h for h in self.hypotheses}
        if len(hypotheses) != len(self.hypotheses):
            raise ValueError("Direction hypothesis IDs must be unique")
        for axis, selected in (("up", self.selected_up), ("front", self.selected_front)):
            if selected is not None and (
                selected not in hypotheses or hypotheses[selected].axis != axis
            ):
                raise ValueError("Selected direction must reference a hypothesis of the same axis")
        if any(m.hypothesis_id not in hypotheses for m in self.measurements):
            raise ValueError("Measurement references an unknown direction hypothesis")
        for issue in self.issues:
            if (
                issue.kind == "competing_definitions"
                and sum(h.axis == issue.axis for h in self.hypotheses) < 2
            ):
                raise ValueError("Competing definitions require at least two direction hypotheses")
        if self.symmetry.yaw != "unknown" and not self.symmetry.evidence_ids:
            raise ValueError("A symmetry hypothesis needs visible evidence")
        return self

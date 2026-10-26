"""Versioned conventions, geometric frames, and strict VLM decisions."""

import json
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from experiments.orientation.geometry import Matrix, rotation


class Rule(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    id: str = Field(min_length=1)
    version: int = Field(ge=1)
    category: str
    status: Literal["draft", "fixture"]
    up: str = Field(min_length=1)
    front: str = Field(min_length=1)
    pointso_up: str = Field(min_length=1)
    pointso_front: str = Field(min_length=1)
    required_part: str = Field(min_length=1)
    unique_front: bool


class Selection(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    decision: Literal["selected", "needs_evidence", "no_unique_front"]
    candidate_id: int | None
    evidence_view_ids: list[int]
    visible_evidence: str
    missing_evidence: str

    @model_validator(mode="after")
    def check_decision(self) -> "Selection":
        if self.decision == "selected":
            if (
                self.candidate_id is None
                or not self.evidence_view_ids
                or not self.visible_evidence.strip()
            ):
                raise ValueError("A selection needs a candidate and cited visual evidence")
        elif self.candidate_id is not None:
            raise ValueError("Abstention must not carry a selected candidate")
        if self.decision == "needs_evidence" and not self.missing_evidence.strip():
            raise ValueError("An evidence request must describe what is missing")
        return self


def parse_selection(text: str, valid_ids: set[int]) -> Selection:
    # Accept a single fenced JSON block, but never salvage a fragment from prose.
    text = text.strip()
    if text.startswith("```json\n") and text.endswith("```"):
        text = text[8:-3].strip()
    result = Selection.model_validate(json.loads(text))
    if result.candidate_id is not None and result.candidate_id not in valid_ids:
        raise ValueError("Unknown candidate")
    if not set(result.evidence_view_ids).issubset(valid_ids):
        raise ValueError("Unknown evidence view")
    return result


def unit(value: Any) -> Matrix:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError("Expected a finite 3D direction")
    length = np.linalg.norm(vector)
    if length < 1e-6:
        raise ValueError("Direction has zero length")
    return vector / length


def frame(up: Any, front: Any) -> Matrix:
    up = unit(up)
    front = unit(front)
    if abs(float(np.dot(front, up))) > 0.5:
        raise ValueError("Semantic up/front must be within 30 degrees of orthogonal")
    front = front - np.dot(front, up) * up
    if np.linalg.norm(front) < 0.1:
        raise ValueError("Up and front are nearly parallel")
    front = unit(front)
    return rotation(np.stack([np.cross(up, front), up, front]))


def direction_error(first: Any, second: Any) -> float:
    return float(np.degrees(np.arccos(np.clip(unit(first) @ unit(second), -1, 1))))


def selection_consensus(responses: list[dict[str, Any]]) -> int | None:
    if len(responses) != 2:
        return None
    decisions = [r.get("parsed") for r in responses]
    if any(d is None or d["decision"] != "selected" for d in decisions):
        return None
    ids = [d["candidate_id"] for d in decisions if d is not None]
    return int(ids[0]) if len(set(ids)) == 1 else None

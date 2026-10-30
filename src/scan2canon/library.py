"""Object-library contracts. Stored model claims are proposals, never category standards."""

from typing import Annotated, Literal
from uuid import UUID

from pydantic import Field, model_validator

from scan2canon.domain import Model

Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Text = Annotated[str, Field(min_length=1, max_length=2000)]


class EvidenceRef(Model):
    id: Text
    location: Text
    sha256: Digest
    kind: Literal["mesh", "image", "report", "model_reply", "request"]


class ObjectRecord(Model):
    id: UUID
    name: Text
    source_mesh_sha256: Digest
    # A reconstruction instance, not automatic physical-object deduplication.


class IdentityClaim(Model):
    functional_identity: Text
    appearance: Text
    parts: tuple[Text, ...]
    evidence_ids: tuple[Text, ...] = Field(min_length=1)


class DirectionClaim(Model):
    id: Text
    axis: Literal["up", "front"]
    role: Literal["upright", "functional", "presentation", "structural", "model_reference"]
    description: Text
    from_part: Text
    to_part: Text
    evidence_ids: tuple[Text, ...] = Field(min_length=1)


class Assessment(Model):
    schema_version: Literal[1] = 1
    id: UUID
    object_id: UUID
    state: Literal["proposed"] = "proposed"
    producer: Text
    model: Text
    implementation_sha256: Digest
    report_sha256: Digest
    evidence: tuple[EvidenceRef, ...] = Field(min_length=3)
    identities: tuple[IdentityClaim, ...] = Field(min_length=1, max_length=3)
    directions: tuple[DirectionClaim, ...]
    semantic_status: Literal["resolved", "uncertain"]
    pose_scope: Literal["full_frame", "up_only", "unresolved"]
    selected_up: str | None
    selected_front: str | None

    @model_validator(mode="after")
    def references(self) -> "Assessment":
        evidence = {e.id: e for e in self.evidence}
        directions = {d.id: d for d in self.directions}
        if len(evidence) != len(self.evidence) or len(directions) != len(self.directions):
            raise ValueError("Evidence and direction IDs must be unique within an assessment")
        citations = [h.evidence_ids for h in self.identities] + [
            d.evidence_ids for d in self.directions
        ]
        for ids in citations:
            if any(i not in evidence or evidence[i].kind != "image" for i in ids):
                raise ValueError("Claims must reference registered image evidence")
        if not any(e.kind == "report" and e.sha256 == self.report_sha256 for e in self.evidence):
            raise ValueError("Assessment must retain its source report")
        for axis, selected in (("up", self.selected_up), ("front", self.selected_front)):
            if selected is not None and (
                selected not in directions or directions[selected].axis != axis
            ):
                raise ValueError("Selected direction must reference the matching axis")
        if self.pose_scope != "unresolved" and self.selected_up is None:
            raise ValueError("Exported proposals must identify their up meaning")
        if self.pose_scope == "full_frame" and self.selected_front is None:
            raise ValueError("Full-frame proposals must identify their front meaning")
        if self.pose_scope == "up_only" and self.selected_front is not None:
            raise ValueError("Up-only proposals cannot establish front")
        return self


class AxisCorrespondence(Model):
    axis: Literal["up", "front"]
    left_direction_id: Text
    right_direction_id: Text
    rationale: Text


class SharedAxisMeaning(Model):
    axis: Literal["up", "front"]
    from_part: Text
    to_part: Text


class SharedFrameDefinition(Model):
    """A proposed agreement for these instances, not an established category standard."""

    version: Literal[1] = 1
    label: Text
    applicability: tuple[Text, ...] = Field(min_length=1, max_length=8)
    axes: tuple[SharedAxisMeaning, ...] = Field(min_length=1, max_length=2)
    alternatives: Text

    @model_validator(mode="after")
    def axis_set(self) -> "SharedFrameDefinition":
        axes = [a.axis for a in self.axes]
        if len(set(axes)) != len(axes) or "up" not in axes:
            raise ValueError("A frame must define up and at most one front")
        return self


class MatchProposal(Model):
    schema_version: Literal[1] = 1
    id: UUID
    left_assessment_id: UUID
    right_assessment_id: UUID
    state: Literal["proposed"] = "proposed"
    identity_relation: Literal["same_function", "different_function", "uncertain"]
    shared_frame: SharedFrameDefinition | None = None
    correspondences: tuple[AxisCorrespondence, ...]
    evidence: EvidenceRef
    producer: Text
    rationale: Text

    @model_validator(mode="after")
    def distinct(self) -> "MatchProposal":
        if self.left_assessment_id == self.right_assessment_id:
            raise ValueError("A cross-instance match needs two different assessments")
        axes = [c.axis for c in self.correspondences]
        if len(set(axes)) != len(axes):
            raise ValueError("At most one proposed correspondence per axis")
        if self.shared_frame is not None:
            if self.identity_relation != "same_function":
                raise ValueError("This shared-function frame requires a same-function proposal")
            if set(axes) != {a.axis for a in self.shared_frame.axes}:
                raise ValueError("Every shared-frame axis needs an instance correspondence")
        return self

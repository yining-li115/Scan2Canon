"""Apply an explicit pair-derived frame while preserving the existing evidence gates."""

import json
from pathlib import Path
from typing import Any

from experiments.canonicalization.agent_contracts import AgentTurn
from experiments.canonicalization.agent_controller import AgentController
from experiments.canonicalization.model_policy import InvalidPolicyReply, ModelPolicy, PolicyRequest
from experiments.canonicalization.open_controller import Reply
from experiments.canonicalization.open_tools import OpenGeometryTools
from experiments.orientation.common import verify_views
from scan2canon.library import MatchProposal, SharedAxisMeaning


def validate_frame_choice(reply: AgentTurn, proposal: MatchProposal) -> None:
    if reply.choice is None or reply.choice.scope == "unresolved":
        return
    assert proposal.shared_frame is not None
    definitions: dict[str, SharedAxisMeaning] = {d.axis: d for d in proposal.shared_frame.axes}
    hypotheses = {h.id: h for h in reply.directions.hypotheses}
    for axis in ["up", "front"] if reply.choice.scope == "full_frame" else ["up"]:
        selected = reply.directions.selected_up if axis == "up" else reply.directions.selected_front
        expected = definitions.get(axis)
        hypothesis = hypotheses.get(selected or "")
        if (
            expected is None
            or hypothesis is None
            or selected != "shared-" + axis
            or (
                hypothesis.axis != axis
                or hypothesis.from_part != expected.from_part
                or hypothesis.to_part != expected.to_part
            )
        ):
            raise InvalidPolicyReply(
                "Exported axes must use the exact shared-frame IDs and part meanings; "
                "do not redefine the frame"
            )


class SharedFrameController(AgentController):
    route = "shared-frame-alignment-v1"

    def __init__(
        self,
        policy: ModelPolicy,
        tools: OpenGeometryTools,
        proposal: MatchProposal,
        source_mesh_sha256: str,
    ) -> None:
        if proposal.shared_frame is None:
            raise ValueError("This match has no shared frame to apply")
        super().__init__(policy, tools)
        self.proposal = proposal
        self.source_mesh_sha256 = source_mesh_sha256

    @property
    def policy_identity(self) -> dict[str, str]:
        assert self.proposal.shared_frame is not None
        return {
            **super().policy_identity,
            "shared_frame_proposal_id": str(self.proposal.id),
            "shared_frame_definition": self.proposal.shared_frame.model_dump_json(),
            "source_mesh_sha256": self.source_mesh_sha256,
        }

    def run(
        self,
        views: Path,
        predictions: Path,
        output: Path,
        observations: Path | None = None,
        resume: bool = False,
        max_calls: int = 4,
    ) -> Path:
        if verify_views(views)["mesh_sha256"] != self.source_mesh_sha256:
            raise ValueError("Shared frame is bound to a different source instance")
        return super().run(views, predictions, output, observations, resume, max_calls)

    def _call(
        self,
        root: Path,
        state: dict[str, Any],
        name: str,
        request: PolicyRequest,
        reply_type: type[Reply],
        budget: int,
        policy: ModelPolicy | None = None,
    ) -> Reply:
        assert self.proposal.shared_frame is not None
        instruction = """\nTASK FRAME OVERRIDE:
A prior cross-instance comparison proposed the shared frame below.
Apply it as an explicit task convention to this instance, not a universal intrinsic front.
This task resolves the choice between semantic conventions only; it does not resolve observation,
geometry or pose uncertainty. Other reasonable fronts are alternatives, not an automatic reason
to reject a clearly observed instance of THIS convention. If parts are missing or do not correspond,
return up_only or unresolved. Never force an unsupported candidate to fit the requested frame.
For each exported axis use direction hypothesis ID shared-up or shared-front, with from_part and
to_part strings copied EXACTLY from the definition. Locate the corresponding parts in the actual
images. Direction issues should describe remaining evidence/pose conflicts for this frame.
For full_frame, front_mode unique means a unique meaning under this TASK convention only.
Keep the same geometry, photo, support-refinement and projected-point checks. Front often points
along the primary camera viewing ray; request a side or top render to obtain a measurable relation.
Use the same candidate's up and front evidence. Preserve budget for inspecting refinement and views.
Shared frame (unverified pair proposal):\n"""
        bound = PolicyRequest(
            request.task,
            request.prompt
            + instruction
            + json.dumps(self.proposal.shared_frame.model_dump(mode="json")),
            request.images,
            request.response_schema,
        )
        reply = super()._call(root, state, name, bound, reply_type, budget, policy)
        if isinstance(reply, AgentTurn):
            validate_frame_choice(reply, self.proposal)
        return reply

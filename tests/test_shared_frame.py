"""An explicit shared task meaning cannot silently change during alignment."""

from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

pytest.importorskip("numpy")
from experiments.artifacts import read_json  # noqa: E402
from experiments.canonicalization.agent_contracts import AgentTurn  # noqa: E402
from experiments.canonicalization.model_policy import InvalidPolicyReply  # noqa: E402
from experiments.canonicalization.open_tools import OpenGeometryTools  # noqa: E402
from experiments.canonicalization.shared_frame import SharedFrameController, validate_frame_choice  # noqa: E402
from scan2canon.library import (
    AxisCorrespondence,
    MatchProposal,
    SharedAxisMeaning,
    SharedFrameDefinition,
)  # noqa: E402
from tests.canonical_fixtures import FixturePolicy, pose  # noqa: E402
from tests.canonical_fixtures import bundles as bundles  # noqa: E402
from tests.canonical_fixtures import rendering as rendering  # noqa: E402
from tests.test_canonicalization_agent import turn  # noqa: E402
from tests.test_object_library import fixture_entry  # noqa: E402


def proposal() -> MatchProposal:
    _, a = fixture_entry()
    _, b = fixture_entry()
    return MatchProposal(
        id=uuid4(),
        left_assessment_id=a.id,
        right_assessment_id=b.id,
        identity_relation="same_function",
        evidence=a.evidence[1],
        producer="test-fixture",
        rationale="fixture",
        correspondences=(
            AxisCorrespondence(
                axis="up", left_direction_id="up", right_direction_id="up", rationale="fixture"
            ),
        ),
        shared_frame=SharedFrameDefinition(
            label="fixture shared up",
            applicability=("fixture only",),
            axes=(SharedAxisMeaning(axis="up", from_part="start", to_part="end"),),
            alternatives="Front unspecified",
        ),
    )


def aligned_turn() -> dict[str, Any]:
    reply = turn(choice=pose(scope="up_only"))
    d = reply["directions"]
    d["selected_up"] = "shared-up"
    d["hypotheses"][0].update(id="shared-up", from_part="start", to_part="end")
    d["measurements"][0]["hypothesis_id"] = "shared-up"
    return reply


def test_meaning_change_is_rejected_then_corrected_in_same_budget(
    bundles: tuple[Path, Path],
    rendering: list[Path],
    tmp_path: Path,
) -> None:
    frame = proposal()
    bad = aligned_turn()
    bad["directions"]["hypotheses"][0]["to_part"] = "different part"
    with pytest.raises(InvalidPolicyReply, match="part meanings"):
        validate_frame_choice(AgentTurn.model_validate(bad), frame)
    policy = FixturePolicy([bad, aligned_turn()])
    digest = read_json(bundles[0] / "manifest.json")["mesh_sha256"]
    controller = SharedFrameController(policy, OpenGeometryTools("fixture"), frame, digest)
    output = controller.run(*bundles, tmp_path / "job", max_calls=2)
    result = read_json(output / "canonicalization.json")
    assert result["scope"] == "up_only"
    assert result["policy"]["shared_frame_proposal_id"] == str(frame.id)
    assert len(policy.requests) == 2
    assert "TASK FRAME OVERRIDE" in policy.requests[0].prompt
    assert "validation_error" in policy.requests[1].prompt
    assert controller.run(*bundles, tmp_path / "job", max_calls=2, resume=True) == output
    assert len(policy.requests) == 2


def test_source_binding_blocks_unrelated_mesh(
    bundles: tuple[Path, Path],
    rendering: list[Path],
    tmp_path: Path,
) -> None:
    policy = FixturePolicy([])
    controller = SharedFrameController(policy, OpenGeometryTools("fixture"), proposal(), "0" * 64)
    with pytest.raises(ValueError, match="different source"):
        controller.run(*bundles, tmp_path / "job")
    assert not policy.requests

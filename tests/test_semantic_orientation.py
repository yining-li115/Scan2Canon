import json
from typing import Any

import numpy as np
import pytest

from experiments.orientation.geometry import angular_error, axis_rotation
from experiments.semantic_orientation.contracts import (
    direction_error,
    frame,
    parse_selection,
    selection_consensus,
)
from experiments.semantic_orientation.evaluate import methods


def reply(**changes: Any) -> str:
    return json.dumps(
        {
            "decision": "selected",
            "candidate_id": 2,
            "evidence_view_ids": [2],
            "visible_evidence": "The backrest lies behind the open seat edge.",
            "missing_evidence": "",
            **changes,
        }
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"candidate_id": 8},
        {"candidate_id": "2"},
        {"evidence_view_ids": [99]},
        {"evidence_view_ids": []},
        {"decision": "needs_evidence"},
        {"decision": "needs_evidence", "candidate_id": None, "missing_evidence": ""},
        {"approved": True},
    ],
)
def test_vlm_cannot_invent_candidates_or_bypass_review(changes: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        parse_selection(reply(**changes), set(range(8)))


def test_abstention_and_order_conflict_never_select_a_pose() -> None:
    chosen = parse_selection(reply(), set(range(8))).model_dump()
    other = parse_selection(reply(candidate_id=3), set(range(8))).model_dump()
    abstain = parse_selection(
        reply(
            decision="needs_evidence",
            candidate_id=None,
            missing_evidence="Show paper-entry slot",
        ),
        set(range(8)),
    ).model_dump()
    assert selection_consensus([{"parsed": chosen}, {"parsed": chosen}]) == 2
    for second in (other, abstain, None):
        assert selection_consensus([{"parsed": chosen}, {"parsed": second}]) is None


def test_semantic_directions_recover_known_rotation_and_handedness() -> None:
    q = axis_rotation(0, 41) @ axis_rotation(1, 67) @ axis_rotation(2, -23)
    result = frame(q[:, 1], q[:, 2])
    np.testing.assert_allclose(result @ q, np.eye(3), atol=1e-7)
    assert np.linalg.det(result) == pytest.approx(1)
    assert direction_error(result[2], q[:, 2]) < 1e-6


@pytest.mark.parametrize("front", [[0, 1, 0], [0, 0, 0], [np.nan, 0, 1]])
def test_degenerate_semantic_axes_do_not_export_a_frame(front: list[float]) -> None:
    with pytest.raises(ValueError):
        frame([0, 1, 0], front)


def test_vlm_abstention_preserves_independent_baselines() -> None:
    case = {
        "oa_rotations": [np.eye(3).tolist()] * 3,
        "support": {"up_in_input_frame": [0, 1, 0]},
        "candidates": [{"id": 0, "rotation": np.eye(3).tolist()}],
    }
    pointso = {"queries": {"up": {"direction": [0, 1, 0]}, "front": {"direction": [0, 0, 1]}}}
    result = methods(case, pointso, {"consistent_candidate_id": None})
    assert result["vlm_support"] is None
    assert angular_error(result["pointso_raw"], np.eye(3)) == pytest.approx(0)

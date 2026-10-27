from pathlib import Path
from typing import Any

import pytest

np = pytest.importorskip("numpy")

from experiments.artifacts import write_json  # noqa: E402
from experiments.canonicalization.consistency import common_frame, comparisons, run  # noqa: E402
from experiments.orientation.geometry import axis_rotation  # noqa: E402


def row(key: str, matrix: Any, scope: str = "full_frame") -> dict[str, Any]:
    return {"case_id": key, "common_rotation": matrix, "scope": scope}


def test_input_rotations_are_removed_in_the_correct_order() -> None:
    target = axis_rotation(0, 27) @ axis_rotation(1, 43)
    rows = []
    predictions = []
    for index, q in enumerate([axis_rotation(2, 73), axis_rotation(0, -54)]):
        prediction = target @ q.T
        predictions.append(prediction)
        recovered = common_frame(prediction, q)
        np.testing.assert_allclose(recovered, target, atol=1e-12)
        rows.append(row(str(index), recovered))
    assert not np.allclose(*predictions)
    assert comparisons(rows)["all_rotations_frame_consistent"]


def test_yaw_is_ignored_only_when_front_is_unspecified() -> None:
    poses = [np.eye(3), axis_rotation(1, 180)]
    up = comparisons([row(str(i), r, "up_only") for i, r in enumerate(poses)])
    assert up["all_rotations_up_consistent"]
    assert up["max_frame_difference_degrees"] is None
    assert not up["all_rotations_frame_consistent"]
    full = comparisons([row(str(i), r) for i, r in enumerate(poses)])
    assert full["all_rotations_up_consistent"]
    assert full["max_frame_difference_degrees"] == pytest.approx(180)
    assert not full["all_rotations_frame_consistent"]


def test_abstentions_do_not_count_as_consistency_success() -> None:
    missing = row("missing", None, "unresolved")
    result = comparisons([missing, row("also-missing", None, "unresolved")])
    assert result["measurement"] == "not_measurable"
    assert result["decision_scopes_consistent"]
    assert not result["all_rotations_up_consistent"]
    mixed = comparisons([missing, row("a", np.eye(3)), row("b", np.eye(3))])
    assert mixed["up_pair_count"] == 1
    assert not mixed["all_rotations_up_consistent"]


def test_consistently_wrong_rotation_is_not_a_correctness_measure() -> None:
    wrong = axis_rotation(0, 180)
    result = comparisons([row("a", wrong), row("b", wrong)])
    assert result["all_rotations_frame_consistent"]
    assert np.dot(wrong[1], [0, 1, 0]) == pytest.approx(-1)


def test_missing_decision_is_detected_before_reference_access(tmp_path: Path) -> None:
    protocol = tmp_path / "protocol.json"
    write_json(protocol, {"cases": [{"case_id": "pending", "prior_result": None}]})
    output = tmp_path / "report.json"
    with pytest.raises(FileNotFoundError, match="canonicalization.json"):
        run(protocol, tmp_path / "references-must-not-be-read", output)
    assert not output.exists()


def test_invalid_rotations_and_duplicate_cases_are_rejected() -> None:
    with pytest.raises(ValueError, match="proper rotation"):
        common_frame(np.diag([1, 1, -1]), np.eye(3))
    with pytest.raises(ValueError, match="distinct"):
        comparisons([row("a", np.eye(3)), row("a", np.eye(3))])
    with pytest.raises(ValueError, match="finite"):
        comparisons([row("a", np.eye(3)), row("b", np.eye(3))], float("nan"))

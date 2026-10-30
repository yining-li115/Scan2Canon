"""Numeric projection checks and bounded recovery, independent of object category."""

import math
from pathlib import Path
from typing import Any

import pytest

np = pytest.importorskip("numpy")

from experiments.artifacts import read_json  # noqa: E402
from experiments.canonicalization.agent_controller import AgentController  # noqa: E402
from experiments.canonicalization.direction_checks import (  # noqa: E402
    evaluate,
    projected_relation,
    validate_export,
)
from experiments.canonicalization.direction_contracts import DirectionAssessment  # noqa: E402
from experiments.canonicalization.open_tools import OpenGeometryTools  # noqa: E402
from tests.canonical_fixtures import FixturePolicy, pose  # noqa: E402
from tests.canonical_fixtures import bundles as bundles  # noqa: E402
from tests.canonical_fixtures import rendering as rendering  # noqa: E402
from tests.test_canonicalization_agent import direction_fixture, turn  # noqa: E402


def test_projected_check_matches_pinhole_projection_at_off_center_landmark() -> None:
    fov = 0.8
    tail_3d = np.array([0.3, -0.2, -2.0])
    direction = np.array([0.2, 0.8, 0.1])
    direction /= np.linalg.norm(direction)

    def project(point: Any) -> list[float]:
        return [
            0.5 + point[0] / -point[2] / (2 * math.tan(fov / 2)),
            0.5 - point[1] / -point[2] / (2 * math.tan(fov / 2)),
        ]

    a, b = project(tail_3d), project(tail_3d + 0.5 * direction)
    result = projected_relation(a, b, 0.005, direction.tolist(), fov)
    assert result["status"] == "compatible"
    assert result["angle_degrees"] < 1e-5
    assert projected_relation(a, b, 0.005, (-direction).tolist(), fov)["status"] == "conflict"


@pytest.mark.parametrize(
    "tail,head,uncertainty,axis,reason",
    [
        ([0.5, 0.5], [0.5, 0.2], 0.01, [0, 0, 1], "axis_near_viewing_ray"),
        ([0.5, 0.5], [0.5, 0.49], 0.01, [0, 1, 0], "landmarks_too_close_or_uncertain"),
        ([0.5, 0.5], [0.5, 0.2], 0.1, [0, 1, 0], "landmarks_too_close_or_uncertain"),
    ],
)
def test_insufficient_projection_is_not_a_pass(
    tail: list[float], head: list[float], uncertainty: float, axis: list[float], reason: str
) -> None:
    report = projected_relation(tail, head, uncertainty, axis, 0.8)
    assert report == {"status": "indeterminate", "reason": reason}


def test_no_measurements_or_unregistered_photo_cannot_authorize_export(tmp_path: Path) -> None:
    value = direction_fixture(pose())
    image = tmp_path / "photo.png"
    image.write_bytes(b"fixture")
    for h in value["hypotheses"]:
        h["evidence_ids"] = ["photo:one"]
    for m in value["measurements"]:
        m["image_id"] = "photo:one"
    assessment = DirectionAssessment.model_validate(value)
    report = evaluate(assessment, {"candidates": [{"id": 0}]}, {"photo:one": image})
    assert all(c["status"] == "indeterminate" for c in report["checks"])
    assert not report["symmetry_verified"]
    with pytest.raises(ValueError, match="compatible projected part check"):
        validate_export(assessment, report, "full_frame", 0)
    with pytest.raises(ValueError, match="compatible projected part check"):
        validate_export(assessment, {"checks": []}, "up_only", 0)


def test_competing_definitions_are_explicit_and_do_not_certify_symmetry() -> None:
    value = direction_fixture(pose())
    value["issues"] = [
        {
            "axis": "front",
            "kind": "competing_definitions",
            "detail": "Two roles",
            "evidence_ids": ["candidate:0"],
        }
    ]
    with pytest.raises(ValueError, match="at least two"):
        DirectionAssessment.model_validate(value)
    value["hypotheses"].append({**value["hypotheses"][1], "id": "display", "role": "presentation"})
    parsed = DirectionAssessment.model_validate(value)
    assert parsed.symmetry.yaw == "unknown"
    checks = {
        "checks": [
            {"hypothesis_id": h, "candidate_id": 0, "status": "compatible"} for h in ("up", "front")
        ]
    }
    with pytest.raises(ValueError, match="Resolve front"):
        validate_export(parsed, checks, "full_frame", 0)
    parsed.selected_front = None
    validate_export(parsed, checks, "up_only", 0)


def test_bad_interim_measurement_does_not_block_valid_observation(
    bundles: tuple[Path, Path], rendering: list[Path], tmp_path: Path
) -> None:
    bad = direction_fixture(pose())
    bad["measurements"][0]["image_id"] = "candidate:1"
    policy = FixturePolicy(
        [
            turn("inspect_image", directions=bad, image_id="source:0", crop=[0.1, 0.1, 0.9, 0.9]),
            turn(directions=bad),
            turn(),
        ]
    )
    root = tmp_path / "job"
    controller = AgentController(policy, OpenGeometryTools("fixture"))
    result = controller.run(*bundles, root, max_calls=3)
    report = read_json(result / "canonicalization.json")
    assert report["action_history"][0]["action"]["kind"] == "inspect_image"
    assert "crop:crop-1" in dict(policy.requests[1].images)
    assert read_json(root / "direction-checks/agent-1.json")["checks"] == []
    assert "Measurement candidate differs" in policy.requests[2].prompt
    assert report["scope"] == "full_frame"
    assert controller.run(*bundles, root, max_calls=3, resume=True) == result
    assert len(policy.requests) == 3


def test_same_image_uncertainty_cannot_shrink_after_feedback(
    bundles: tuple[Path, Path], rendering: list[Path], tmp_path: Path
) -> None:
    uncertain = direction_fixture(pose(scope="up_only"))
    uncertain["measurements"][0]["uncertainty"] = 0.1
    policy = FixturePolicy(
        [
            turn(directions=uncertain, choice=pose(scope="up_only")),
            turn(choice=pose(scope="up_only")),
        ]
    )
    root = tmp_path / "job"
    controller = AgentController(policy, OpenGeometryTools("fixture"))
    with pytest.raises(ValueError, match="compatible projected part check"):
        controller.run(*bundles, root, max_calls=2)
    check = read_json(root / "direction-checks/agent-2.json")["checks"][0]
    assert check["reported_uncertainty"] == 0.01
    assert check["uncertainty"] == 0.1 and check["uncertainty_floor_applied"]
    assert check["status"] == "indeterminate"
    assert not list(root.glob("work/agent-export*/canonical.glb"))


def test_numeric_conflict_returns_feedback_and_corrected_reply_can_resume(
    bundles: tuple[Path, Path], rendering: list[Path], tmp_path: Path
) -> None:
    wrong = direction_fixture(pose())
    wrong["measurements"][0]["head"] = {"x": 0.5, "y": 0.8}
    policy = FixturePolicy([turn(directions=wrong), turn()])
    controller = AgentController(policy, OpenGeometryTools("fixture"))
    root = tmp_path / "job"
    result = controller.run(*bundles, root, max_calls=2)
    rejected = read_json(root / "direction-checks/agent-1.json")
    assert rejected["checks"][0]["status"] == "conflict"
    assert "direction_check_feedback" in policy.requests[1].prompt
    report = read_json(result / "canonicalization.json")
    assert report["scope"] == "full_frame"
    assert all(c["status"] == "compatible" for c in report["direction_checks"]["checks"])
    assert not report["direction_checks"]["semantic_correctness_verified"]
    assert controller.run(*bundles, root, max_calls=2, resume=True) == result
    assert len(policy.requests) == 2


def test_conflicting_checks_cannot_be_hidden_by_one_passing_check(
    bundles: tuple[Path, Path], rendering: list[Path], tmp_path: Path
) -> None:
    value = direction_fixture(pose())
    value["measurements"].append({**value["measurements"][0], "head": {"x": 0.5, "y": 0.8}})
    policy = FixturePolicy([turn(directions=value)] * 2)
    root = tmp_path / "job"
    with pytest.raises(ValueError, match="without conflicting checks"):
        AgentController(policy, OpenGeometryTools("fixture")).run(*bundles, root, max_calls=2)
    assert not list(root.rglob("canonical.glb"))


def test_model_reference_cannot_bypass_missing_or_conflicting_checks() -> None:
    value = direction_fixture(pose())
    value["hypotheses"][1]["role"] = "model_reference"
    assessment = DirectionAssessment.model_validate(value)
    for status in ("indeterminate", "conflict"):
        checks = {
            "checks": [
                {"hypothesis_id": "up", "candidate_id": 0, "status": "compatible"},
                {"hypothesis_id": "front", "candidate_id": 0, "status": status},
            ]
        }
        with pytest.raises(ValueError, match="compatible projected part check"):
            validate_export(assessment, checks, "full_frame", 0)

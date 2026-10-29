"""Geometric refinement recovery, refusal and agent tool lifecycle."""

from pathlib import Path
from typing import Any

import pytest

np = pytest.importorskip("numpy")
trimesh = pytest.importorskip("trimesh")

from experiments.artifacts import read_json, write_json  # noqa: E402
from experiments.canonicalization.agent_controller import AgentController  # noqa: E402
from experiments.canonicalization.open_tools import OpenGeometryTools  # noqa: E402
from experiments.canonicalization.up_refinement import contact_metrics, refine_up  # noqa: E402
from experiments.orientation.geometry import axis_rotation  # noqa: E402
from tests.canonical_fixtures import FixturePolicy, pose, render_fixture  # noqa: E402
from tests.canonical_fixtures import bundles as bundles  # noqa: E402
from tests.canonical_fixtures import rendering as rendering  # noqa: E402
from tests.test_canonicalization_agent import turn  # noqa: E402


def candidate_set(matrix: Any, front_available: bool = True) -> dict[str, Any]:
    return {
        "candidates": [
            {
                "id": i,
                "rotation": (axis_rotation(1, i * 90) @ matrix).tolist(),
                "front_available": front_available,
                "source_view": i,
                "symmetry_fit": 1 if front_available else 0,
            }
            for i in range(3)
        ]
    }


def test_refinement_recovers_support_after_arbitrary_input_rotation() -> None:
    mesh = trimesh.creation.box(extents=[2, 1, 2])
    q = axis_rotation(0, 31) @ axis_rotation(1, 67) @ axis_rotation(2, -43)
    transform = np.eye(4)
    transform[:3, :3] = q
    mesh.apply_transform(transform)
    before = mesh.vertices.copy()
    original = axis_rotation(0, 14) @ q.T
    result = refine_up(mesh, candidate_set(original), [0, 1, 2])
    assert len(result["new_candidates"]) == 3
    for c in result["new_candidates"]:
        matrix = np.array(c["rotation"])
        np.testing.assert_allclose(matrix[1], q @ [0, 1, 0], atol=1e-7)
        np.testing.assert_allclose(matrix @ matrix.T, np.eye(3), atol=1e-7)
        assert np.linalg.det(matrix) == pytest.approx(1)
        assert c["refinement"]["up_correction_degrees"] == pytest.approx(14)
        assert c["refinement"]["contact_coverage_after"] > 0.99
        assert c["refinement"]["contact_coverage_before"] < 0.1
    np.testing.assert_array_equal(mesh.vertices, before)


def test_refinement_refuses_large_corrections_and_missing_support() -> None:
    mesh = trimesh.creation.box(extents=[2, 1, 2])
    result = refine_up(mesh, candidate_set(axis_rotation(0, 45)), [0, 1, 2])
    assert result["status"] == "no_supported_refinement"
    assert not result["new_candidates"]
    sphere = trimesh.creation.icosphere(subdivisions=1)
    result = refine_up(sphere, candidate_set(np.eye(3)), [0, 1, 2])
    assert not result["new_candidates"]
    with pytest.raises(ValueError, match="unit length"):
        contact_metrics(mesh, [0, 2, 0])


def test_refinement_preserves_unresolved_front_and_projected_parent_direction() -> None:
    mesh = trimesh.creation.box(extents=[2, 1, 2])
    rows = candidate_set(axis_rotation(0, 12), front_available=False)
    result = refine_up(mesh, rows, [0, 1, 2])
    for c in result["new_candidates"]:
        assert not c["front_available"] and c["symmetry_fit"] == 0
        parent = np.array(rows["candidates"][c["refinement"]["parent_candidate_id"]]["rotation"])
        matrix = np.array(c["rotation"])
        expected = parent[2] - (parent[2] @ matrix[1]) * matrix[1]
        np.testing.assert_allclose(matrix[2], expected / np.linalg.norm(expected), atol=1e-7)


def tool_render(mesh: Path, output: Path, blender: str, **options: Any) -> Path:
    return render_fixture(mesh, output, len(options["poses"]), options["poses"])


def test_agent_refinement_is_inspected_before_export_and_replays_without_calls(
    bundles: tuple[Path, Path],
    rendering: list[Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("experiments.canonicalization.agent_tools.render", tool_render)
    policy = FixturePolicy(
        [
            turn("refine_up", candidate_ids=[0, 1, 2]),
            turn(choice=pose(candidate_id=3, evidence_ids=["candidate:3", "candidate:0"])),
        ]
    )
    controller = AgentController(policy, OpenGeometryTools("fixture"))
    root = tmp_path / "job"
    output = controller.run(*bundles, root)
    report = read_json(output / "canonicalization.json")
    assert report["decision"]["candidate_id"] == 3
    assert report["artifact"]["appearance_preserved"]
    assert not report["dataset_approved"]
    assert "candidate:3" in dict(policy.requests[1].images)
    assert len(report["candidates"]["candidates"]) == 6
    assert controller.run(*bundles, root, resume=True) == output
    assert len(policy.requests) == 2


def test_refined_choice_requires_parent_comparison(
    bundles: tuple[Path, Path],
    rendering: list[Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("experiments.canonicalization.agent_tools.render", tool_render)
    policy = FixturePolicy(
        [
            turn("refine_up", candidate_ids=[0, 1, 2]),
            turn(choice=pose(candidate_id=3, evidence_ids=["candidate:3", "candidate:1"])),
        ]
    )
    root = tmp_path / "job"
    with pytest.raises(ValueError, match="original parent"):
        AgentController(policy, OpenGeometryTools("fixture")).run(*bundles, root, max_calls=2)
    assert not list(root.rglob("canonical.glb"))


def test_refinement_render_failure_recovers_tool_without_redispatching_model(
    bundles: tuple[Path, Path],
    rendering: list[Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken(*args: Any, **kwargs: Any) -> Path:
        raise RuntimeError("Render interrupted")

    monkeypatch.setattr("experiments.canonicalization.agent_tools.render", broken)
    policy = FixturePolicy(
        [
            turn("refine_up", candidate_ids=[0, 1, 2]),
            turn(choice=pose(candidate_id=3, evidence_ids=["candidate:3", "candidate:0"])),
        ]
    )
    controller = AgentController(policy, OpenGeometryTools("fixture"))
    root = tmp_path / "job"
    with pytest.raises(RuntimeError, match="interrupted"):
        controller.run(*bundles, root)
    monkeypatch.setattr("experiments.canonicalization.agent_tools.render", tool_render)
    result = controller.run(*bundles, root, resume=True)
    assert (result / "canonical.glb").is_file()
    assert len(policy.requests) == 2
    assert (root / "work/evidence-1-1").exists()
    assert (root / "work/evidence-1-2/evidence.json").is_file()


def tilt_fixture_predictions(predictions: Path) -> None:
    report = read_json(predictions / "predictions.json")
    for p in report["predictions"]:
        p["angles_degrees"][2] = -12
    write_json(predictions / "predictions.json", report)


def test_export_gate_requires_inspection_of_supported_correction(
    bundles: tuple[Path, Path],
    rendering: list[Path],
    tmp_path: Path,
) -> None:
    tilt_fixture_predictions(bundles[1])
    policy = FixturePolicy([turn(), turn()])
    root = tmp_path / "job"
    with pytest.raises(ValueError, match="Inspect the available support-plane correction"):
        AgentController(policy, OpenGeometryTools("fixture")).run(*bundles, root, max_calls=2)
    assert "up_refinement_review_required" in policy.requests[0].prompt
    assert not list(root.rglob("canonical.glb"))


@pytest.mark.parametrize("compare_refined", [True, False])
def test_agent_can_retain_original_only_after_comparing_refinement(
    bundles: tuple[Path, Path],
    rendering: list[Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    compare_refined: bool,
) -> None:
    tilt_fixture_predictions(bundles[1])
    monkeypatch.setattr("experiments.canonicalization.agent_tools.render", tool_render)
    policy = FixturePolicy(
        [
            turn("refine_up", candidate_ids=[0, 1, 2]),
            turn(
                choice=pose(
                    evidence_ids=[
                        "candidate:0",
                        "candidate:3" if compare_refined else "candidate:1",
                    ]
                )
            ),
        ]
    )
    controller = AgentController(policy, OpenGeometryTools("fixture"))
    root = tmp_path / "job"
    if compare_refined:
        result = controller.run(*bundles, root, max_calls=2)
        assert read_json(result / "canonicalization.json")["decision"]["candidate_id"] == 0
    else:
        with pytest.raises(ValueError, match="compare its refined alternative"):
            controller.run(*bundles, root, max_calls=2)
        assert not list(root.rglob("canonical.glb"))

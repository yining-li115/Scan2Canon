from pathlib import Path
from typing import Any

import pytest

np = pytest.importorskip("numpy")
trimesh = pytest.importorskip("trimesh")

from experiments.artifacts import read_json, sha256, write_json  # noqa: E402
from experiments.canonicalization.evidence import direction_cluster  # noqa: E402
from experiments.canonicalization.semantic import RULES, run  # noqa: E402
from experiments.canonicalization.upright import candidates  # noqa: E402
from experiments.canonicalization.workflow import Decision, export, finish  # noqa: E402
from experiments.orientation.geometry import axis_rotation, rotation  # noqa: E402
from experiments.semantic_orientation.contracts import direction_error  # noqa: E402


def test_direction_consensus_retains_competing_axis() -> None:
    evidence = direction_cluster([[0, 1, 0], [0, 1, 0.1], [0, -1, 0]], 20)
    assert evidence["count"] == 2
    assert evidence["runner_up_count"] == 1
    assert direction_error(evidence["direction"], [0, 1, 0]) < 4
    assert direction_cluster([], 20)["direction"] is None
    with pytest.raises(ValueError, match="zero length"):
        direction_cluster([[0, 0, 0]], 20)


def test_upright_alternatives_cover_rotated_supports_without_changing_mesh() -> None:
    mesh = trimesh.creation.box(extents=[1, 2, 3])
    applied = axis_rotation(0, 38) @ axis_rotation(2, 71)
    mesh.vertices = mesh.vertices @ applied.T
    before = np.array(mesh.vertices)
    choices = candidates(mesh)
    assert 6 <= len(choices) <= 12
    for up in np.vstack([applied.T, -applied.T]):
        assert min(direction_error(up, c["rotation"][1]) for c in choices) < 1e-5
    for choice in choices:
        rotation(choice["rotation"])
    np.testing.assert_array_equal(before, mesh.vertices)


def test_semantic_budget_fails_before_model_load_or_output_creation(tmp_path: Path) -> None:
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="budget"):
        run([tmp_path] * 2, out, tmp_path, budget=2)
    assert not out.exists()
    with pytest.raises(ValueError, match="Repeated"):
        run([tmp_path] * 2, out, tmp_path, budget=4)


def test_export_is_rigid_reversible_and_does_not_overwrite(tmp_path: Path) -> None:
    source = tmp_path / "source.glb"
    mesh = trimesh.creation.box(extents=[1, 2, 4])
    mesh.apply_translation([4, 9, -3])
    mesh.export(source)
    target = tmp_path / "canonical.glb"
    r = axis_rotation(0, 47) @ axis_rotation(1, -15)
    result = export(source, r, target)
    assert result["inverse_vertex_error"] < result["tolerance"]
    moved = trimesh.load_scene(target, process=False).to_mesh()
    assert moved.bounds[0, 1] == pytest.approx(0, abs=1e-6)
    assert moved.area == pytest.approx(mesh.area, rel=1e-6)
    assert moved.volume == pytest.approx(mesh.volume, rel=1e-6)
    before = target.read_bytes()
    with pytest.raises(FileExistsError):
        export(source, r, target)
    assert target.read_bytes() == before
    with pytest.raises(ValueError, match="proper rotation"):
        export(source, np.diag([1, 1, -1]), tmp_path / "invalid.glb")
    assert not (tmp_path / "invalid.glb").exists()


def test_agent_cannot_inject_rotation_or_skip_evidence_budget() -> None:
    valid: dict[str, Any] = {
        "schema_version": 1,
        "actor": "interactive_visual_agent",
        "case_sha256": "a" * 64,
        "rules_sha256": "b" * 64,
        "model_calls": 1,
        "selection": {
            "decision": "selected",
            "candidate_id": 0,
            "evidence_view_ids": [0],
            "visible_evidence": "Rim above base",
            "missing_evidence": "",
        },
    }
    Decision.model_validate(valid)
    with pytest.raises(ValueError):
        Decision.model_validate({**valid, "rotation": np.eye(3).tolist()})
    with pytest.raises(ValueError):
        Decision.model_validate({**valid, "model_calls": 3})
    with pytest.raises(ValueError):
        Decision.model_validate(
            {**valid, "selection": {**valid["selection"], "evidence_view_ids": []}}
        )


def make_case(root: Path, task: str, matrices: list[Any]) -> Path:
    root.mkdir()
    poses = root / "poses"
    poses.mkdir()
    trimesh.creation.box().export(poses / "source.glb")
    views = []
    for i in range(len(matrices)):
        path = poses / f"{i}.png"
        path.write_bytes(b"render fixture")
        views.append({"id": i, "image": path.name, "sha256": sha256(path)})
    write_json(
        poses / "manifest.json",
        {
            "schema_version": 1,
            "status": "succeeded",
            "mesh": "source.glb",
            "mesh_sha256": sha256(poses / "source.glb"),
            "views": views,
        },
    )
    write_json(poses / "poses.json", {"rotations": matrices})
    path = root / ("upright.json" if task == "up" else "front.json")
    write_json(
        path,
        {
            "schema_version": 1,
            "status": "succeeded",
            "task": task,
            "category": "mug",
            "mesh_sha256": sha256(poses / "source.glb"),
            "views_manifest_sha256": sha256(poses / "manifest.json"),
            "candidates": [{"id": i, "rotation": r} for i, r in enumerate(matrices)],
        },
    )
    return path


def save_decision(path: Path, descriptor_path: Path, selection: int | None) -> Path:
    write_json(
        path,
        {
            "schema_version": 1,
            "actor": "interactive_visual_agent",
            "case_sha256": sha256(descriptor_path),
            "rules_sha256": sha256(RULES),
            "model_calls": 1,
            "selection": {
                "decision": "selected" if selection is not None else "needs_evidence",
                "candidate_id": selection,
                "evidence_view_ids": [0],
                "visible_evidence": "Fixture evidence",
                "missing_evidence": "Need a rim view" if selection is None else "",
            },
        },
    )
    return path


def test_workflow_preserves_partial_axes_and_blocks_mismatched_front(tmp_path: Path) -> None:
    identity = np.eye(3).tolist()
    up = make_case(tmp_path / "up", "up", [identity] * 3)
    decision = save_decision(tmp_path / "up-decision.json", up, 0)
    partial = finish(up.parent, decision, tmp_path / "partial")
    report = read_json(partial / "canonicalization.json")
    assert report["scope"] == "up_only_front_pending"
    assert report["yaw"] == "unspecified"
    assert report["dataset_approval"] == "not_granted"
    front = make_case(tmp_path / "front", "front", [axis_rotation(0, 35).tolist()] * 3)
    content = read_json(front)
    content.update(up_case_sha256=sha256(up), up_decision_sha256=sha256(decision))
    write_json(front, content)
    front_decision = save_decision(tmp_path / "front-decision.json", front, 0)
    with pytest.raises(ValueError, match="changes the established up"):
        finish(up.parent, decision, tmp_path / "bad", front.parent, front_decision)
    assert not (tmp_path / "bad").exists()
    evidence = read_json(decision)
    evidence["selection"]["candidate_id"] = 99
    write_json(decision, evidence)
    with pytest.raises(ValueError, match="Unknown candidate"):
        finish(up.parent, decision, tmp_path / "unknown")


def test_abstention_exports_no_mesh_and_changed_evidence_is_rejected(tmp_path: Path) -> None:
    up = make_case(tmp_path / "up", "up", [np.eye(3).tolist()] * 3)
    decision = save_decision(tmp_path / "decision.json", up, None)
    result = finish(up.parent, decision, tmp_path / "result")
    assert read_json(result / "canonicalization.json")["scope"] == "unresolved"
    assert not list(result.glob("*.glb"))
    (up.parent / "poses/0.png").write_bytes(b"changed evidence")
    with pytest.raises(ValueError, match="modified artifact"):
        finish(up.parent, decision, tmp_path / "changed")

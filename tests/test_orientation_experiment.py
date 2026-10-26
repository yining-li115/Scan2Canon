import json
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

np = pytest.importorskip("numpy", reason="Install the canonical extra for geometry tests")
trimesh = pytest.importorskip("trimesh", reason="Install the canonical extra for GLB tests")

from experiments.artifacts import sha256, write_json  # noqa: E402
from experiments.orientation.common import verify_views  # noqa: E402
from experiments.orientation.geometry import (  # noqa: E402
    GLB_TO_BLENDER,
    OA_CAMERA_TO_WORLD,
    OA_TO_CANONICAL,
    axis_rotation,
    mesh_to_oa,
    rotation,
    symmetry_error,
)
from experiments.orientation.solve import hypotheses, solve  # noqa: E402


def test_pose_recovery_for_arbitrarily_rotated_mesh() -> None:
    # Start with canonical OA axes, rotate the entire object and its front-view camera.
    # A perfect front-view estimate is still zero, regardless of the mesh rotation.
    for axis in range(3):
        for degrees in (-135, -31, 0, 73, 179):
            object_rotation = axis_rotation(axis, degrees)
            camera = GLB_TO_BLENDER @ object_rotation @ OA_CAMERA_TO_WORLD
            recovered = mesh_to_oa([0, 0, 0], camera)
            np.testing.assert_allclose(recovered @ object_rotation, np.eye(3), atol=1e-7)


@pytest.mark.parametrize("bad", [np.diag([-1, 1, 1]), np.eye(3) * 2, np.full((3, 3), np.nan)])
def test_reflections_scale_and_nonfinite_matrices_are_rejected(bad: Any) -> None:
    with pytest.raises(ValueError, match="proper rotation"):
        rotation(bad)


def test_symmetry_equivalence_does_not_hide_up_down_flip() -> None:
    assert symmetry_error(np.eye(3), axis_rotation(2, 180), 2) < 1e-6
    assert symmetry_error(np.eye(3), axis_rotation(2, 90), 4) < 1e-6
    assert symmetry_error(np.eye(3), axis_rotation(0, 180), 2) > 179
    assert symmetry_error(np.eye(3), axis_rotation(2, 67), 0) < 1e-6


def synthetic_views() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    camera = GLB_TO_BLENDER @ OA_CAMERA_TO_WORLD
    return (
        {"views": [{"id": i, "camera_to_world": camera.tolist()} for i in range(4)]},
        [{"id": i, "angles_degrees": [0, 0, 0], "symmetry_fit": 1} for i in range(4)],
    )


def test_consensus_preserves_competing_hypotheses() -> None:
    manifest, predictions = synthetic_views()
    predictions[2]["angles_degrees"] = [180, 0, 0]
    predictions[3]["angles_degrees"] = [180, 0, 0]
    result = hypotheses(manifest, predictions)
    assert result["decision"] == "needs_evidence"
    assert len(result["candidates"]) == 2
    assert result["candidates"][0]["support_fraction_all_views"] == 0.5


def test_flat_azimuth_fit_never_becomes_unique_front() -> None:
    manifest, predictions = synthetic_views()
    for prediction in predictions:
        prediction["symmetry_fit"] = 0
    result = hypotheses(manifest, predictions)
    assert result["decision"] == "needs_evidence"
    assert any("azimuth_unresolved" in reason for reason in result["reasons"])


def test_tied_symmetry_modes_are_preserved_independently_of_input_order() -> None:
    manifest, predictions = synthetic_views()
    for prediction in predictions:
        prediction["symmetry_fit"] = 2 if prediction["id"] < 2 else 4
    result = hypotheses(manifest, predictions)
    assert result == hypotheses(manifest, list(reversed(predictions)))
    assert result["decision"] == "needs_evidence"
    assert result["symmetry_fit"] is None
    assert result["front_observable"] is None
    assert result["symmetry_fit_modes"] == [2, 4]
    assert [c["symmetry_fit"] for c in result["candidates"]] == [2, 4]
    assert [c["supporting_views"] for c in result["candidates"]] == [[0, 1], [2, 3]]


def test_missing_or_duplicated_views_cannot_inflate_consensus() -> None:
    manifest, predictions = synthetic_views()
    with pytest.raises(ValueError, match="exactly once"):
        hypotheses(manifest, predictions[:-1] + [predictions[0]])


@pytest.fixture
def portable_run(tmp_path: Path) -> tuple[Path, Path]:
    views, predictions = tmp_path / "views", tmp_path / "inference"
    views.mkdir()
    predictions.mkdir()
    mesh = trimesh.creation.box(extents=[1, 2, 3])
    mesh.apply_translation([2, -3, 4])
    mesh.visual.vertex_colors = [200, 50, 100, 255]
    mesh.export(views / "source.glb")
    manifest, estimates = synthetic_views()
    manifest.update(
        schema_version=1,
        status="succeeded",
        mesh="source.glb",
        mesh_sha256=sha256(views / "source.glb"),
        center_glb=[2, -3, 4],
    )
    for view in manifest["views"]:
        view["image"] = f"view-{view['id']}.png"
        Image.new("RGBA", (32, 32), "white").save(views / view["image"])
        view["sha256"] = sha256(views / view["image"])
    write_json(views / "manifest.json", manifest)
    np.savez(predictions / "logits.npz", fixture=np.zeros((4, 900)))
    write_json(
        predictions / "predictions.json",
        {
            "schema_version": 1,
            "status": "succeeded",
            "predictions": estimates,
            "views_manifest_sha256": sha256(views / "manifest.json"),
            "logits_sha256": sha256(predictions / "logits.npz"),
        },
    )
    return views, predictions


def test_export_preserves_geometry_color_and_pending_review(
    portable_run: tuple[Path, Path], tmp_path: Path
) -> None:
    views, predictions = portable_run
    moved = views.rename(tmp_path / "moved")
    source_digest = sha256(moved / "source.glb")
    output = solve(moved, predictions, tmp_path / "result")
    report = json.loads((output / "canonicalization.json").read_text())
    assert report["status"] == "succeeded"
    assert report["dataset_approval"] == "pending_review"
    before = trimesh.load_scene(moved / "source.glb").to_mesh()
    after = trimesh.load_scene(output / "candidate-01.glb").to_mesh()
    np.testing.assert_allclose(after.bounds.mean(axis=0), 0, atol=1e-6)
    np.testing.assert_allclose(
        after.vertices, (before.vertices - [2, -3, 4]) @ OA_TO_CANONICAL.T, atol=1e-6
    )
    np.testing.assert_array_equal(after.faces, before.faces)
    np.testing.assert_array_equal(after.visual.vertex_colors, before.visual.vertex_colors)
    assert sha256(moved / "source.glb") == source_digest
    with pytest.raises(FileExistsError):
        solve(moved, predictions, output)


def test_tampered_render_and_cross_bundle_predictions_fail(
    portable_run: tuple[Path, Path], tmp_path: Path
) -> None:
    views, predictions = portable_run
    manifest = json.loads((views / "manifest.json").read_text())
    manifest["center_glb"] = [1, 2, 3]
    write_json(views / "manifest.json", manifest)
    with pytest.raises(ValueError, match="different render bundle"):
        solve(views, predictions, tmp_path / "failed")
    failed = json.loads((tmp_path / "failed/canonicalization.json").read_text())
    assert failed["status"] == "failed"
    assert failed["dataset_approval"] == "pending_review"
    (views / "view-0.png").write_bytes(b"modified")
    with pytest.raises(ValueError, match="modified artifact"):
        verify_views(views)

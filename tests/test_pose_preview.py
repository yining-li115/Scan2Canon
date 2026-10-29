"""Presentation axes use the same perspective and glTF basis as the mesh render."""

import math
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
from experiments.artifacts import read_json, sha256, write_json  # noqa: E402
from experiments.orientation.pose_preview import preview, project  # noqa: E402
from tests.canonical_fixtures import bundles as bundles  # noqa: E402


def test_projection_preserves_depth_and_rectangular_image_aspect() -> None:
    points = np.array([[0, 0, -2], [1, 0, -2], [0, 1, -2], [1, 0, -4]])
    pixels = project(points, np.eye(3), np.zeros(3), math.pi / 2, (400, 200))
    assert np.allclose(pixels, [[200, 100], [300, 100], [200, 0], [250, 100]])
    with pytest.raises(ValueError, match="in front"):
        project(np.array([[0, 0, 1]]), np.eye(3), np.zeros(3), math.pi / 2, (400, 200))


def test_gltf_y_up_is_vertical_in_front_camera() -> None:
    basis = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    # Blender camera on -Y, looking at the origin with Z as image-up.
    camera = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    points = np.vstack((np.zeros(3), basis.T))
    pixels = project(points, camera, np.array([0, -4, 0]), math.pi / 2, (400, 400))
    assert np.allclose(pixels, [[200, 200], [250, 200], [200, 150], [200, 200]])


def test_overlay_preserves_evidence_and_marks_candidate_scope(
    bundles: tuple[Path, Path], tmp_path: Path
) -> None:
    bundle = bundles[0]
    data = read_json(bundle / "manifest.json")
    data.update(preset="canonical", center_glb=[0, 0, 0], radius=1)
    for view in data["views"]:
        view.update(label="Synthetic view", camera_position=[0, 0, 4])
    write_json(bundle / "manifest.json", data)
    original = {p.name: sha256(p) for p in bundle.iterdir() if p.is_file()}
    result = preview(bundle, tmp_path / "overlay", "up_only")
    assert result.is_file()
    assert read_json(result.parent / "manifest.json")["scope"] == "up_only"
    assert original == {p.name: sha256(p) for p in bundle.iterdir() if p.is_file()}
    write_json(bundle / "poses.json", {"rotations": [np.eye(3).tolist()] * 3})
    with pytest.raises(ValueError, match="unresolved scope"):
        preview(bundle, tmp_path / "bad", "full_frame")
    assert not (tmp_path / "bad").exists()
    result = preview(bundle, tmp_path / "candidates", "unresolved")
    assert read_json(result.parent / "manifest.json")["candidate_poses_sha256"]

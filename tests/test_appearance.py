"""Appearance provenance and actual rigid-export preservation."""

from pathlib import Path
from typing import Any

import pytest
from PIL import Image

np = pytest.importorskip("numpy")
trimesh = pytest.importorskip("trimesh")
from experiments.appearance import describe, require_color  # noqa: E402
from experiments.canonicalization.workflow import export  # noqa: E402


def test_texture_and_uv_survive_rotation(tmp_path: Path) -> None:
    mesh = trimesh.creation.box()
    texture = Image.new("RGB", (4, 4), "red")
    texture.putpixel((1, 1), (10, 20, 240))
    mesh.visual = trimesh.visual.TextureVisuals(
        uv=np.arange(len(mesh.vertices) * 2).reshape(-1, 2) / 16,
        image=texture,
    )
    source, target = tmp_path / "source.glb", tmp_path / "rotated.glb"
    mesh.export(source)
    before = describe(source)
    require_color(before)
    assert before["has_texture"]
    report = export(source, [[0, 0, 1], [0, 1, 0], [-1, 0, 0]], target)
    assert report["appearance_preserved"]
    assert describe(target) == before


def test_export_detects_stripped_colors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mesh = trimesh.creation.box()
    mesh.visual.vertex_colors = [20, 80, 210, 255]
    source = tmp_path / "source.glb"
    mesh.export(source)
    real_export = trimesh.Scene.export

    def strip(scene: Any, *args: Any, **kwargs: Any) -> Any:
        for geometry in scene.geometry.values():
            geometry.visual.vertex_colors = [175, 175, 175, 255]
        return real_export(scene, *args, **kwargs)

    monkeypatch.setattr(trimesh.Scene, "export", strip)
    with pytest.raises(ValueError, match="colors, UVs or materials"):
        export(source, np.eye(3), tmp_path / "lost.glb")


def test_strict_oa_rejects_gray_before_loading_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from experiments.orientation import infer
    from tests.canonical_fixtures import render_fixture

    mesh = trimesh.creation.box()
    mesh.visual.vertex_colors = [175, 175, 175, 255]
    path = tmp_path / "gray.glb"
    mesh.export(path)
    views = render_fixture(path, tmp_path / "views", 3)
    monkeypatch.setattr(infer, "verify_source", lambda path: None)
    with pytest.raises(ValueError, match="no color/texture"):
        infer.run(views, tmp_path / "oa", tmp_path, tmp_path, require_appearance=True)
    from experiments.artifacts import read_json

    assert read_json(tmp_path / "oa/predictions.json")["status"] == "failed"

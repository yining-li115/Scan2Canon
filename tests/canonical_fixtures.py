"""Explicit model and rendering fixtures shared by canonicalization worker tests."""

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

np = pytest.importorskip("numpy")
trimesh = pytest.importorskip("trimesh")
from experiments.artifacts import sha256, write_json  # noqa: E402
from experiments.canonicalization.model_policy import PolicyRequest  # noqa: E402
from experiments.orientation.common import config  # noqa: E402


def identity(**changes: Any) -> dict[str, Any]:
    return {
        "status": "resolved",
        "hypotheses": [
            {
                "functional_identity": "kitchen timer",
                "appearance_description": "tomato-shaped enclosure",
                "functional_parts": ["numbered dial"],
                "evidence_ids": ["source:0"],
                "reasoning": "Scripted fixture, not a semantic measurement",
            }
        ],
        "up_description": "The fixture's dial is upright",
        "front_mode": "unique",
        "front_description": "The numbered dial faces forward",
        "missing_evidence": "",
        **changes,
    }


def pose(**changes: Any) -> dict[str, Any]:
    return {
        "scope": "full_frame",
        "candidate_id": 0,
        "evidence_ids": ["candidate:0", "candidate:1"],
        "visible_evidence": "Scripted fixture comparison, not a semantic measurement",
        "mesh_correspondence": "",
        "missing_evidence": "",
        **changes,
    }


class FixturePolicy:
    identity = {"backend": "test_fixture", "model": "scripted-not-inference"}

    def __init__(self, replies: list[Any]) -> None:
        self.replies = iter(replies)
        self.requests: list[PolicyRequest] = []

    def select(self, request: PolicyRequest) -> dict[str, Any]:
        self.requests.append(request)
        reply = next(self.replies)
        if isinstance(reply, Exception):
            raise reply
        return {"raw_text": json.dumps(reply)}


def render_fixture(mesh: Path, output: Path, count: int, poses: list[Any] | None = None) -> Path:
    output.mkdir(parents=True)
    shutil.copyfile(mesh, output / "source.glb")
    views = []
    from experiments.orientation.geometry import GLB_TO_BLENDER

    # Explicit synthetic camera; not a real render or semantic annotation.
    q = 2**-0.5
    canonical_camera = np.array([[q, 0, -q], [0, 1, 0], [q, 0, q]])
    for i in range(count):
        path = output / f"view-{i}.png"
        Image.new("RGB", (16, 16), (i * 10, 50, 0)).save(path)
        views.append(
            {
                "id": i,
                "image": path.name,
                "sha256": sha256(path),
                "camera_to_world": (
                    GLB_TO_BLENDER @ np.asarray(poses[i]).T @ canonical_camera
                    if poses is not None
                    else np.eye(3)
                ).tolist(),
            }
        )
    write_json(
        output / "manifest.json",
        {
            "schema_version": 1,
            "status": "succeeded",
            "mesh": "source.glb",
            "mesh_sha256": sha256(mesh),
            "views": views,
            "mesh_to_blender": GLB_TO_BLENDER.tolist(),
            "camera_horizontal_fov_radians": 0.8,
        },
    )
    return output


@pytest.fixture
def bundles(tmp_path: Path) -> tuple[Path, Path]:
    mesh = trimesh.creation.box(extents=[1, 2, 3])
    mesh.visual.vertex_colors = [220, 70, 40, 255]
    path = tmp_path / "source.glb"
    mesh.export(path)
    views = render_fixture(path, tmp_path / "views", 3)
    predictions = tmp_path / "oa"
    predictions.mkdir()
    np.savez(predictions / "logits.npz", **{f"view_{i:02d}": np.zeros(900) for i in range(3)})
    write_json(predictions / "checkpoint-manifest.json", {"settings": config(), "sha256": "a" * 64})
    write_json(
        predictions / "predictions.json",
        {
            "schema_version": 1,
            "status": "succeeded",
            "settings": config(),
            "views_manifest_sha256": sha256(views / "manifest.json"),
            "logits_sha256": sha256(predictions / "logits.npz"),
            "predictions": [
                {"id": i, "angles_degrees": [i * 90, 0, 0], "symmetry_fit": 1} for i in range(3)
            ],
        },
    )
    return views, predictions


@pytest.fixture
def rendering(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    calls = []

    def render(mesh: Path, output: Path, blender: str, **options: Any) -> Path:
        calls.append(output)
        return render_fixture(mesh, output, len(options["poses"]), options["poses"])

    monkeypatch.setattr("experiments.canonicalization.open_tools.render", render)
    return calls

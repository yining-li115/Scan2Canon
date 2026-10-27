"""Propose horizontal front directions after semantic upright adjudication."""

from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from experiments.artifacts import sha256, write_json
from experiments.orientation.common import verify_views
from experiments.orientation.geometry import rotation
from experiments.orientation.render import render
from experiments.semantic_orientation.contracts import frame


def candidates(mesh: Any, upright: Any) -> list[dict[str, Any]]:
    basis = rotation(upright)
    points, _ = trimesh.sample.sample_surface(mesh, 10000, seed=17)
    horizontal = (points @ basis.T)[:, [0, 2]]
    _, vectors = np.linalg.eigh(np.cov(horizontal.T))
    result: list[dict[str, Any]] = []
    for direction in vectors.T:
        for sign in (1, -1):
            front = basis.T @ [sign * direction[0], 0, sign * direction[1]]
            result.append(
                {
                    "id": len(result),
                    "rotation": frame(basis[1], front).tolist(),
                    "method": "Projected surface PCA horizontal axis",
                }
            )
    return result


def prepare(views: Path, category: str, upright: Any, output: Path, blender: str) -> Path:
    manifest = verify_views(views)
    mesh_path = views / manifest["mesh"]
    mesh = trimesh.load_scene(mesh_path, process=False).to_mesh()
    choices = candidates(mesh, upright)
    output.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "task": "front",
        "category": category,
        "mesh_sha256": manifest["mesh_sha256"],
        "candidates": choices,
        "runner_sha256": sha256(Path(__file__)),
    }
    try:
        render(
            mesh_path,
            output / "poses",
            blender,
            poses=[c["rotation"] for c in choices],
            pose_camera=(0, -1, 0.25),
        )
        report["views_manifest_sha256"] = sha256(output / "poses/manifest.json")
        report["status"] = "succeeded"
    except BaseException:
        report["status"] = "failed"
        raise
    finally:
        write_json(output / "front.json", report)
    return output

"""Generate geometry-derived upright alternatives for semantic adjudication."""

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from experiments.artifacts import read_json, sha256, write_json
from experiments.orientation.common import verify_views
from experiments.orientation.render import render
from experiments.orientation.support import support_candidates
from experiments.semantic_orientation.contracts import direction_error, frame


def candidates(mesh: Any) -> list[dict[str, Any]]:
    points, _ = trimesh.sample.sample_surface(mesh, 10000, seed=17)
    _, axes = np.linalg.eigh(np.cov(points.T))
    axes = axes.T
    # Reuse the support calculation, but rank by geometric probability, not OA votes.
    supports = support_candidates(mesh, [np.eye(3)])
    supports.sort(key=lambda c: -c["quasistatic_probability_proxy"])
    choices: list[tuple[Any, str]] = []
    for axis in axes:
        for sign in (1, -1):
            up = sign * axis
            nearest = min(supports, key=lambda c: direction_error(up, c["up_in_input_frame"]))
            if direction_error(up, nearest["up_in_input_frame"]) < 10:
                up = np.asarray(nearest["up_in_input_frame"])
            choices.append((up, "surface PCA axis, support-snapped within 10 degrees"))
    choices.extend((np.asarray(c["up_in_input_frame"]), "stable support plane") for c in supports)
    result: list[dict[str, Any]] = []
    for up, method in choices:
        if any(direction_error(up, c["rotation"][1]) < 12 for c in result):
            continue
        front = min(axes, key=lambda v: abs(float(v @ up)))
        matrix = frame(up, front)
        result.append({"id": len(result), "rotation": matrix.tolist(), "method": method})
        if len(result) == 12:
            break
    return result


def prepare(views: Path, category: str, output: Path, blender: str) -> Path:
    manifest = verify_views(views)
    mesh_path = views / manifest["mesh"]
    mesh = trimesh.load_scene(mesh_path, process=False).to_mesh()
    choices = candidates(mesh)
    output.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "category": category,
        "mesh_sha256": manifest["mesh_sha256"],
        "candidates": choices,
        "runner_sha256": sha256(Path(__file__)),
    }
    try:
        render(mesh_path, output / "poses", blender, poses=[c["rotation"] for c in choices])
        report["views_manifest_sha256"] = sha256(output / "poses/manifest.json")
        report["status"] = "succeeded"
    except BaseException:
        report["status"] = "failed"
        raise
    finally:
        write_json(output / "upright.json", report)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("renders", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--partition", choices=("development", "holdout"), required=True)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--blender", default="blender")
    args = parser.parse_args()
    if not 1 <= args.limit <= 30:
        parser.error("Limit must be between 1 and 30")
    selected = [
        c
        for c in read_json(args.renders / "index.json")["cases"]
        if c["partition"] == args.partition
    ][: args.limit]
    for case in selected:
        target = args.output / case["case_id"]
        if target.exists():
            raise FileExistsError(target)
        prepare(args.renders / case["views_directory"], case["category"], target, args.blender)
        print(target, flush=True)

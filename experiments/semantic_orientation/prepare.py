"""Prepare shared mesh evidence, surface points, and eight yaw candidates."""

import argparse
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from experiments.artifacts import read_json, sha256, write_json
from experiments.orientation.common import verify_views
from experiments.orientation.geometry import GLB_TO_BLENDER, OA_TO_CANONICAL, mesh_to_oa
from experiments.orientation.render import render
from experiments.orientation.support import support_candidates
from experiments.semantic_orientation.contracts import Rule, frame


def sample_surface(mesh: Any, count: int = 30000, seed: int = 17) -> Any:
    xyz, face_ids = trimesh.sample.sample_surface(mesh, count, seed=seed)
    # to_color resolves texture UVs to vertex colors; barycentric interpolation
    # preserves the sampled surface positions and RGB correspondence.
    visual = mesh.visual.to_color() if mesh.visual.kind == "texture" else mesh.visual
    colors = np.asarray(visual.vertex_colors[:, :3], dtype=float) / 255.0
    barycentric = trimesh.triangles.points_to_barycentric(mesh.triangles[face_ids], xyz)
    rgb = np.einsum("ni,nij->nj", barycentric, colors[mesh.faces[face_ids]])
    return np.column_stack([xyz, np.clip(rgb, 0, 1)]).astype(np.float32)


def prepare(views: Path, predictions: Path, rule_path: Path, output: Path, blender: str) -> Path:
    manifest = verify_views(views)
    evidence = read_json(predictions / "predictions.json")
    rule = Rule.model_validate(read_json(rule_path))
    if (
        evidence["status"] != "succeeded"
        or evidence["views_manifest_sha256"] != sha256(views / "manifest.json")
        or evidence["logits_sha256"] != sha256(predictions / "logits.npz")
    ):
        raise ValueError("OA evidence does not match the case")
    cameras = {v["id"]: v["camera_to_world"] for v in manifest["views"]}
    estimates = evidence["predictions"]
    if len(estimates) != len(cameras) or {p["id"] for p in estimates} != set(cameras):
        raise ValueError("OA predictions must cover every camera exactly once")
    frames = [mesh_to_oa(p["angles_degrees"], cameras[p["id"]]) for p in estimates]
    output.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {"schema_version": 1, "status": "running", "rule": rule.model_dump()}
    write_json(output / "case.json", report)
    try:
        shutil.copyfile(views / manifest["mesh"], output / "input.glb")
        scene = trimesh.load_scene(output / "input.glb", process=False)
        mesh = scene.to_mesh()
        np.save(output / "points.npy", sample_surface(mesh), allow_pickle=False)
        support = support_candidates(mesh, frames)[0]
        transform = np.asarray(support["transform"])
        leveled = scene.copy()
        leveled.apply_transform(transform)
        leveled.export(output / "leveled.glb")
        render(output / "leveled.glb", output / "ring", blender, views=8, preset="ring")
        ring = verify_views(output / "ring")
        candidates = []
        for view in ring["views"]:
            # Camera backward points from the object toward the viewer. Project
            # to the support plane; the renderer's 0.35 elevation is not front tilt.
            backward = GLB_TO_BLENDER.T @ np.asarray(view["camera_to_world"])[:, 2]
            input_front = transform[:3, :3].T @ backward
            matrix = frame(support["up_in_input_frame"], input_front)
            candidates.append({"id": view["id"], "rotation": matrix.tolist()})
        report.update(
            status="succeeded",
            mesh="input.glb",
            mesh_sha256=sha256(output / "input.glb"),
            points="points.npy",
            points_sha256=sha256(output / "points.npy"),
            point_sampling={"count": 30000, "seed": 17, "color": "RGB in [0,1]"},
            ring_manifest_sha256=sha256(output / "ring/manifest.json"),
            oa_predictions_sha256=sha256(predictions / "predictions.json"),
            oa_rotations=[(OA_TO_CANONICAL @ r).tolist() for r in frames],
            support=support,
            candidates=candidates,
            yaw_step_degrees=45,
            evidence_origin="Mesh renders only; generated surfaces are not real observations",
            approval="pending_review",
            runner_sha256=sha256(Path(__file__)),
        )
    except BaseException as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        write_json(output / "case.json", report)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("views", "predictions", "rule", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--blender", default="blender")
    args = parser.parse_args()
    print(prepare(args.views, args.predictions, args.rule, args.output, args.blender))

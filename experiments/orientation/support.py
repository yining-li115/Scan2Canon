"""Generate resting-pose hypotheses and rank their up axes using OA evidence.

Inspired by the support-surface stage in CanoVerse, not a full reproduction.
Convex-hull uniform density is a geometric proxy, not measured mass distribution.
"""

import argparse
from pathlib import Path
from typing import Any

from experiments.artifacts import read_json, sha256, write_json
from experiments.orientation.common import verify_views


def support_candidates(
    mesh: Any, orientations: Any, threshold: float = 25.0
) -> list[dict[str, Any]]:
    import numpy as np
    import trimesh

    from experiments.orientation.geometry import rotation

    if not np.isfinite(threshold) or not 0 < threshold < 90:
        raise ValueError("Up agreement threshold must be between 0 and 90 degrees")
    frames = np.asarray([rotation(r) for r in orientations])
    if not len(frames):
        raise ValueError("At least one orientation is required")
    hull = mesh.convex_hull
    if not hull.is_volume or not np.isfinite(hull.center_mass).all():
        raise ValueError("A finite volumetric convex hull is required")
    transforms, probabilities = trimesh.poses.compute_stable_poses(
        hull, center_mass=hull.center_mass, sigma=0, n_samples=1, seed=0
    )
    candidates = []
    vertices = np.asarray(mesh.vertices)
    center = mesh.bounds.mean(axis=0)
    for index, (pose, probability) in enumerate(zip(transforms, probabilities, strict=True)):
        # trimesh's stable-pose frame is Z-up. OA also stores up in row 2.
        up = rotation(pose[:3, :3])[2]
        errors = np.degrees(np.arccos(np.clip(frames[:, 2] @ up, -1, 1)))
        reference = int(np.argmin(errors))
        front = frames[reference, 0] - np.dot(frames[reference, 0], up) * up
        if np.linalg.norm(front) < 1e-6:
            # Pose yaw is arbitrary; retain an explicit geometric representative.
            front = pose[0, :3]
        front /= np.linalg.norm(front)
        matrix = rotation(np.stack([np.cross(up, front), up, front]))
        transform = np.eye(4)
        transform[:3, :3] = matrix
        transform[:3, 3] = -matrix @ center
        moved = vertices @ matrix.T + transform[:3, 3]
        transform[1, 3] -= moved[:, 1].min()
        candidates.append(
            {
                "stable_pose_index": index,
                "up_in_input_frame": up.tolist(),
                "transform": transform.tolist(),
                "quasistatic_probability_proxy": float(probability),
                "supporting_prediction_indices": np.flatnonzero(errors <= threshold).tolist(),
                "up_errors_degrees": errors.tolist(),
                "median_up_error_degrees": float(np.median(errors)),
                "yaw_reference_prediction_index": reference,
            }
        )
    candidates.sort(
        key=lambda c: (
            -len(c["supporting_prediction_indices"]),
            c["median_up_error_degrees"],
            -c["quasistatic_probability_proxy"],
        )
    )
    return candidates


def run(views: Path, predictions: Path, output: Path, maximum: int = 6) -> Path:
    import numpy as np
    import trimesh

    from experiments.orientation.geometry import mesh_to_oa

    if not 1 <= maximum <= 24:
        raise ValueError("Expected 1–24 exports")
    output.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "decision": "needs_review",
        "dataset_approval": "pending_review",
        "front_established": False,
        "method": "trimesh stable poses of the convex hull; OA up-only ranking",
        "mass_assumption": "Uniform-density convex hull; true mass distribution is unknown",
        "yaw": "Projected OA front is a display representative, not an accepted front",
        "origin": "Input bounding-box center in XZ; lowest vertex rests on Y=0",
        "up_agreement_threshold_degrees": 25.0,
        "deduplication_threshold_degrees": 8.0,
        "runner_sha256": sha256(Path(__file__)),
    }
    write_json(output / "support.json", report)
    try:
        manifest = verify_views(views)
        inference = read_json(predictions / "predictions.json")
        if (
            inference["status"] != "succeeded"
            or inference["views_manifest_sha256"] != sha256(views / "manifest.json")
            or inference["logits_sha256"] != sha256(predictions / "logits.npz")
        ):
            raise ValueError("Invalid or mismatched orientation evidence")
        cameras = {v["id"]: v["camera_to_world"] for v in manifest["views"]}
        estimates = inference["predictions"]
        if len(estimates) != len(cameras) or {p["id"] for p in estimates} != set(cameras):
            raise ValueError("Predictions must cover each view exactly once")
        orientations = [mesh_to_oa(p["angles_degrees"], cameras[p["id"]]) for p in estimates]
        scene = trimesh.load_scene(views / manifest["mesh"], process=False)
        mesh = scene.to_mesh()
        candidates = support_candidates(mesh, orientations)
        exported: list[dict[str, Any]] = []
        for candidate in candidates:
            up = np.asarray(candidate["up_in_input_frame"])
            if any(
                np.degrees(np.arccos(np.clip(up @ np.asarray(c["up_in_input_frame"]), -1, 1))) <= 8
                for c in exported
            ):
                continue
            moved = scene.copy()
            moved.apply_transform(candidate["transform"])
            path = output / f"support-{len(exported) + 1:02d}.glb"
            moved.export(path)
            exported.append(
                {
                    **candidate,
                    "mesh": path.name,
                    "sha256": sha256(path),
                    "supporting_views": [
                        estimates[i]["id"] for i in candidate["supporting_prediction_indices"]
                    ],
                }
            )
            if len(exported) == maximum:
                break
        if not exported:
            raise ValueError("No stable-pose candidates found")
        report.update(
            status="succeeded",
            input_mesh_sha256=manifest["mesh_sha256"],
            predictions_sha256=sha256(predictions / "predictions.json"),
            support_plane_count=len(candidates),
            all_support_planes=candidates,
            exports=exported,
        )
    except BaseException as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        write_json(output / "support.json", report)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("views", type=Path)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--maximum", type=int, default=6)
    args = parser.parse_args()
    print(run(args.views, args.predictions, args.output, args.maximum))

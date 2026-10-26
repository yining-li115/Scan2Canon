"""Rank orientation hypotheses and export reviewable, shape-preserving GLB candidates."""

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from experiments.artifacts import read_json, sha256, write_json
from experiments.orientation.common import verify_views


def rank_symmetry_group(
    cameras: dict[int, Any], predictions: list[dict[str, Any]], order: int, threshold: float
) -> list[dict[str, Any]]:
    """Rank poses within one predicted symmetry convention, without mixing votes."""
    import numpy as np

    from experiments.orientation.geometry import mesh_to_oa, symmetry_error

    selected = sorted((p for p in predictions if p["symmetry_fit"] == order), key=lambda p: p["id"])
    rotations = [mesh_to_oa(p["angles_degrees"], cameras[p["id"]]) for p in selected]
    distances = np.array([[symmetry_error(a, b, order) for b in rotations] for a in rotations])
    support = (distances <= threshold).sum(axis=1)
    ranking = sorted(
        range(len(rotations)),
        key=lambda i: (-int(support[i]), float(distances[i].sum()), selected[i]["id"]),
    )
    candidates = []
    representatives: list[int] = []
    for index in ranking:
        if any(distances[index, previous] <= threshold for previous in representatives):
            continue
        representatives.append(index)
        candidates.append(
            {
                "symmetry_fit": order,
                "equivalent_yaw_offsets_degrees": (
                    [360 * k / order for k in range(order)] if order else None
                ),
                "source_view": selected[index]["id"],
                "mesh_to_oa": rotations[index].tolist(),
                "supporting_views": [
                    selected[j]["id"]
                    for j in range(len(selected))
                    if distances[index, j] <= threshold
                ],
                "support_fraction_all_views": int(support[index]) / len(predictions),
                "median_error_degrees": float(np.median(distances[index])),
            }
        )
        if len(candidates) == 3:
            break
    return candidates


def hypotheses(
    manifest: dict[str, Any], predictions: list[dict[str, Any]], threshold: float = 25.0
) -> dict[str, Any]:
    import numpy as np

    if not np.isfinite(threshold) or not 0 < threshold < 90:
        raise ValueError("Agreement threshold must be between 0 and 90 degrees")
    cameras = {v["id"]: v["camera_to_world"] for v in manifest["views"]}
    if (
        not cameras
        or len(predictions) != len(cameras)
        or {p["id"] for p in predictions} != set(cameras)
    ):
        raise ValueError("Predictions must cover each view exactly once")
    if any(p["symmetry_fit"] not in (0, 1, 2, 4) for p in predictions):
        raise ValueError("Unsupported symmetry fit")
    counts = Counter(p["symmetry_fit"] for p in predictions)
    most_votes = max(counts.values())
    modes = sorted(order for order, votes in counts.items() if votes == most_votes)
    candidates = [
        candidate
        for mode in modes
        for candidate in rank_symmetry_group(cameras, predictions, mode, threshold)
    ]
    order = modes[0] if len(modes) == 1 else None
    reasons = []
    if 0 in modes:
        reasons.append("azimuth_unresolved: front direction is not established")
    if len(modes) > 1:
        reasons.append("symmetry_fit_tie: all tied modes retained")
    if most_votes / len(predictions) < 0.75:
        reasons.append("symmetry_fits_disagree")
    if max(c["support_fraction_all_views"] for c in candidates) < 0.6:
        reasons.append("view_orientations_disagree")
    if any(mode in (2, 4) for mode in modes):
        reasons.append("multiple_equivalent_fronts_predicted: requires symmetry review")
    return {
        "symmetry_fit": order,
        "symmetry_fit_modes": modes,
        "equivalent_yaw_offsets_degrees": (
            [360 * k / order for k in range(order)] if order else None
        ),
        "front_observable": None if order is None else order != 0,
        "symmetry_fit_counts": dict(sorted(counts.items())),
        "agreement_threshold_degrees": threshold,
        "decision": "needs_evidence" if reasons else "candidate_ready_for_review",
        "reasons": reasons,
        "candidates": candidates,
        "candidate_order": "Grouped by symmetry fit; ranked only within each group",
        "score_note": "View agreement is a heuristic, not a calibrated correctness probability",
    }


def solve(views: Path, predictions: Path, output: Path, threshold: float = 25.0) -> Path:
    import numpy as np
    import trimesh

    from experiments.orientation.geometry import centered_transform

    views, predictions, output = (p.resolve() for p in (views, predictions, output))
    output.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "dataset_approval": "pending_review",
    }
    write_json(output / "canonicalization.json", report)
    try:
        manifest = verify_views(views)
        inference = read_json(predictions / "predictions.json")
        if inference.get("status") != "succeeded" or inference.get("schema_version") != 1:
            raise ValueError("Inference is incomplete or unsupported")
        if inference["views_manifest_sha256"] != sha256(views / "manifest.json"):
            raise ValueError("Predictions belong to a different render bundle")
        if sha256(predictions / "logits.npz") != inference["logits_sha256"]:
            raise ValueError("Logits checksum mismatch")
        report.update(hypotheses(manifest, inference["predictions"], threshold))
        report.update(
            {
                "input_mesh_sha256": manifest["mesh_sha256"],
                "predictions_sha256": sha256(predictions / "predictions.json"),
                "solver_sha256": sha256(Path(__file__)),
                "views_manifest_sha256": sha256(views / "manifest.json"),
                "coordinate_convention": "scan2canon-y-up-z-front-v1; right-handed column vectors",
                "transform_equation": "p_canonical = R @ p_glb + t",
                "origin": "Input world-space bounding-box center",
                "scale": "unchanged; physical dimensions remain unverified",
                "semantic_convention": "OA learned front/up; category rules not yet verified",
                "held_out_real_view_validation": "not_performed",
                "geometry": "Only rigid transforms; no regeneration or topology edits",
            }
        )
        scene = trimesh.load_scene(views / manifest["mesh"], process=False)
        if not scene.geometry or not np.isfinite(scene.bounds).all():
            raise ValueError("Empty or non-finite input geometry")
        center = scene.bounds.mean(axis=0)
        if not np.allclose(center, manifest["center_glb"], atol=max(scene.extents) * 1e-4):
            raise ValueError("Blender and GLB centers disagree; inspect coordinate conversion")
        for index, candidate in enumerate(report["candidates"], 1):
            transform = centered_transform(np.asarray(candidate["mesh_to_oa"]), center)
            transformed = scene.copy()
            transformed.apply_transform(transform)
            filename = f"candidate-{index:02d}.glb"
            transformed.export(str(output / filename))
            candidate.update(
                {
                    "transform": transform.tolist(),
                    "mesh": filename,
                    "sha256": sha256(output / filename),
                }
            )
        report["status"] = "succeeded"
    except BaseException as error:
        report.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        write_json(output / "canonicalization.json", report)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("views", type=Path)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--agreement-degrees", type=float, default=25.0)
    args = parser.parse_args()
    print(solve(args.views, args.predictions, args.output, args.agreement_degrees))


if __name__ == "__main__":
    main()

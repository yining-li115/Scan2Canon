"""Orientation consensus and support geometry; no scoring references or model SDKs."""

from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from experiments.orientation.batch import evidence
from experiments.orientation.common import verify_views
from experiments.orientation.geometry import OA_TO_CANONICAL, angular_error, mesh_to_oa
from experiments.orientation.support import support_candidates
from experiments.semantic_orientation.contracts import direction_error, frame, unit


def direction_cluster(vectors: Any, threshold: float) -> dict[str, Any]:
    values = np.asarray([unit(v) for v in vectors])
    if not len(values):
        return {"direction": None, "indices": [], "count": 0, "runner_up_count": 0}
    distances = np.degrees(np.arccos(np.clip(values @ values.T, -1, 1)))
    votes = distances <= threshold
    winner = min(range(len(values)), key=lambda i: (-int(votes[i].sum()), distances[i].sum()))
    members = np.flatnonzero(votes[winner])
    mean = unit(values[members].mean(axis=0))
    # Count a competing direction only if its center is outside this cluster.
    other = [int(votes[i].sum()) for i in range(len(values)) if not votes[winner, i]]
    return {
        "direction": mean.tolist(),
        "indices": members.tolist(),
        "count": len(members),
        "runner_up_count": max(other, default=0),
        "spread_degrees": max(direction_error(v, mean) for v in values[members]),
    }


def collect(views: Path, predictions: Path, threshold: float = 25.0) -> dict[str, Any]:
    manifest = verify_views(views)
    inference = evidence(views, predictions)
    cameras = {v["id"]: v["camera_to_world"] for v in manifest["views"]}
    estimates = inference["predictions"]
    if any(p["symmetry_fit"] not in (0, 1, 2, 4) for p in estimates):
        raise ValueError("Unsupported OA symmetry fit")
    oa = [mesh_to_oa(p["angles_degrees"], cameras[p["id"]]) for p in estimates]
    frames = np.asarray([OA_TO_CANONICAL @ r for r in oa])
    medoid = min(range(len(frames)), key=lambda i: sum(angular_error(frames[i], r) for r in frames))
    raw = frames[medoid]
    mesh = trimesh.load_scene(views / manifest["mesh"], process=False).to_mesh()
    supports = support_candidates(mesh, oa, threshold)
    distinct: list[dict[str, Any]] = []
    for candidate in supports:
        if all(
            direction_error(candidate["up_in_input_frame"], c["up_in_input_frame"]) > 15
            for c in distinct
        ):
            distinct.append(candidate)
    if not distinct:
        raise ValueError("No stable support candidates")
    best = distinct[0]
    up = np.asarray(best["up_in_input_frame"])
    up_indices = best["supporting_prediction_indices"]
    eligible = [
        i
        for i in up_indices
        if estimates[i]["symmetry_fit"] == 1 and abs(float(frames[i, 2] @ up)) < 0.5
    ]
    projected = [unit(frames[i, 2] - float(frames[i, 2] @ up) * up) for i in eligible]
    front = direction_cluster(projected, threshold)
    front["prediction_indices"] = [eligible[i] for i in front.pop("indices")]
    assisted = (
        frame(up, front["direction"])
        if front["direction"] is not None
        else np.asarray(best["transform"])[:3, :3]
    )
    return {
        "view_count": len(estimates),
        "raw_rotation": raw.tolist(),
        "support_rotation": assisted.tolist(),
        "up_cluster": direction_cluster(frames[:, 1], threshold),
        "support_count": len(supports),
        "support_candidates": distinct[:6],
        "up_votes": len(up_indices),
        "other_up_votes": max(
            (len(c["supporting_prediction_indices"]) for c in distinct[1:]), default=0
        ),
        "front_cluster": front,
        "symmetry_fits": [p["symmetry_fit"] for p in estimates],
        "mass_assumption": "Uniform-density convex hull proxy; physical center of mass unknown",
        "evidence_scope": "Correlated renders of one mesh; no new real-world observations",
    }

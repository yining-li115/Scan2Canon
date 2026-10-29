"""Bounded support-plane refinement of existing pose proposals, without changing geometry."""

from typing import Any

import numpy as np
from scipy.spatial import ConvexHull, QhullError  # type: ignore[import-untyped]

from experiments.orientation.geometry import OA_TO_CANONICAL, angular_error, rotation
from experiments.orientation.support import support_candidates

MAX_CORRECTION_DEGREES = 25.0
MIN_CONTACT_COVERAGE = 0.10
CONTACT_BAND_FRACTION = 0.003


def contact_metrics(mesh: Any, up: Any) -> dict[str, float]:
    """Measure a support polygon near the lowest surface; not measured physical stability."""
    normal = np.asarray(up, dtype=float)
    if normal.shape != (3,) or not np.isfinite(normal).all():
        raise ValueError("Expected a finite support normal")
    if not np.isclose(np.linalg.norm(normal), 1, atol=1e-5):
        raise ValueError("Support normal must have unit length")
    vertices = np.asarray(mesh.vertices)
    diagonal = float(np.linalg.norm(mesh.extents))
    if diagonal <= 1e-8 or not np.isfinite(vertices).all():
        raise ValueError("Invalid mesh extent or vertices")
    height = vertices @ normal
    band = diagonal * CONTACT_BAND_FRACTION
    contact = height <= height.min() + band
    tangent = np.eye(3)[int(np.argmin(np.abs(normal)))]
    tangent -= (tangent @ normal) * normal
    tangent /= np.linalg.norm(tangent)
    projected = vertices @ np.stack([tangent, np.cross(normal, tangent)]).T
    try:
        footprint = float(ConvexHull(projected).volume)
        area = float(ConvexHull(projected[contact]).volume) if contact.sum() >= 3 else 0.0
    except QhullError:
        footprint, area = 0.0, 0.0
    return {
        "contact_coverage": area / footprint if footprint > 0 else 0.0,
        "contact_polygon_area": area,
        "footprint_area": footprint,
        "contact_band_mesh_units": band,
    }


def refine_up(mesh: Any, candidates: dict[str, Any], ids: list[int]) -> dict[str, Any]:
    """Create at most three alternatives; preserve parents and their unresolved yaw status."""
    rows = {c["id"]: c for c in candidates["candidates"]}
    if len(ids) != 3 or len(set(ids)) != 3 or not set(ids) <= rows.keys():
        raise ValueError("Choose three distinct existing candidates")
    if any("refinement" in rows[i] for i in ids):
        raise ValueError("Refinement cannot be chained onto a refined candidate")
    return _propose(mesh, candidates, ids)


def assess_up(mesh: Any, candidates: dict[str, Any]) -> dict[str, Any]:
    """Flag inspectable support corrections before accepting a final pose choice."""
    ids = [c["id"] for c in candidates["candidates"]]
    if not 1 <= len(ids) <= 12:
        raise ValueError("Support assessment requires one to twelve original candidates")
    try:
        result = _propose(mesh, candidates, ids)
    except (ValueError, QhullError) as error:
        return {"status": "unavailable", "error_type": type(error).__name__, "assessments": []}
    assessments = []
    for attempt in result["attempts"]:
        assessments.append(
            {
                "candidate_id": attempt["parent_candidate_id"],
                "up_refinement_review_required": (
                    attempt["status"] == "proposed" and attempt["up_correction_degrees"] > 3.0
                ),
                "diagnostic": {k: v for k, v in attempt.items() if k != "candidate_id"},
            }
        )
    return {
        "status": "assessed",
        "assessments": assessments,
        "scope": "Requires visual comparison, not mandatory acceptance of geometric correction",
    }


def _propose(mesh: Any, candidates: dict[str, Any], ids: list[int]) -> dict[str, Any]:
    rows = {c["id"]: c for c in candidates["candidates"]}
    supports = support_candidates(
        mesh, [OA_TO_CANONICAL.T @ rotation(rows[i]["rotation"]) for i in ids]
    )
    # Geometry is evaluated once per support plane, independently of model confidence.
    planes = [{**s, **contact_metrics(mesh, s["up_in_input_frame"])} for s in supports]
    alternatives: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    for identifier in ids:
        parent = rows[identifier]
        original = rotation(parent["rotation"])
        eligible = []
        for plane in planes:
            up = np.asarray(plane["up_in_input_frame"])
            angle = float(np.degrees(np.arccos(np.clip(original[1] @ up, -1, 1))))
            if (
                angle <= MAX_CORRECTION_DEGREES
                and plane["contact_coverage"] >= MIN_CONTACT_COVERAGE
            ):
                eligible.append((angle, plane))
        eligible.sort(key=lambda p: (p[0], -p[1]["contact_coverage"]))
        attempt: dict[str, Any] = {"parent_candidate_id": identifier}
        attempts.append(attempt)
        if not eligible:
            attempt.update(status="rejected", reason="no_supported_plane_within_correction_limit")
            continue
        angle, best = eligible[0]
        up = np.asarray(best["up_in_input_frame"])
        if any(
            other_angle - angle < 3
            and np.degrees(np.arccos(np.clip(up @ plane["up_in_input_frame"], -1, 1))) > 5
            for other_angle, plane in eligible[1:]
        ):
            attempt.update(status="rejected", reason="competing_support_planes")
            continue
        front = original[2] - (original[2] @ up) * up
        if np.linalg.norm(front) < 1e-6:
            attempt.update(status="rejected", reason="front_projection_degenerate")
            continue
        front /= np.linalg.norm(front)
        refined = rotation(np.stack([np.cross(up, front), up, front]))
        update = angular_error(refined, original)
        if update > MAX_CORRECTION_DEGREES:
            attempt.update(status="rejected", reason="rotation_update_exceeds_limit")
            continue
        record = {
            "method": "convex_hull_support_plane_v1",
            "parent_candidate_id": identifier,
            "support_up_in_mesh": up.tolist(),
            "stable_pose_index": best["stable_pose_index"],
            "up_correction_degrees": angle,
            "rotation_update_degrees": update,
            "contact_coverage_before": contact_metrics(mesh, original[1])["contact_coverage"],
            "contact_coverage_after": best["contact_coverage"],
            "quasistatic_probability_proxy": best["quasistatic_probability_proxy"],
            "yaw": "Parent front projected onto new horizontal plane; no new front evidence",
        }
        new_id = max(rows) + 1 + len(alternatives)
        alternatives.append(
            {**parent, "id": new_id, "rotation": refined.tolist(), "refinement": record}
        )
        attempt.update(status="proposed", candidate_id=new_id, **record)
    return {
        "status": "proposed" if alternatives else "no_supported_refinement",
        "max_correction_degrees": MAX_CORRECTION_DEGREES,
        "min_contact_coverage": MIN_CONTACT_COVERAGE,
        "contact_band_fraction": CONTACT_BAND_FRACTION,
        "mass_assumption": "Uniform-density convex hull; not the real mass distribution",
        "scope": "Geometric alternatives only; agent must inspect before selecting",
        "attempts": attempts,
        "new_candidates": alternatives,
    }

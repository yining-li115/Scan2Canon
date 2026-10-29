"""CPU-only pose search over OA marginals; scores are not correctness probabilities."""

from typing import Any

import numpy as np

from experiments.orientation.geometry import (
    GLB_TO_BLENDER,
    OA_CAMERA_TO_WORLD,
    Matrix,
    angular_error,
    axis_rotation,
    mesh_to_oa,
    rotation,
)

SETTINGS: dict[str, Any] = {
    "version": "oa-marginal-fusion-v1",
    "random_seed": 20261029,
    "global_samples": 4096,
    "refinement_seeds": 18,
    "seed_separation_degrees": 25.0,
    "candidate_separation_degrees": 15.0,
    "refinement_bound_radians": 0.35,
    "maximum_candidates": 6,
}


def object_angles(matrices: Matrix) -> Matrix:
    """Invert Rx(roll) Ry(elevation) Rz(-azimuth), in degrees.

    At exact gimbal lock choose azimuth zero; scoring marginalizes the coupled
    azimuth/roll family rather than treating that arbitrary choice as evidence.
    """
    elevation = np.arcsin(np.clip(matrices[..., 0, 2], -1, 1))
    azimuth = np.arctan2(matrices[..., 0, 1], matrices[..., 0, 0])
    roll = np.arctan2(-matrices[..., 1, 2], matrices[..., 2, 2])
    polar = np.abs(np.cos(elevation)) < 1e-6
    azimuth = np.where(polar, 0, azimuth)
    roll = np.where(polar, np.arctan2(matrices[..., 2, 1], matrices[..., 1, 1]), roll)
    return np.degrees(np.stack((azimuth, elevation, roll), axis=-1))


class MarginalEvidence:
    """Equal-view, equal-axis mean log score relative to uniform bin scores."""

    def __init__(self, cameras: Matrix, logits: Matrix) -> None:
        if cameras.ndim != 3 or not 3 <= len(cameras) <= 64:
            raise ValueError("Expected 3–64 registered cameras")
        self.cameras = np.stack([rotation(c) for c in cameras])
        if logits.shape != (len(cameras), 900) or not np.isfinite(logits).all():
            raise ValueError("Expected finite 900-bin logits for every registered view")
        self.bases = OA_CAMERA_TO_WORLD @ self.cameras.transpose(0, 2, 1) @ GLB_TO_BLENDER
        self.tables: list[Matrix] = []
        for segment in (logits[:, :360], logits[:, 360:540], logits[:, 540:]):
            values = -np.logaddexp(0, -segment)
            values -= np.logaddexp.reduce(values, axis=1)[:, None]
            self.tables.append(values + np.log(segment.shape[1]))
        self.informative = any(float(np.ptp(t, axis=1).max()) > 1e-5 for t in self.tables)

    def _sample(self, axis: int, degrees: Matrix) -> Matrix:
        table = self.tables[axis]
        bins = degrees + (0, 90, 180)[axis]
        count = table.shape[1]
        bins = np.clip(bins, 0, count - 1) if axis == 1 else bins % count
        bins = np.where(bins >= count, 0, bins)
        low = np.floor(bins).astype(int)
        high = np.minimum(low + 1, count - 1) if axis == 1 else (low + 1) % count
        fraction = bins - low
        views = np.arange(len(table)).reshape((-1,) + (1,) * (bins.ndim - 1))
        return np.asarray(
            (1 - fraction) * table[views, low] + fraction * table[views, high], dtype=float
        )

    def per_view(self, poses: Matrix) -> Matrix:
        """Return (pose, view) scores; poses map input GLB coordinates to OA axes."""
        projected = self.bases[:, None] @ poses.transpose(0, 2, 1)[None]
        angles = object_angles(projected)
        azimuth = self._sample(0, angles[..., 0])
        elevation = self._sample(1, angles[..., 1])
        roll = self._sample(2, angles[..., 2])
        # Exact Euler singularities describe a family of equivalent coordinates.
        polar = np.abs(np.cos(np.radians(angles[..., 1]))) < 1e-6
        if polar.any():
            az = np.broadcast_to(np.arange(0.0, 360.0, 5), (*polar.shape, 72))
            ro = angles[..., 2, None] + np.sign(angles[..., 1, None]) * az
            coupled = self._sample(0, az) + self._sample(2, ro)
            marginal = np.logaddexp.reduce(coupled, axis=-1) - np.log(72)
            azimuth = np.where(polar, marginal, azimuth)
            roll = np.where(polar, 0, roll)
        return ((azimuth + elevation + roll) / 3).T

    def scores(self, poses: Matrix) -> Matrix:
        return np.concatenate(
            [self.per_view(poses[i : i + 256]).mean(axis=1) for i in range(0, len(poses), 256)]
        )


def fuse(cameras: Matrix, logits: Matrix) -> dict[str, Any]:
    """Bounded deterministic search, preserving diverse alternatives for review."""
    from scipy.optimize import minimize  # type: ignore[import-untyped]
    from scipy.spatial.transform import Rotation  # type: ignore[import-untyped]

    evidence = MarginalEvidence(cameras, logits)
    report: dict[str, Any] = {
        "settings": SETTINGS.copy(),
        "decision": "needs_evidence",
        "semantic_correctness_verified": False,
        "symmetry_verified": False,
        "score_note": "Mean log normalized sigmoid score relative to uniform; correlated "
        "views and factorized Euler marginals are not a calibrated joint posterior",
        "candidates": [],
    }
    if not evidence.informative:
        return {**report, "reason": "all_marginals_flat", "evaluated_poses": 0}
    angles = np.column_stack(
        (
            logits[:, :360].argmax(1),
            logits[:, 360:540].argmax(1) - 90,
            logits[:, 540:].argmax(1) - 180,
        )
    )
    bases = [mesh_to_oa(a.tolist(), c) for a, c in zip(angles, cameras, strict=True)]
    # Use the first camera frame rather than a preferred input-world upright axis.
    random = Rotation.random(SETTINGS["global_samples"], random_state=SETTINGS["random_seed"])
    pool = np.concatenate(
        (
            random.as_matrix() @ evidence.bases[0],
            np.array([axis_rotation(2, yaw) @ b for b in bases for yaw in range(0, 360, 30)]),
        )
    )
    scores = evidence.scores(pool)
    seed_ids: list[int] = []
    for i in np.argsort(-scores, kind="stable"):
        if all(
            angular_error(pool[i], pool[j]) >= SETTINGS["seed_separation_degrees"] for j in seed_ids
        ):
            seed_ids.append(int(i))
        if len(seed_ids) == SETTINGS["refinement_seeds"]:
            break
    refined, diagnostics = [], []
    for i in seed_ids:
        base = pool[i]

        def objective(delta: Matrix, base: Matrix = base) -> float:
            pose = Rotation.from_rotvec(delta).as_matrix() @ base
            return -float(evidence.scores(pose[None])[0])

        bound = SETTINGS["refinement_bound_radians"]
        fit = minimize(
            objective,
            np.zeros(3),
            method="Powell",
            bounds=[(-bound, bound)] * 3,
            options={"maxiter": 50, "maxfev": 1200, "xtol": 1e-5, "ftol": 1e-6},
        )
        proposed = rotation(Rotation.from_rotvec(fit.x).as_matrix() @ base)
        # Failed/poorer optimization cannot replace a better saved seed.
        improved = -float(fit.fun) > scores[i]
        refined.append(proposed if improved else base)
        diagnostics.append(
            {
                "optimizer_success": bool(fit.success),
                "evaluations": int(fit.nfev),
                "used_improvement": bool(improved),
            }
        )
    refined_array = np.asarray(refined)
    final_scores = evidence.scores(refined_array)
    selected: list[int] = []
    for i in np.argsort(-final_scores, kind="stable"):
        if all(
            angular_error(refined_array[i], refined_array[j])
            >= SETTINGS["candidate_separation_degrees"]
            for j in selected
        ):
            selected.append(int(i))
        if len(selected) == SETTINGS["maximum_candidates"]:
            break
    report.update(
        reason="distribution_proposals_require_independent_semantic_evidence",
        evaluated_poses=len(pool) + sum(d["evaluations"] for d in diagnostics),
        candidates=[
            {
                "id": rank,
                "mesh_to_oa": refined_array[i].tolist(),
                "score": float(final_scores[i]),
                "gap_from_best": float(final_scores[selected[0]] - final_scores[i]),
                "per_view_scores": evidence.per_view(refined_array[i : i + 1])[0].tolist(),
                "optimization": diagnostics[i],
            }
            for rank, i in enumerate(selected)
        ],
        search_note="Finite multistart search; retained modes are not exhaustive; score gaps "
        "are not calibrated ambiguity thresholds",
    )
    return report

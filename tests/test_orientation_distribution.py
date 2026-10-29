"""Known synthetic rotations exercise math, ambiguity and recovery, not model accuracy."""

import pytest

np = pytest.importorskip("numpy")

from experiments.orientation.distribution import MarginalEvidence, fuse, object_angles  # noqa: E402
from experiments.orientation.geometry import (  # noqa: E402
    GLB_TO_BLENDER,
    OA_CAMERA_TO_WORLD,
    Matrix,
    angular_error,
    axis_rotation,
    oa_object_rotation,
)


def synthetic(two_fronts: bool = False) -> tuple[Matrix, Matrix, Matrix]:
    target = axis_rotation(0, 23) @ axis_rotation(1, -37)
    angles = [[a, e, r] for a, e, r in ((5, 10, -8), (120, -20, 12), (244, 30, 40))]
    cameras = np.array(
        [GLB_TO_BLENDER @ target.T @ oa_object_rotation(*a).T @ OA_CAMERA_TO_WORLD for a in angles]
    )
    rows = []
    for az, el, ro in angles:
        segments = []
        for center, count, offset in ((az, 360, 0), (el, 180, 90), (ro, 360, 180)):
            d = abs(np.arange(count) - (center + offset))
            if count == 360:
                d = np.minimum(d, count - d)
            if two_fronts and offset == 0:
                d = np.minimum(d, abs(180 - d))
            probability = 0.01 + 0.8 * np.exp(-0.5 * (d / 7) ** 2)
            segments.append(np.log(probability / (1 - probability)))
        rows.append(np.concatenate(segments))
    return target, cameras, np.array(rows)


@pytest.mark.parametrize("elevation", [-90, -89, -40, 0, 50, 89, 90])
def test_inverse_euler_round_trip_including_poles(elevation: float) -> None:
    matrix = oa_object_rotation(173, elevation, -140)
    recovered = object_angles(matrix)
    np.testing.assert_allclose(oa_object_rotation(*recovered), matrix, atol=1e-7)


def test_fusion_recovers_known_pose_and_reports_review_only() -> None:
    expected, cameras, logits = synthetic()
    result = fuse(cameras, logits)
    assert angular_error(np.array(result["candidates"][0]["mesh_to_oa"]), expected) < 1
    assert result["decision"] == "needs_evidence"
    assert result["semantic_correctness_verified"] is False
    # A changed world basis cannot change the evidence score of the same pose.
    q = axis_rotation(0, 48) @ axis_rotation(2, -31)
    moved = GLB_TO_BLENDER @ q @ GLB_TO_BLENDER.T @ cameras
    np.testing.assert_allclose(
        MarginalEvidence(cameras, logits).scores(expected[None]),
        MarginalEvidence(moved, logits).scores((expected @ q.T)[None]),
        atol=1e-10,
    )


def test_competing_fronts_survive_instead_of_becoming_a_symmetry_claim() -> None:
    expected, cameras, logits = synthetic(two_fronts=True)
    result = fuse(cameras, logits)
    poses = [np.array(c["mesh_to_oa"]) for c in result["candidates"]]
    for target in (expected, axis_rotation(2, 180) @ expected):
        assert min(angular_error(p, target) for p in poses) < 1
    assert result["symmetry_verified"] is False
    assert result["candidates"][1]["gap_from_best"] < 0.01


def test_flat_evidence_abstains_and_corrupt_logits_fail() -> None:
    _, cameras, logits = synthetic()
    result = fuse(cameras, np.zeros_like(logits))
    assert result["candidates"] == []
    assert result["reason"] == "all_marginals_flat"
    logits[0, 42] = np.nan
    with pytest.raises(ValueError, match="finite"):
        fuse(cameras, logits)


def test_polar_scores_do_not_depend_on_arbitrary_euler_branch() -> None:
    from experiments.orientation.geometry import mesh_to_oa

    _, cameras, logits = synthetic()
    evidence = MarginalEvidence(cameras, logits)
    a = mesh_to_oa([0, 90, 25], cameras[0])
    b = mesh_to_oa([120, 90, 145], cameras[0])
    np.testing.assert_allclose(a, b, atol=1e-10)
    np.testing.assert_allclose(evidence.scores(np.array([a])), evidence.scores(np.array([b])))

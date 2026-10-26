import pytest
from pydantic import ValidationError

from scan2canon.domain import CanonicalPose, Matrix


@pytest.mark.parametrize(
    "rotation",
    [
        ((-1, 0, 0), (0, 1, 0), (0, 0, 1)),
        ((2, 0, 0), (0, 1, 0), (0, 0, 1)),
        ((float("nan"), 0, 0), (0, 1, 0), (0, 0, 1)),
        ((1, 0, 0), (1, 0, 0), (0, 0, 1)),
    ],
)
def test_invalid_rotations_are_rejected(rotation: Matrix) -> None:
    with pytest.raises(ValidationError):
        CanonicalPose(
            rotation=rotation,
            convention_id="test-v1",
            category="cup",
            symmetry="none",
        )


def test_rigid_pose_preserves_symmetry_information() -> None:
    pose = CanonicalPose(
        rotation=((0, 0, 1), (0, 1, 0), (-1, 0, 0)),
        convention_id="y-up-v1",
        category="bottle",
        symmetry="continuous_about_up",
    )
    assert CanonicalPose.model_validate_json(pose.model_dump_json()) == pose
    assert not pose.scale_known

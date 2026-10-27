from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
trimesh = pytest.importorskip("trimesh")

from experiments.canonicalization.evidence import direction_cluster  # noqa: E402
from experiments.canonicalization.semantic import run  # noqa: E402
from experiments.canonicalization.upright import candidates  # noqa: E402
from experiments.orientation.geometry import axis_rotation, rotation  # noqa: E402
from experiments.semantic_orientation.contracts import direction_error  # noqa: E402


def test_direction_consensus_retains_competing_axis() -> None:
    evidence = direction_cluster([[0, 1, 0], [0, 1, 0.1], [0, -1, 0]], 20)
    assert evidence["count"] == 2
    assert evidence["runner_up_count"] == 1
    assert direction_error(evidence["direction"], [0, 1, 0]) < 4
    assert direction_cluster([], 20)["direction"] is None
    with pytest.raises(ValueError, match="zero length"):
        direction_cluster([[0, 0, 0]], 20)


def test_upright_alternatives_cover_rotated_supports_without_changing_mesh() -> None:
    mesh = trimesh.creation.box(extents=[1, 2, 3])
    applied = axis_rotation(0, 38) @ axis_rotation(2, 71)
    mesh.vertices = mesh.vertices @ applied.T
    before = np.array(mesh.vertices)
    choices = candidates(mesh)
    assert 6 <= len(choices) <= 12
    for up in np.vstack([applied.T, -applied.T]):
        assert min(direction_error(up, c["rotation"][1]) for c in choices) < 1e-5
    for choice in choices:
        rotation(choice["rotation"])
    np.testing.assert_array_equal(before, mesh.vertices)


def test_semantic_budget_fails_before_model_load_or_output_creation(tmp_path: Path) -> None:
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="budget"):
        run([tmp_path] * 2, out, tmp_path, budget=2)
    assert not out.exists()
    with pytest.raises(ValueError, match="Repeated"):
        run([tmp_path] * 2, out, tmp_path, budget=4)

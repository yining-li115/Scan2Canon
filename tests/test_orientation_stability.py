"""Resampling metrics and recovery use synthetic evidence, not model accuracy labels."""

from pathlib import Path
from typing import Any

import pytest

np = pytest.importorskip("numpy")
from experiments.artifacts import read_json, sha256, write_json  # noqa: E402
from experiments.orientation import stability  # noqa: E402
from experiments.orientation.geometry import axis_rotation  # noqa: E402
from tests.canonical_fixtures import bundles as bundles  # noqa: E402
from tests.canonical_fixtures import render_fixture  # noqa: E402


def test_splits_are_disjoint_complete_and_deterministic() -> None:
    plan = stability.subsets(12)
    assert plan == stability.subsets(12)
    assert len(plan) == 25
    seen = set()
    for i in range(6):
        a, b = set(plan[f"split-{i}-a"]), set(plan[f"split-{i}-b"])
        assert len(a) == len(b) == 6 and not a & b
        assert a | b == set(range(12))
        seen.add(tuple(sorted(a)))
    assert len(seen) == 6
    with pytest.raises(ValueError):
        stability.subsets(7)


def test_stability_distinguishes_front_flip_from_up_and_keeps_missing_fits() -> None:
    identity = {"candidates": [{"mesh_to_oa": np.eye(3).tolist()}]}
    fits = {k: identity for k in stability.subsets(12)}
    fits["split-0-b"] = {"candidates": [{"mesh_to_oa": axis_rotation(2, 180).tolist()}]}
    fits["omit-0"] = {"candidates": []}
    result = stability.summarize(fits)
    first = result["disjoint_halves"][0]["difference"]
    assert first == {"frame_degrees": 180, "front_degrees": 180, "up_degrees": 0}
    assert result["missing_top1_subsets"] == ["omit-0"]
    assert not result["semantic_accuracy_measured"]
    assert not result["acceptance_threshold_calibrated"]


def test_interrupted_fit_resumes_verified_stages_and_rejects_corruption(
    bundles: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, predictions = bundles
    views = render_fixture(source / "source.glb", tmp_path / "eight-views", 8)
    np.savez(predictions / "logits.npz", **{f"view_{i:02d}": np.zeros(900) for i in range(8)})
    inference = read_json(predictions / "predictions.json")
    inference.update(
        views_manifest_sha256=sha256(views / "manifest.json"),
        logits_sha256=sha256(predictions / "logits.npz"),
        predictions=[{"id": i, "angles_degrees": [0, 0, 0], "symmetry_fit": 0} for i in range(8)],
    )
    write_json(predictions / "predictions.json", inference)
    calls = []

    def fit(cameras: Any, logits: Any) -> dict[str, Any]:
        calls.append(len(cameras))
        if len(calls) == 2:
            raise RuntimeError("Interrupted fixture fit")
        return {"candidates": [{"mesh_to_oa": np.eye(3).tolist()}]}

    monkeypatch.setattr(stability, "fuse", fit)
    job = tmp_path / "job"
    with pytest.raises(RuntimeError, match="Interrupted"):
        stability.evaluate(views, predictions, job)
    assert read_json(job / "state.json")["status"] == "failed"
    stability.evaluate(views, predictions, job, resume=True)
    assert calls.count(8) == 1  # The completed full fit was reused.
    assert len(calls) == len(stability.subsets(8)) + 1
    assert read_json(job / "summary.json")["decoded_pairwise_up_statistics"]["max"] == 0
    before = len(calls)
    (job / "work/all-1/fit.json").write_text("{}")
    with pytest.raises(ValueError, match="modified"):
        stability.evaluate(views, predictions, job, resume=True)
    assert len(calls) == before

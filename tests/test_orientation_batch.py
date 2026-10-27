from pathlib import Path
from typing import Any

import pytest

from experiments.artifacts import read_json, sha256, write_json
from experiments.orientation import batch, infer


def test_batch_budget_retry_preserves_success_and_rejects_corruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    renders = tmp_path / "renders"
    renders.mkdir()
    cases = []
    for i in range(3):
        key = f"case-{i:016x}"
        folder = renders / key
        folder.mkdir()
        (folder / "source.glb").write_bytes(b"fixture mesh")
        views = []
        for j in range(3):
            (folder / f"{j}.png").write_bytes(b"fixture image")
            views.append({"id": j, "image": f"{j}.png", "sha256": sha256(folder / f"{j}.png")})
        write_json(
            folder / "manifest.json",
            {
                "schema_version": 1,
                "status": "succeeded",
                "mesh": "source.glb",
                "mesh_sha256": sha256(folder / "source.glb"),
                "views": views,
            },
        )
        cases.append(
            {
                "case_id": key,
                "partition": "development" if i < 2 else "holdout",
                "views_directory": key,
                "views_manifest_sha256": sha256(folder / "manifest.json"),
                "mesh_sha256": sha256(folder / "source.glb"),
            }
        )
    write_json(renders / "index.json", {"status": "succeeded", "cases": cases})
    calls = []
    fail = False

    def predict(views: Path, output: Path, *_args: Any) -> Path:
        calls.append(views.name)
        output.mkdir()
        if fail:
            (output / "failure.txt").write_text("interrupted")
            raise RuntimeError("GPU interrupted")
        (output / "logits.npz").write_bytes(b"fixture logits")
        write_json(
            output / "predictions.json",
            {
                "status": "succeeded",
                "views_manifest_sha256": sha256(views / "manifest.json"),
                "logits_sha256": sha256(output / "logits.npz"),
                "predictions": [{"id": j} for j in range(3)],
            },
        )
        return output

    monkeypatch.setattr(infer, "run", predict)
    output = tmp_path / "results"
    batch.run(renders, output, "development", tmp_path, tmp_path, budget=1)
    assert read_json(output / "batch.json")["status"] == "budget_exhausted"
    fail = True
    with pytest.raises(RuntimeError, match="interrupted"):
        batch.run(renders, output, "development", tmp_path, tmp_path)
    fail = False
    batch.run(renders, output, "development", tmp_path, tmp_path)
    result = read_json(output / "batch.json")
    assert result["status"] == "succeeded"
    assert calls == [cases[0]["case_id"], cases[1]["case_id"], cases[1]["case_id"]]
    assert list(output.glob("*/failure.txt"))
    first = output / result["cases"][cases[0]["case_id"]]["directory"]
    (first / "logits.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="mismatched"):
        batch.run(renders, output, "development", tmp_path, tmp_path)
    assert len(calls) == 3
    with pytest.raises(ValueError, match="different input"):
        batch.run(renders, output, "holdout", tmp_path, tmp_path)

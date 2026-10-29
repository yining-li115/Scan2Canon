from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from experiments.artifacts import read_json, sha256, write_json
from experiments.orientation import image_probe
from experiments.orientation.common import verify_views


def bundle(root: Path) -> Path:
    root.mkdir()
    Image.new("RGB", (24, 40), "red").save(root / "photo.png")
    write_json(
        root / "images.json",
        {
            "schema_version": 1,
            "kind": "oa_image_probe",
            "images": [{"id": 0, "image": "photo.png", "sha256": sha256(root / "photo.png")}],
        },
    )
    return root


def test_modified_photo_fails_before_model_load(tmp_path: Path, monkeypatch: Any) -> None:
    inputs = bundle(tmp_path / "inputs")
    (inputs / "photo.png").write_bytes(b"modified")
    monkeypatch.setattr(image_probe, "verify_source", lambda _: pytest.fail("Model setup reached"))
    with pytest.raises(ValueError, match="modified artifact"):
        image_probe.run(inputs, tmp_path / "output", tmp_path, tmp_path)
    assert read_json(tmp_path / "output/probe.json")["status"] == "failed"


def test_probe_preserves_failure_and_cannot_be_used_as_registered_views(
    tmp_path: Path, monkeypatch: Any
) -> None:
    inputs = bundle(tmp_path / "inputs")
    output = tmp_path / "output"
    monkeypatch.setattr(image_probe, "verify_source", lambda _: None)
    monkeypatch.setattr(image_probe, "verify_checkpoint", lambda _: (tmp_path / "weights", {}))

    def interrupted(*args: Any, **kwargs: Any) -> dict[str, Any]:
        write_json(output / "partial-predictions.json", {"predictions": []})
        raise RuntimeError("GPU interrupted")

    monkeypatch.setattr(image_probe, "predict_images", interrupted)
    with pytest.raises(RuntimeError, match="GPU interrupted"):
        image_probe.run(inputs, output, tmp_path, tmp_path)
    assert read_json(output / "probe.json")["status"] == "failed"
    assert (output / "partial-predictions.json").is_file()
    with pytest.raises(FileExistsError):
        image_probe.run(inputs, output, tmp_path, tmp_path)
    with pytest.raises(FileNotFoundError):
        verify_views(inputs)


@pytest.mark.parametrize("ids", [[0, 0], [True], [-1]])
def test_duplicate_or_invalid_ids_are_rejected(tmp_path: Path, ids: list[Any]) -> None:
    inputs = bundle(tmp_path / "inputs")
    manifest = read_json(inputs / "images.json")
    row = manifest["images"][0]
    manifest["images"] = [{**row, "id": i} for i in ids]
    write_json(inputs / "images.json", manifest)
    with pytest.raises(ValueError, match="Image IDs"):
        image_probe.verify_images(inputs)

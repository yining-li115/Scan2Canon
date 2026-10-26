import json
from pathlib import Path
from typing import Any

import pytest
from PIL import Image, ImageDraw

from experiments.artifacts import sha256, write_json
from experiments.sam3d.common import config
from experiments.sam3d.download import verify_checkpoint
from experiments.sam3d.prepare import prepare, verify_inputs
from experiments.sam3d.run import run


@pytest.fixture
def masked_input(tmp_path: Path) -> Path:
    image, mask = tmp_path / "image.png", tmp_path / "mask.png"
    Image.new("RGB", (32, 48), "gray").save(image)
    foreground = Image.new("L", (32, 48))
    ImageDraw.Draw(foreground).rectangle((8, 8, 24, 40), fill=255)
    foreground.save(mask)
    return prepare(image, mask, tmp_path / "input", "manual test annotation")


def test_masked_input_preserves_sources_and_survives_move(masked_input: Path) -> None:
    manifest = json.loads((masked_input / "manifest.json").read_text())
    assert sha256(masked_input / "source.png") == manifest["source_sha256"]
    moved = masked_input.rename(masked_input.with_name("moved"))
    image, mask = verify_inputs(moved)
    assert image.is_file() and mask.is_file()
    with pytest.raises(FileExistsError):
        prepare(image, mask, moved, "manual")


@pytest.mark.parametrize("kind", ["empty", "full", "soft", "wrong_size"])
def test_invalid_masks_are_rejected_before_output_creation(tmp_path: Path, kind: str) -> None:
    image, mask = tmp_path / "image.png", tmp_path / "mask.png"
    Image.new("RGB", (32, 48)).save(image)
    size = (16, 24) if kind == "wrong_size" else (32, 48)
    fill = {"empty": 0, "full": 255, "soft": 127, "wrong_size": 0}[kind]
    Image.new("L", size, fill).save(mask)
    output = tmp_path / "prepared"
    with pytest.raises(ValueError, match="Mask must"):
        prepare(image, mask, output, "test")
    assert not output.exists()


def test_modified_mask_cannot_be_used(masked_input: Path) -> None:
    (masked_input / "mask.png").write_bytes(b"different")
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_inputs(masked_input)


def test_checkpoint_manifest_cannot_escape_bundle(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.write_bytes(b"weights")
    bundle = tmp_path / "weights"
    (bundle / "checkpoints").mkdir(parents=True)
    (bundle / "checkpoints/pipeline.yaml").write_bytes(b"config")
    write_json(
        bundle / "manifest.json",
        {
            "settings": config(),
            "files": {
                "checkpoints/pipeline.yaml": sha256(bundle / "checkpoints/pipeline.yaml"),
                "../outside": sha256(outside),
            },
        },
    )
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_checkpoint(bundle)


def test_generation_failure_is_recorded_without_approval(
    masked_input: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import experiments.sam3d.run as runner

    monkeypatch.setattr(runner, "verify_source", lambda _: None)
    monkeypatch.setattr(runner, "verify_checkpoint", lambda _: {"settings": config()})

    def fail(*args: Any) -> Any:
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(runner, "infer", fail)
    output = tmp_path / "run"
    with pytest.raises(RuntimeError, match="out of memory"):
        run(masked_input, tmp_path / "weights", tmp_path / "source", output)
    report = json.loads((output / "run.json").read_text())
    assert report["status"] == "failed"
    assert report["dataset_approval"] == "pending_review"
    assert report["held_out_view_validation"] == "not_performed"
    assert "artifacts" not in report
    with pytest.raises(FileExistsError):
        run(masked_input, tmp_path / "weights", tmp_path / "source", output)

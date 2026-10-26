import json
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from experiments.artifacts import sha256, verified_inputs, write_json
from experiments.mapanything.common import config
from experiments.mapanything.prepare import prepare
from experiments.mapanything.run import run


@pytest.fixture
def capture(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    source.mkdir()
    for index in range(9):
        Image.new("RGB", (32, 48), (index * 20, 10, 0)).save(source / f"{index:03d}.png")
    return source


def test_prepare_portable_ordered_subset(capture: Path, tmp_path: Path) -> None:
    output = prepare(capture, tmp_path / "prepared", views=3)
    manifest = json.loads((output / "manifest.json").read_text())
    assert [frame["source_index"] for frame in manifest["frames"]] == [0, 4, 8]
    assert len(verified_inputs(output)) == 3
    for frame in manifest["frames"]:
        assert frame["source_sha256"] == sha256(capture / frame["source_name"])
        with Image.open(output / frame["path"]) as image:
            assert image.mode == "RGB"
            assert image.size == (32, 48)
    moved = output.rename(tmp_path / "moved")
    assert len(verified_inputs(moved)) == 3


def test_prepare_allows_single_image_baseline(capture: Path, tmp_path: Path) -> None:
    output = prepare(capture, tmp_path / "single", views=1)
    assert len(verified_inputs(output)) == 1


@pytest.mark.parametrize("views", [0, 10])
def test_prepare_rejects_invalid_view_count(capture: Path, tmp_path: Path, views: int) -> None:
    with pytest.raises(ValueError, match="Requested"):
        prepare(capture, tmp_path / "prepared", views)
    assert not (tmp_path / "prepared").exists()


def test_prepare_rejects_duplicates_without_partial_capture(capture: Path, tmp_path: Path) -> None:
    (capture / "008.png").write_bytes((capture / "000.png").read_bytes())
    with pytest.raises(ValueError, match="duplicate"):
        prepare(capture, tmp_path / "prepared", views=2)
    assert not (tmp_path / "prepared").exists()


def test_prepared_inputs_cannot_be_overwritten(capture: Path, tmp_path: Path) -> None:
    output = prepare(capture, tmp_path / "prepared", views=3)
    original = (output / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        prepare(capture, output, views=1)
    assert (output / "manifest.json").read_bytes() == original


def test_input_hash_detects_changed_pixels(capture: Path, tmp_path: Path) -> None:
    output = prepare(capture, tmp_path / "prepared", views=3)
    verified_inputs(output)[0].write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum mismatch"):
        verified_inputs(output)


def test_manifest_cannot_reference_external_files(tmp_path: Path) -> None:
    outside = tmp_path / "outside.jpg"
    outside.write_bytes(b"outside")
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    write_json(
        inputs / "manifest.json",
        {
            "schema_version": 1,
            "frames": [{"path": "../outside.jpg", "sha256": sha256(outside)}],
        },
    )
    with pytest.raises(ValueError, match="escapes"):
        verified_inputs(inputs)


def test_failure_is_recorded_without_overwriting_previous_run(tmp_path: Path) -> None:
    output = tmp_path / "run"
    with pytest.raises(FileNotFoundError):
        run(tmp_path / "missing", tmp_path / "weights", output)
    report = json.loads((output / "run.json").read_text())
    assert report["status"] == "failed"
    assert report["error"]["type"] == "FileNotFoundError"
    with pytest.raises(FileExistsError):
        run(tmp_path / "missing", tmp_path / "weights", output)
    assert json.loads((output / "run.json").read_text()) == report


def test_inference_error_is_preserved(
    capture: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import experiments.mapanything.run as runner

    inputs = prepare(capture, tmp_path / "inputs", views=3)
    checkpoint = tmp_path / "checkpoint"
    (checkpoint / "model").mkdir(parents=True)
    files = {}
    for name in ("config.json", "model.safetensors"):
        path = checkpoint / "model" / name
        path.write_bytes(b"test checkpoint")
        files[name] = sha256(path)
    write_json(checkpoint / "manifest.json", {**config(), "files": files})
    monkeypatch.setattr(runner, "verify_dinov2", lambda _: None)

    def driver(*args: Any, **kwargs: Any) -> Any:
        from subprocess import CompletedProcess

        return CompletedProcess(args, 0, stdout="test driver")

    def failure(*args: Any) -> Any:
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr("experiments.mapanything.run.subprocess.run", driver)
    monkeypatch.setattr(runner, "infer", failure)
    with pytest.raises(RuntimeError, match="out of memory"):
        run(inputs, checkpoint, tmp_path / "run")
    report = json.loads((tmp_path / "run" / "run.json").read_text())
    assert report["status"] == "failed"
    assert report["error"]["message"] == "CUDA out of memory"
    assert (tmp_path / "run" / "input-manifest.json").is_file()
    assert "metrics" not in report

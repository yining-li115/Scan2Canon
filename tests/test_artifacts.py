import hashlib
from io import BytesIO
from pathlib import Path
from uuid import uuid4

import pillow_heif
import pytest
from PIL import Image, UnidentifiedImageError

from scan2canon.storage.artifacts import LocalArtifacts


def test_capture_is_a_snapshot_not_a_source_path(tmp_path: Path, images: Path) -> None:
    store = LocalArtifacts(tmp_path / "artifacts")
    capture = store.import_capture(images)
    original = (images / "0.png").read_bytes()
    stored = store.read(capture.frames[0].key)
    (images / "0.png").unlink()
    assert store.read(capture.frames[0].key) == stored
    assert store.capture(capture.id) == capture
    assert capture.schema_version == 2
    frame = capture.frames[0]
    assert frame.source_key is not None
    assert store.read(frame.source_key) == original
    assert frame.source_sha256 == hashlib.sha256(original).hexdigest()
    assert frame.sha256 == hashlib.sha256(stored).hexdigest()
    assert frame.key.endswith(".jpg")


def test_heic_heif_and_jpeg_can_share_a_capture(tmp_path: Path, images: Path) -> None:
    store = LocalArtifacts(tmp_path / "artifacts")
    for index, extension in enumerate(("HEIC", "heif")):
        source = images / f"{index}.png"
        with Image.open(source) as image:
            pillow_heif.from_pillow(image).save(images / f"{index}.{extension}")
        source.unlink()
    with Image.open(images / "2.png") as image:
        image.save(images / "2.jpg")
    (images / "2.png").unlink()
    original_files = sorted(images.iterdir())
    capture = store.import_capture(images)
    for frame, source in zip(capture.frames, original_files, strict=True):
        assert frame.source_key is not None
        assert store.read(frame.source_key) == source.read_bytes()
        with Image.open(BytesIO(store.read(frame.key))) as image:
            image.load()
            assert image.format == "JPEG" and image.mode == "RGB"


def test_old_capture_manifests_remain_readable(tmp_path: Path, images: Path) -> None:
    store = LocalArtifacts(tmp_path / "artifacts")
    capture = store.import_capture(images)
    # The original schema had only the frame key, hash and dimensions.
    payload = capture.model_dump_json(
        exclude={
            "schema_version": True,
            "frames": {"__all__": {"source_key", "source_sha256", "normalization_version"}},
        }
    )
    (store.root / "captures" / str(capture.id) / "manifest.json").write_text(payload)
    restored = store.capture(capture.id)
    assert restored.schema_version == 1
    assert restored.frames[0].source_key is None
    assert store.exists(restored.frames[0].key)


def test_two_heic_views_still_do_not_meet_capture_requirement(tmp_path: Path) -> None:
    source = tmp_path / "photos"
    source.mkdir()
    for index, color in enumerate(("red", "blue")):
        pillow_heif.from_pillow(Image.new("RGB", (32, 32), color)).save(source / f"{index}.HEIC")
    with pytest.raises(ValueError, match="3 to 200"):
        LocalArtifacts(tmp_path / "artifacts").import_capture(source)


def test_invalid_capture_leaves_no_partial_manifest(tmp_path: Path, images: Path) -> None:
    store = LocalArtifacts(tmp_path / "artifacts")
    (images / "2.png").write_bytes(b"not an image")
    with pytest.raises(UnidentifiedImageError):
        store.import_capture(images)
    assert list(store.root.iterdir()) == []


def test_duplicate_images_are_not_three_views(tmp_path: Path, images: Path) -> None:
    store = LocalArtifacts(tmp_path / "artifacts")
    for path in images.iterdir():
        Image.new("RGB", (32, 24), "red").save(path)
    with pytest.raises(ValueError, match="distinct"):
        store.import_capture(images)
    assert list(store.root.iterdir()) == []


@pytest.mark.parametrize("key", ["../secret", "/etc/passwd", "sub/../../secret", "jobs/a/../b"])
def test_artifact_paths_cannot_escape(tmp_path: Path, key: str) -> None:
    store = LocalArtifacts(tmp_path / "artifacts")
    with pytest.raises(ValueError, match="within"):
        store.read(key)


def test_symlink_cannot_escape(tmp_path: Path) -> None:
    store = LocalArtifacts(tmp_path / "artifacts")
    outside = tmp_path / "secret"
    outside.write_text("private")
    (store.root / "link").symlink_to(outside)
    with pytest.raises(ValueError, match="within"):
        store.read("link")


def test_repeated_artifact_names_do_not_overwrite(tmp_path: Path) -> None:
    store = LocalArtifacts(tmp_path / "artifacts")
    job_id = uuid4()
    first = store.write(job_id, "report.json", b"first")
    second = store.write(job_id, "report.json", b"second")
    assert first != second
    assert store.read(first) == b"first"

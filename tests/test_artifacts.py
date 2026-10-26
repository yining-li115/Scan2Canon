from pathlib import Path
from uuid import uuid4

import pytest
from PIL import Image, UnidentifiedImageError

from scan2canon.storage.artifacts import LocalArtifacts


def test_capture_is_a_snapshot_not_a_source_path(tmp_path: Path, images: Path) -> None:
    store = LocalArtifacts(tmp_path / "artifacts")
    capture = store.import_capture(images)
    stored = store.read(capture.frames[0].key)
    (images / "0.png").unlink()
    assert store.read(capture.frames[0].key) == stored
    assert store.capture(capture.id) == capture


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

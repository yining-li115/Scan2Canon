from pathlib import Path

import pytest
from PIL import Image

np = pytest.importorskip("numpy")

from experiments.artifacts import read_json  # noqa: E402
from experiments.datasets.nocs_photos import prepare  # noqa: E402


def make_frame(directory: Path) -> Path:
    directory.mkdir()
    frame = directory / "0000"
    (directory / "0000_meta.txt").write_text("1 3 camera_fixture\n2 5 laptop_fixture\n")
    Image.new("RGB", (80, 60), "blue").save(directory / "0000_color.png")
    mask = np.zeros((60, 80, 3), dtype=np.uint8)
    mask[20:40, 30:50, 0] = 1
    Image.fromarray(mask).save(directory / "0000_mask.png")
    return frame


def test_photo_preparation_uses_identity_and_mask_not_pose_labels(tmp_path: Path) -> None:
    frame = make_frame(tmp_path / "scene")
    output = prepare([frame], "camera_fixture", "camera", "a" * 64, tmp_path / "evidence")
    bundle = read_json(output / "manifest.json")
    assert bundle["registration"] == "unregistered"
    assert bundle["images"][0]["bounding_box_xyxy"] == [18, 8, 62, 52]
    assert bundle["images"][0]["visible_mask_pixels"] == 400
    assert len(bundle["assets"]) == 3
    assert {p.suffix for p in output.iterdir()} == {".png", ".txt", ".json"}
    with pytest.raises(FileExistsError):
        prepare([frame], "camera_fixture", "camera", "a" * 64, output)


@pytest.mark.parametrize(
    ("instance", "category", "error"),
    [
        ("other_camera", "camera", "matching"),
        ("camera_fixture", "laptop", "matching"),
        ("laptop_fixture", "laptop", "insufficient"),
    ],
)
def test_wrong_or_invisible_instance_does_not_create_evidence(
    tmp_path: Path, instance: str, category: str, error: str
) -> None:
    frame = make_frame(tmp_path / "scene")
    output = tmp_path / "evidence"
    with pytest.raises(ValueError, match=error):
        prepare([frame], instance, category, "a" * 64, output)
    assert not output.exists()

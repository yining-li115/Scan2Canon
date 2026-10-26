from pathlib import Path

import pytest
from PIL import Image

from scan2canon.runtime import Runtime, build_runtime


@pytest.fixture
def images(tmp_path: Path) -> Path:
    directory = tmp_path / "images"
    directory.mkdir()
    for index, color in enumerate(("red", "green", "blue")):
        Image.new("RGB", (32, 24), color).save(directory / f"{index}.png")
    return directory


@pytest.fixture
def runtime(tmp_path: Path) -> Runtime:
    return build_runtime(tmp_path / "state")

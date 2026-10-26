"""Download the pinned OA V2 checkpoint and verify its local provenance."""

import argparse
from pathlib import Path
from typing import Any

from experiments.artifacts import read_json, sha256, write_json
from experiments.orientation.common import checked_file, config


def verify_checkpoint(directory: Path) -> tuple[Path, dict[str, Any]]:
    manifest = read_json(directory / "manifest.json")
    if manifest["settings"] != config():
        raise ValueError("Checkpoint settings differ from the pinned configuration")
    path = checked_file(directory, config()["checkpoint_file"], manifest["sha256"])
    return path, manifest


def download(directory: Path) -> Path:
    from huggingface_hub import hf_hub_download

    directory = directory.resolve()
    if (directory / "manifest.json").exists():
        verify_checkpoint(directory)
        return directory
    settings = config()
    path = Path(
        hf_hub_download(
            settings["checkpoint"],
            settings["checkpoint_file"],
            revision=settings["checkpoint_revision"],
            local_dir=str(directory),
        )
    )
    write_json(directory / "manifest.json", {"settings": settings, "sha256": sha256(path)})
    verify_checkpoint(directory)
    return directory


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("weights/orientation"))
    print(download(parser.parse_args().checkpoint_dir))

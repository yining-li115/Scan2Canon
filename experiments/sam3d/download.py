"""Download an authorized, pinned SAM 3D checkpoint and record file hashes."""

import argparse
from pathlib import Path

from experiments.artifacts import read_json, sha256, write_json
from experiments.sam3d.common import config


def verify_checkpoint(directory: Path) -> dict[str, object]:
    manifest = read_json(directory / "manifest.json")
    if manifest.get("settings") != config():
        raise ValueError("Checkpoint settings do not match the pinned experiment")
    files = manifest.get("files", {})
    if "checkpoints/pipeline.yaml" not in files:
        raise ValueError("Checkpoint manifest is missing pipeline.yaml")
    for name, checksum in files.items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory.resolve()) or sha256(path) != checksum:
            raise ValueError(f"Checkpoint checksum mismatch: {name}")
    return manifest


def download(directory: Path) -> Path:
    from huggingface_hub import snapshot_download

    if (directory / "manifest.json").exists():
        verify_checkpoint(directory)
        return directory
    settings = config()
    # Hugging Face reads credentials from its normal login store. Never log tokens.
    snapshot_download(
        repo_id=settings["checkpoint"],
        revision=settings["checkpoint_revision"],
        local_dir=directory,
        allow_patterns=["checkpoints/*", "LICENSE*", "README.md"],
        max_workers=2,
    )
    files = {
        str(path.relative_to(directory)): sha256(path)
        for path in sorted((directory / "checkpoints").rglob("*"))
        if path.is_file()
    }
    if "checkpoints/pipeline.yaml" not in files:
        raise ValueError("Incomplete download: missing pipeline.yaml")
    write_json(
        directory / "manifest.json", {"schema_version": 1, "settings": settings, "files": files}
    )
    return directory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("weights/sam3d"))
    args = parser.parse_args()
    print(download(args.checkpoint_dir))


if __name__ == "__main__":
    main()

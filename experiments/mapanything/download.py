"""Download the pinned research checkpoint and its Torch Hub source dependency."""

import argparse
import subprocess
from pathlib import Path

from experiments.artifacts import sha256, write_json
from experiments.mapanything.common import config, verify_dinov2


def download(directory: Path) -> None:
    from huggingface_hub import snapshot_download

    settings = config()
    directory.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=settings["checkpoint"],
        revision=settings["checkpoint_revision"],
        local_dir=directory / "model",
        allow_patterns=["config.json", "model.safetensors", "README.md"],
    )
    # UniCeption requests this exact Torch Hub cache name. Populate it with a
    # verified commit so model construction does not select a moving branch.
    hub = directory / "torch" / "hub" / "facebookresearch_dinov2_main"
    if not hub.exists():
        hub.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "init", str(hub)], check=True)
    head = subprocess.run(
        ["git", "-C", str(hub), "rev-parse", "--verify", "HEAD"], capture_output=True
    )
    if head.returncode != 0:
        subprocess.run(
            [
                "git",
                "-C",
                str(hub),
                "fetch",
                "--depth=1",
                "https://github.com/facebookresearch/dinov2.git",
                settings["dinov2_revision"],
            ],
            check=True,
        )
        subprocess.run(["git", "-C", str(hub), "checkout", "--detach", "FETCH_HEAD"], check=True)
    verify_dinov2(hub)
    write_json(
        directory / "manifest.json",
        {
            **settings,
            "files": {
                name: sha256(directory / "model" / name)
                for name in ("config.json", "model.safetensors")
            },
        },
    )
    print(f"Downloaded and hashed checkpoint: {directory.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("weights/mapanything"))
    args = parser.parse_args()
    download(args.checkpoint_dir)


if __name__ == "__main__":
    main()

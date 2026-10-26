"""Pinned MapAnything model settings and DINOv2 source verification."""

import subprocess
import tomllib
from pathlib import Path

CONFIG_PATH = Path(__file__).with_name("pyproject.toml")


def config() -> dict[str, str]:
    with CONFIG_PATH.open("rb") as handle:
        settings: dict[str, str] = tomllib.load(handle)["tool"]["scan2canon"]
    return settings


def verify_dinov2(directory: Path) -> None:
    head = subprocess.check_output(
        ["git", "-C", str(directory), "rev-parse", "HEAD"], text=True
    ).strip()
    changes = subprocess.check_output(
        ["git", "-C", str(directory), "status", "--porcelain", "--untracked-files=no"], text=True
    ).strip()
    if head != config()["dinov2_revision"] or changes:
        raise ValueError("DINOv2 cache does not match the clean pinned revision")

"""Pinned model settings and source verification."""

import subprocess
import tomllib
from pathlib import Path

CONFIG_PATH = Path(__file__).with_name("pyproject.toml")


def config() -> dict[str, str]:
    with CONFIG_PATH.open("rb") as handle:
        settings: dict[str, str] = tomllib.load(handle)["tool"]["scan2canon"]
    return settings


def verify_source(source: Path) -> None:
    head = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    changes = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"], text=True
    ).strip()
    if head != config()["source_revision"] or changes:
        raise ValueError("SAM 3D source does not match the clean pinned revision")

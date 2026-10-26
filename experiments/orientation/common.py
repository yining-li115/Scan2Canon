"""Pinned upstream source and verified portable render bundles."""

import subprocess
import tomllib
from pathlib import Path
from typing import Any

from experiments.artifacts import read_json, sha256

CONFIG_PATH = Path(__file__).with_name("pyproject.toml")


def config() -> dict[str, str]:
    with CONFIG_PATH.open("rb") as handle:
        return dict(tomllib.load(handle)["tool"]["scan2canon"])


def verify_source(source: Path) -> None:
    head = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"], text=True
    ).strip()
    if head != config()["source_revision"] or dirty:
        raise ValueError("OA V2 source must match the clean pinned revision")


def checked_file(root: Path, relative: str, digest: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file() or sha256(path) != digest:
        raise ValueError(f"Missing, unsafe or modified artifact: {relative}")
    return path


def verify_views(directory: Path) -> dict[str, Any]:
    manifest = read_json(directory / "manifest.json")
    if manifest.get("schema_version") != 1 or manifest.get("status") != "succeeded":
        raise ValueError("Render bundle is incomplete or unsupported")
    checked_file(directory, manifest["mesh"], manifest["mesh_sha256"])
    views = manifest["views"]
    if not 3 <= len(views) <= 64 or len({v["id"] for v in views}) != len(views):
        raise ValueError("Expected 3–64 unique rendered views")
    for view in views:
        checked_file(directory, view["image"], view["sha256"])
    return manifest

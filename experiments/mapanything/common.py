"""File provenance shared by preparation, downloads, and inference."""

import hashlib
import json
import subprocess
import tomllib
from pathlib import Path
from typing import Any

CONFIG_PATH = Path(__file__).with_name("pyproject.toml")


def config() -> dict[str, str]:
    with CONFIG_PATH.open("rb") as handle:
        settings: dict[str, str] = tomllib.load(handle)["tool"]["scan2canon"]
    return settings


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def verified_inputs(directory: Path) -> list[Path]:
    manifest = read_json(directory / "manifest.json")
    if manifest.get("schema_version") != 1 or not manifest.get("frames"):
        raise ValueError("Unsupported or empty input manifest")
    paths = []
    for frame in manifest["frames"]:
        path = (directory / frame["path"]).resolve()
        if not path.is_relative_to(directory.resolve()):
            raise ValueError("Input path escapes the prepared capture")
        if sha256(path) != frame["sha256"]:
            raise ValueError(f"Input checksum mismatch: {path.name}")
        paths.append(path)
    if len(paths) != len(set(paths)):
        raise ValueError("Repeated input paths in manifest")
    return paths


def verify_dinov2(directory: Path) -> None:
    head = subprocess.check_output(
        ["git", "-C", str(directory), "rev-parse", "HEAD"], text=True
    ).strip()
    changes = subprocess.check_output(
        ["git", "-C", str(directory), "status", "--porcelain", "--untracked-files=no"], text=True
    ).strip()
    if head != config()["dinov2_revision"] or changes:
        raise ValueError("DINOv2 cache does not match the clean pinned revision")

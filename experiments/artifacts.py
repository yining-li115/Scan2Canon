"""Checksummed files and portable manifests for reconstruction experiments."""

import hashlib
import json
from pathlib import Path
from typing import Any


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

"""Pinned sources, weights, and portable experiment inputs."""

import argparse
import subprocess
import tomllib
from pathlib import Path
from typing import Any

from experiments.artifacts import read_json, sha256, write_json
from experiments.orientation.common import checked_file, verify_views
from experiments.semantic_orientation.contracts import Rule


def config() -> dict[str, str]:
    with Path(__file__).with_name("pyproject.toml").open("rb") as handle:
        return dict(tomllib.load(handle)["tool"]["scan2canon"])


def verify_source(source: Path) -> None:
    head = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True)
    dirty = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"], text=True
    )
    if head.strip() != config()["source_revision"] or dirty.strip():
        raise ValueError("SoFar source must match the clean pinned revision")


def verify_case(directory: Path) -> dict[str, Any]:
    case = read_json(directory / "case.json")
    if case.get("schema_version") != 1 or case.get("status") != "succeeded":
        raise ValueError("Incomplete semantic experiment case")
    Rule.model_validate(case["rule"])
    for key in ("mesh", "points"):
        checked_file(directory, case[key], case[f"{key}_sha256"])
    views = verify_views(directory / "ring")
    if sha256(directory / "ring/manifest.json") != case["ring_manifest_sha256"]:
        raise ValueError("Modified ring manifest")
    if {v["id"] for v in views["views"]} != {v["id"] for v in case["candidates"]}:
        raise ValueError("Candidate coverage differs from rendered views")
    return case


def verify_weights(directory: Path, kind: str) -> dict[str, Any]:
    report = read_json(directory / "manifest.json")
    if report["settings"] != config() or report["kind"] != kind:
        raise ValueError("Weights do not match this experiment")
    for name, digest in report["files"].items():
        checked_file(directory, name, digest)
    return report


def download(directory: Path, kind: str) -> Path:
    from huggingface_hub import hf_hub_download, snapshot_download

    if (directory / "manifest.json").exists():
        verify_weights(directory, kind)
        return directory
    settings = config()
    if kind == "pointso":
        paths = [
            Path(
                hf_hub_download(
                    settings["pointso_repository"],
                    settings["pointso_file"],
                    revision=settings["pointso_revision"],
                    local_dir=str(directory),
                )
            )
        ]
    else:
        snapshot_download(
            settings["vlm_repository"],
            revision=settings["vlm_revision"],
            local_dir=str(directory),
            allow_patterns=["*.json", "*.safetensors", "*.txt"],
        )
        paths = [p for p in directory.iterdir() if p.is_file()]
    write_json(
        directory / "manifest.json",
        {
            "kind": kind,
            "settings": settings,
            "files": {str(p.relative_to(directory)): sha256(p) for p in paths},
        },
    )
    verify_weights(directory, kind)
    return directory


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("pointso", "vlm"))
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(download(args.directory, args.kind))

"""Probe OA on checksummed photographs or renders without claiming mesh registration."""

import argparse
from pathlib import Path
from typing import Any

from experiments.artifacts import read_json, sha256, write_json
from experiments.orientation.common import checked_file, config, verify_source
from experiments.orientation.download import verify_checkpoint
from experiments.orientation.infer import predict_images


def verify_images(directory: Path) -> list[tuple[int, Path]]:
    manifest = read_json(directory / "images.json")
    if manifest.get("schema_version") != 1 or manifest.get("kind") != "oa_image_probe":
        raise ValueError("Expected an OA image-probe bundle")
    rows = manifest.get("images", [])
    if not 1 <= len(rows) <= 128:
        raise ValueError("Expected 1–128 diagnostic images")
    ids = [row["id"] for row in rows]
    if any(type(i) is not int or i < 0 for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("Image IDs must be unique nonnegative integers")
    return [(row["id"], checked_file(directory, row["image"], row["sha256"])) for row in rows]


def run(inputs: Path, output: Path, source: Path, checkpoint: Path) -> Path:
    output.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {
        "schema_version": 1,
        "kind": "oa_image_probe",
        "status": "running",
        "settings": config(),
        "mesh_registration": "not_established_by_this_probe",
        "scope": "Image-space predictions only; not a canonical mesh or accuracy certificate",
        "runner_sha256": sha256(Path(__file__)),
        "forward_runner_sha256": sha256(Path(__file__).with_name("infer.py")),
    }
    write_json(output / "probe.json", report)
    try:
        images = verify_images(inputs)
        verify_source(source)
        weights, manifest = verify_checkpoint(checkpoint)
        report["input_manifest_sha256"] = sha256(inputs / "images.json")
        write_json(output / "checkpoint-manifest.json", manifest)
        report.update(predict_images(images, source, weights, output, verify_official=True))
        report["logits_sha256"] = sha256(output / "logits.npz")
        report["status"] = "succeeded"
    except BaseException as error:
        report.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        write_json(output / "probe.json", report)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source-dir", type=Path, default=Path("data/tools/Orient-Anything-V2"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("weights/orientation"))
    args = parser.parse_args()
    print(run(args.inputs, args.output, args.source_dir.resolve(), args.checkpoint_dir.resolve()))


if __name__ == "__main__":
    main()

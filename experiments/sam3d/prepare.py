"""Prepare an upright image and an explicit object mask for single-image inference."""

import argparse
import shutil
from pathlib import Path

from PIL import Image

from experiments.artifacts import read_json, sha256, write_json


def validate_pair(image_path: Path, mask_path: Path) -> None:
    with Image.open(image_path) as image, Image.open(mask_path) as mask:
        if image.mode != "RGB" or image.getexif().get(274, 1) != 1:
            raise ValueError("Use an upright RGB image from capture normalization")
        if mask.size != image.size or mask.mode != "L":
            raise ValueError("Mask must be grayscale and match the upright image dimensions")
        if mask.getexif().get(274, 1) != 1:
            raise ValueError("Mask must be upright")
        histogram = mask.histogram()
        if sum(histogram[1:255]) or not histogram[0] or not histogram[255]:
            raise ValueError("Mask must contain both background (0) and object (255) only")


def prepare(image: Path, mask: Path, output: Path, mask_method: str) -> Path:
    if not mask_method.strip():
        raise ValueError("Record how the object mask was obtained")
    validate_pair(image, mask)
    output.mkdir(parents=True, exist_ok=False)
    # Preserve exact source bytes as well as lossless, metadata-free model inputs.
    shutil.copyfile(image, output / ("source" + image.suffix.lower()))
    shutil.copyfile(mask, output / "source-mask.png")
    with Image.open(image) as rgb, Image.open(mask) as binary:
        rgb.info.clear()
        rgb.save(output / "image.png")
        binary.info.clear()
        binary.save(output / "mask.png")
        width, height = rgb.size
    write_json(
        output / "manifest.json",
        {
            "schema_version": 1,
            "kind": "masked_single_image",
            "source_name": image.name,
            "source_sha256": sha256(image),
            "source_mask_sha256": sha256(mask),
            "mask_method": mask_method,
            "width": width,
            "height": height,
            "files": {name: sha256(output / name) for name in ("image.png", "mask.png")},
            "held_out_views": "not_used_for_generation",
        },
    )
    return output


def verify_inputs(inputs: Path) -> tuple[Path, Path]:
    manifest = read_json(inputs / "manifest.json")
    if manifest.get("schema_version") != 1 or manifest.get("kind") != "masked_single_image":
        raise ValueError("Unsupported input manifest")
    for name in ("image.png", "mask.png"):
        if sha256(inputs / name) != manifest["files"][name]:
            raise ValueError(f"Input checksum mismatch: {name}")
    image, mask = inputs / "image.png", inputs / "mask.png"
    validate_pair(image, mask)
    return image, mask


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("mask", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--mask-method", required=True)
    args = parser.parse_args()
    print(prepare(args.image, args.mask, args.output, args.mask_method))


if __name__ == "__main__":
    main()

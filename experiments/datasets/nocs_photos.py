"""Prepare identity-matched RGB evidence without reading NOCS pose/coordinate labels."""

import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from experiments.artifacts import sha256, write_json

CATEGORIES = {"bottle": 1, "bowl": 2, "camera": 3, "can": 4, "laptop": 5, "mug": 6}
SOURCE = "https://download.cs.stanford.edu/orion/nocs/real_test.zip"


def crop(frame: Path, instance: str, category: str) -> tuple[Image.Image, dict[str, Any]]:
    metadata = frame.with_name(frame.name + "_meta.txt")
    matches = [line.split() for line in metadata.read_text().splitlines() if line.strip()]
    matches = [parts for parts in matches if len(parts) == 3 and parts[2] == instance]
    if len(matches) != 1 or int(matches[0][1]) != CATEGORIES[category]:
        raise ValueError("Photo metadata does not identify exactly one matching instance/category")
    instance_id = int(matches[0][0])
    with Image.open(frame.with_name(frame.name + "_color.png")) as image:
        rgb = image.convert("RGB")
    with Image.open(frame.with_name(frame.name + "_mask.png")) as image:
        # Upstream OpenCV uses BGR channel 2: PIL's RGB channel 0.
        mask = np.asarray(image.convert("RGB"))[:, :, 0]
    if mask.shape != (rgb.height, rgb.width):
        raise ValueError("RGB and instance mask dimensions differ")
    yy, xx = np.nonzero(mask == instance_id)
    if len(xx) < 100:
        raise ValueError("Matching object has insufficient visible mask pixels")
    box = (
        max(0, int(xx.min()) - 12),
        max(0, int(yy.min()) - 12),
        min(rgb.width, int(xx.max()) + 13),
        min(rgb.height, int(yy.max()) + 13),
    )
    return rgb.crop(box), {
        "instance_id": instance_id,
        "source_instance": instance,
        "bounding_box_xyxy": box,
        "visible_mask_pixels": len(xx),
    }


def prepare(
    frames: list[Path], instance: str, category: str, mesh_sha256: str, output: Path
) -> Path:
    ids = [f"{frame.parent.name}-{frame.name}" for frame in frames]
    if not frames or len(ids) != len(set(ids)):
        raise ValueError("Expected distinct photo frames")
    if category not in CATEGORIES:
        raise ValueError("Unsupported NOCS category")
    if output.exists():
        raise FileExistsError(output)
    prepared = [crop(frame, instance, category) for frame in frames]
    output.mkdir(parents=True)
    assets, images = [], []
    for frame, key, (image, record) in zip(frames, ids, prepared, strict=True):
        for kind, extension in [("color", "png"), ("mask", "png"), ("meta", "txt")]:
            source = frame.with_name(f"{frame.name}_{kind}.{extension}")
            target = output / f"{key}-{kind}.{extension}"
            shutil.copyfile(source, target)
            assets.append(
                {
                    "id": f"{key}-{kind}",
                    "path": target.name,
                    "sha256": sha256(target),
                    "archive_member": f"real_test/{frame.parent.name}/{source.name}",
                }
            )
        path = output / f"{key}-crop.png"
        image.save(path)
        images.append({"id": key, "image": path.name, "sha256": sha256(path), **record})
    write_json(
        output / "manifest.json",
        {
            "schema_version": 1,
            "status": "succeeded",
            "category": category,
            "source_instance": instance,
            "source_url": SOURCE,
            "mesh_sha256": mesh_sha256,
            "registration": "unregistered",
            "scope": "Semantic part evidence only; no photo-to-mesh pose is supplied",
            "runner_sha256": sha256(Path(__file__)),
            "assets": assets,
            "images": images,
        },
    )
    return output

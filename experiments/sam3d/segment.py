"""Generate a SAM 2.1 mask from a manually specified object box."""

import argparse
import importlib
import subprocess
import sys
from pathlib import Path

from experiments.artifacts import sha256, write_json
from experiments.sam3d.prepare import prepare

SOURCE_REVISION = "2b90b9f5ceec907a1c18123530e92e794ad901a4"
MODEL = "facebook/sam2.1-hiera-small"
MODEL_REVISION = "ee5bba1d82bb8749febdf90f45e84b687142ba03"


def segment(image: Path, output: Path, box: list[int], source: Path) -> Path:
    import numpy as np
    import torch
    from huggingface_hub import hf_hub_download
    from PIL import Image

    head = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != SOURCE_REVISION:
        raise ValueError("Unexpected SAM 2 source revision")
    if output.exists():
        raise FileExistsError(output)
    with Image.open(image) as opened:
        if opened.mode != "RGB" or opened.getexif().get(274, 1) != 1:
            raise ValueError("Use an upright RGB input")
        rgb = np.asarray(opened)
        width, height = opened.size
    if len(box) != 4 or not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height):
        raise ValueError("Box must be inside the image: x_min y_min x_max y_max")
    sys.path.insert(0, str(source.resolve()))
    build = importlib.import_module("sam2.build_sam").build_sam2
    predictor_type = importlib.import_module("sam2.sam2_image_predictor").SAM2ImagePredictor
    checkpoint = Path(
        hf_hub_download(
            MODEL,
            "sam2.1_hiera_small.pt",
            revision=MODEL_REVISION,
        )
    )
    predictor = predictor_type(build("configs/sam2.1/sam2.1_hiera_s.yaml", str(checkpoint)))
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        predictor.set_image(rgb)
        masks, scores, _ = predictor.predict(box=np.asarray(box), multimask_output=False)
    output.mkdir(parents=True, exist_ok=False)
    mask_path = output / "mask.png"
    foreground = masks[0] > 0
    Image.fromarray(foreground.astype(np.uint8) * 255).save(mask_path)
    write_json(
        output / "segmentation.json",
        {
            "source_revision": head,
            "model": MODEL,
            "model_revision": MODEL_REVISION,
            "checkpoint_sha256": sha256(checkpoint),
            "image_sha256": sha256(image),
            "box_xyxy": box,
            "box_origin": "manual_annotation",
            "score": float(scores[0]),
            "mask_sha256": sha256(mask_path),
            "visual_review": "required",
        },
    )
    # A visualization for mask inspection; it is never used as model input.
    overlay = rgb.copy()
    overlay[foreground] = (rgb[foreground] * 0.55 + np.array([0, 180, 255]) * 0.45).astype(np.uint8)
    Image.fromarray(overlay).save(output / "mask-overlay.jpg")
    prepare(image, mask_path, output / "input", "SAM 2.1 small with manually annotated box")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--box", type=int, nargs=4, required=True)
    parser.add_argument("--source-dir", type=Path, default=Path("data/tools/sam2"))
    args = parser.parse_args()
    print(segment(args.image, args.output, args.box, args.source_dir))


if __name__ == "__main__":
    main()

"""Render comparison exports with identical cameras and explicit review labels."""

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from experiments.artifacts import read_json
from experiments.orientation.common import checked_file, verify_views
from experiments.orientation.render import render

LABELS = {
    "oa_raw_medoid": "OA raw",
    "oa_support": "OA + support",
    "vlm_support": "Qwen + support",
    "pointso_raw": "PointSO raw",
    "pointso_support": "PointSO + support",
}


def preview(directory: Path, blender: str) -> list[Path]:
    report = read_json(directory / "evaluation.json")
    if report["status"] != "succeeded":
        raise ValueError("Cannot preview an incomplete comparison")
    results = []
    font = ImageFont.load_default(size=19)
    small = ImageFont.load_default(size=15)
    for case in report["cases"]:
        if not case["case"].endswith("-base"):
            continue
        output = directory / f"{case['case']}-comparison.png"
        if output.exists():
            raise FileExistsError(output)
        board = Image.new("RGB", (1600, 780), "white")
        draw = ImageDraw.Draw(board)
        draw.text(
            (18, 14),
            f"{case['case']} | same cameras | all outputs pending review",
            fill="black",
            font=font,
        )
        draw.text(
            (18, 46),
            "Rows: predicted front / three-quarter camera. Labels are not verified semantic truth.",
            fill="black",
            font=small,
        )
        for index, (method, label) in enumerate(LABELS.items()):
            x = index * 320
            item = case["methods"][method]
            draw.text((x + 10, 80), label, fill="black", font=font)
            if item["rotation"] is None:
                draw.text((x + 15, 220), "No selected pose", fill="#9a3412", font=font)
                draw.text(
                    (x + 15, 252), "Needs evidence / disagreement", fill="#9a3412", font=small
                )
                continue
            mesh = checked_file(directory, item["mesh"], item["mesh_sha256"])
            views = directory / f"{case['case']}-{method}-preview"
            if not views.exists():
                render(mesh, views, blender, preset="canonical")
            manifest = verify_views(views)
            if manifest["mesh_sha256"] != item["mesh_sha256"]:
                raise ValueError("Preview was rendered from another export")
            for row, view_id in enumerate((0, 5)):
                view = next(v for v in manifest["views"] if v["id"] == view_id)
                with Image.open(views / view["image"]) as source:
                    tile = Image.new("RGBA", source.size, "white")
                    tile.alpha_composite(source.convert("RGBA"))
                    tile.thumbnail((318, 300))
                    board.paste(tile.convert("RGB"), (x, 115 + 300 * row))
            if "rotation_error_degrees" in item:
                error = item["rotation_error_degrees"]
                draw.text((x + 10, 725), f"Frame error: {error:.1f} deg", fill="black", font=small)
            else:
                draw.text((x + 10, 725), "Front ground truth unavailable", fill="black", font=small)
        board.save(output)
        results.append(output)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--blender", default="blender")
    args = parser.parse_args()
    for path in preview(args.directory, args.blender):
        print(path)

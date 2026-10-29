"""Presentation-only XYZ overlays projected through saved canonical render cameras."""

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from experiments.artifacts import read_json, sha256, write_json
from experiments.orientation.common import verify_views
from experiments.orientation.geometry import rotation


def project(
    points: np.ndarray,
    camera_to_world: np.ndarray,
    camera_position: np.ndarray,
    fov: float,
    size: tuple[int, int],
) -> np.ndarray:
    camera = (points - camera_position) @ camera_to_world
    if not 0 < fov < math.pi or np.any(camera[:, 2] >= -1e-8):
        raise ValueError("Overlay points must be in front of a calibrated camera")
    focal = size[0] / (2 * math.tan(fov / 2))
    return np.column_stack(
        (
            size[0] / 2 + focal * camera[:, 0] / -camera[:, 2],
            size[1] / 2 - focal * camera[:, 1] / -camera[:, 2],
        )
    )


def preview(bundle: Path, output: Path, scope: str) -> Path:
    if scope not in ("full_frame", "up_only", "unresolved"):
        raise ValueError("Unknown pose scope")
    manifest = verify_views(bundle)
    pose_file = bundle / "poses.json"
    poses = read_json(pose_file)["rotations"] if pose_file.exists() else None
    if poses is not None:
        if scope != "unresolved" or len(poses) != len(manifest["views"]):
            raise ValueError("Candidate previews require unresolved scope and matching poses")
    elif manifest.get("preset") not in ("canonical", "axes"):
        raise ValueError("Use direct canonical renders or an explicit candidate-pose bundle")
    output.mkdir(parents=True, exist_ok=False)
    basis = np.asarray(manifest["mesh_to_blender"], dtype=float)
    center = basis @ np.asarray(manifest["center_glb"], dtype=float)
    font = ImageFont.load_default(size=18)
    rows = []
    tiles = []
    for index, view in enumerate(manifest["views"]):
        axes = basis @ rotation(poses[index]).T if poses is not None else basis
        points = np.vstack((center, center + 0.7 * manifest["radius"] * axes.T))
        source = bundle / view["image"]
        with Image.open(source) as original:
            base = Image.new("RGBA", original.size, "white")
            base.alpha_composite(original.convert("RGBA"))
        camera = np.asarray(view["camera_to_world"])
        pixels = project(
            points,
            camera,
            np.asarray(view["camera_position"]),
            manifest["camera_horizontal_fov_radians"],
            base.size,
        )
        draw = ImageDraw.Draw(base)
        start = pixels[0]
        for axis, color in enumerate(("#e03535", "#168b38", "#2463ec")):
            end = pixels[axis + 1]
            delta = end - start
            length = float(np.linalg.norm(delta))
            unresolved = scope == "unresolved" or (scope == "up_only" and axis != 1)
            label = "XYZ"[axis] + (" ?" if unresolved else "")
            if length < 8:
                x, y = end
                draw.ellipse((x - 6, y - 6, x + 6, y + 6), outline=color, width=2)
                toward = float((axes[:, axis] @ camera)[2]) > 0
                if toward:
                    draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color)
                else:
                    draw.line((x - 4, y - 4, x + 4, y + 4), fill=color, width=2)
                    draw.line((x - 4, y + 4, x + 4, y - 4), fill=color, width=2)
                label += " (toward)" if toward else " (away)"
            else:
                unit = delta / length
                if unresolved:
                    for offset in range(0, int(length), 12):
                        a, b = start + unit * offset, start + unit * min(offset + 7, length)
                        draw.line((tuple(a), tuple(b)), fill=color, width=3)
                else:
                    draw.line((tuple(start), tuple(end)), fill=color, width=3)
                side = np.array([-unit[1], unit[0]]) * 5
                draw.polygon(
                    [tuple(end), tuple(end - unit * 12 + side), tuple(end - unit * 12 - side)],
                    fill=color,
                )
            draw.text(
                tuple(end + np.array([8, -12])),
                label,
                fill=color,
                font=font,
                stroke_width=2,
                stroke_fill="white",
            )
        canvas = Image.new("RGB", (base.width, base.height + 60), "white")
        canvas.paste(base.convert("RGB"), (0, 35))
        text = ImageDraw.Draw(canvas)
        text.text(
            (10, 8),
            view["label"].replace("Front", "View").replace("Back", "View"),
            font=font,
            fill="black",
        )
        caption = (
            "XYZ: proposed frame"
            if scope == "full_frame"
            else "Y: proposed up; X/Z: unresolved"
            if scope == "up_only"
            else "XYZ: unresolved"
        )
        text.text((10, base.height + 36), caption, font=font, fill="black")
        target = output / view["image"]
        canvas.save(target)
        tiles.append(canvas)
        rows.append(
            {
                "image": target.name,
                "sha256": sha256(target),
                "source_sha256": sha256(source),
                "axis_pixels_in_source": pixels.tolist(),
            }
        )
    sheet = Image.new(
        "RGB", (tiles[0].width * 3, tiles[0].height * math.ceil(len(tiles) / 3)), "white"
    )
    for i, tile in enumerate(tiles):
        sheet.paste(tile, ((i % 3) * tile.width, (i // 3) * tile.height))
    sheet.save(output / "axes.png")
    write_json(
        output / "manifest.json",
        {
            "scope": scope,
            "source_manifest_sha256": sha256(bundle / "manifest.json"),
            "overlay_version_sha256": sha256(Path(__file__)),
            "candidate_poses_sha256": sha256(pose_file) if poses is not None else None,
            "views": rows,
            "axes_sha256": sha256(output / "axes.png"),
            "presentation_only": True,
            "semantic_correctness_verified": False,
            "origin": "rendered mesh bounding-box center",
            "axis_colors": ["red", "green", "blue"],
        },
    )
    return output / "axes.png"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--scope", choices=("full_frame", "up_only", "unresolved"), required=True)
    args = parser.parse_args()
    print(preview(args.bundle, args.output, args.scope))


if __name__ == "__main__":
    main()

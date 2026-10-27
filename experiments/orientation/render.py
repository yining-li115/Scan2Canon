"""Render a GLB into a checksummed, portable OA V2 input bundle."""

import argparse
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from experiments.artifacts import read_json, sha256, write_json


def contact_sheet(directory: Path, views: list[dict[str, Any]]) -> None:
    sheet = Image.new("RGB", (4 * 260, ((len(views) + 3) // 4) * 282), "white")
    draw = ImageDraw.Draw(sheet)
    for index, view in enumerate(views):
        with Image.open(directory / view["image"]) as image:
            tile = Image.new("RGBA", image.size, "white")
            tile.alpha_composite(image.convert("RGBA"))
            tile.thumbnail((256, 256))
            x, y = index % 4 * 260, index // 4 * 282
            sheet.paste(tile.convert("RGB"), (x, y))
            draw.text((x + 6, y + 260), view.get("label", f"View {view['id']:02d}"), fill="black")
    sheet.save(directory / "contact-sheet.jpg")


def render(
    mesh: Path,
    output: Path,
    blender: str,
    views: int = 12,
    preset: str = "sphere",
    poses: list[Any] | None = None,
    pose_camera: tuple[float, float, float] = (1, -2, 1),
) -> Path:
    mesh, output = mesh.resolve(), output.resolve()
    if not mesh.is_file() or mesh.suffix.lower() != ".glb" or not 3 <= views <= 64:
        raise ValueError("Expected an existing GLB and 3–64 views")
    if preset not in ("sphere", "canonical", "ring", "axes"):
        raise ValueError("Unknown camera preset")
    if poses is not None:
        from experiments.orientation.geometry import rotation

        if not 3 <= len(poses) <= 24:
            raise ValueError("Expected 3–24 candidate poses")
        poses = [rotation(p).tolist() for p in poses]
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    report: dict[str, Any] = {"schema_version": 1, "status": "running"}
    write_json(output / "manifest.json", report)
    try:
        shutil.copyfile(mesh, output / "source.glb")
        script = Path(__file__).with_name("blender_render.py").resolve()
        extra = []
        if poses is not None:
            write_json(output / "poses.json", {"rotations": poses, "camera_direction": pose_camera})
            extra = ["--poses", str(output / "poses.json")]
        with (output / "render.log").open("w") as log:
            subprocess.run(
                [
                    blender,
                    "--background",
                    "--factory-startup",
                    "--python-exit-code",
                    "1",
                    "--python",
                    str(script),
                    "--",
                    str(output / "source.glb"),
                    str(output),
                    "--views",
                    str(views),
                    "--preset",
                    preset,
                    *extra,
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        report.update(read_json(output / "cameras.json"))
        for view in report["views"]:
            view["sha256"] = sha256(output / view["image"])
        contact_sheet(output, report["views"])
        report.update(
            {
                "status": "succeeded",
                "mesh": "source.glb",
                "mesh_sha256": sha256(output / "source.glb"),
                "renderer_sha256": sha256(script),
            }
        )
    except BaseException as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        write_json(output / "manifest.json", report)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mesh", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--blender", default="blender")
    parser.add_argument("--views", type=int, default=12)
    parser.add_argument(
        "--preset", choices=("sphere", "canonical", "ring", "axes"), default="sphere"
    )
    args = parser.parse_args()
    print(render(args.mesh, args.output, args.blender, args.views, args.preset))


if __name__ == "__main__":
    main()

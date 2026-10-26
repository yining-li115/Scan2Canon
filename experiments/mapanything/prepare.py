"""Create a portable, immutable image subset using Scan2Canon's image decoder."""

import argparse
import hashlib
import tempfile
from pathlib import Path

from experiments.artifacts import sha256, write_json
from scan2canon.images import NORMALIZATION_VERSION, SUPPORTED_SUFFIXES, prepare_image


def prepare(source: Path, output: Path, views: int) -> Path:
    if output.exists():
        raise FileExistsError(f"Use a new output directory: {output}")
    files = sorted(
        path
        for path in source.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )
    if not 1 <= views <= len(files):
        raise ValueError(f"Requested {views} views, but found {len(files)} images")
    # Evenly cover the ordered capture, including both endpoints for multiple views.
    indices = [0] if views == 1 else [i * (len(files) - 1) // (views - 1) for i in range(views)]
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".prepare-", dir=output.parent) as temporary:
        staging = Path(temporary) / "capture"
        (staging / "images").mkdir(parents=True)
        frames = []
        for number, index in enumerate(indices):
            original = files[index]
            if original.stat().st_size > 25 * 1024 * 1024:
                raise ValueError(f"Source image exceeds 25 MiB: {original.name}")
            content = original.read_bytes()
            image = prepare_image(content)
            relative = f"images/{number:04d}.jpg"
            target = staging / relative
            target.write_bytes(image.jpeg)
            frames.append(
                {
                    "path": relative,
                    "sha256": sha256(target),
                    "source_name": original.name,
                    "source_index": index,
                    "source_sha256": hashlib.sha256(content).hexdigest(),
                    "width": image.width,
                    "height": image.height,
                }
            )
        if len({frame["sha256"] for frame in frames}) != views:
            raise ValueError("Selected frames include duplicate normalized images")
        write_json(
            staging / "manifest.json",
            {
                "schema_version": 1,
                "source_count": len(files),
                "selection": "uniform indices over lexicographically sorted filenames",
                "normalization": NORMALIZATION_VERSION,
                "frames": frames,
            },
        )
        staging.rename(output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--views", type=int, default=8)
    args = parser.parse_args()
    print(prepare(args.source, args.output, args.views))


if __name__ == "__main__":
    main()

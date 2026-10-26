import hashlib
import os
import warnings
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import UUID, uuid4

from PIL import Image

from scan2canon.domain import Capture, Frame

MAX_IMAGE_BYTES = 25 * 1024 * 1024
MAX_CAPTURE_BYTES = 512 * 1024 * 1024
MAX_PIXELS = 50_000_000


class LocalArtifacts:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        path = (self.root / key).resolve()
        if Path(key).is_absolute() or ".." in Path(key).parts or not path.is_relative_to(self.root):
            raise ValueError("Artifact keys must stay within the artifact root")
        return path

    def read(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def exists(self, key: str) -> bool:
        return self._path(key).is_file()

    def write(self, job_id: UUID, name: str, content: bytes) -> str:
        if not name or Path(name).name != name or name in {".", ".."}:
            raise ValueError("Artifact name must be a filename")
        # Unique paths preserve previous attempts instead of overwriting their evidence.
        key = f"jobs/{job_id}/{uuid4().hex}/{name}"
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp")
        with temporary.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        return key

    def capture(self, capture_id: UUID) -> Capture:
        return Capture.model_validate_json(self.read(f"captures/{capture_id}/manifest.json"))

    def import_capture(self, source: Path) -> Capture:
        if not source.is_dir():
            raise ValueError("Capture source must be a directory of JPEG or PNG images")
        paths = sorted(
            path
            for path in source.iterdir()
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        if not 3 <= len(paths) <= 200:
            raise ValueError("A capture requires 3 to 200 JPEG or PNG images")
        capture_id = uuid4()
        prefix = f"captures/{capture_id}"
        frames: list[Frame] = []
        total_bytes = 0
        with TemporaryDirectory(dir=self.root, prefix=".import-") as directory:
            staging = Path(directory)
            for index, path in enumerate(paths):
                with path.open("rb") as handle:
                    content = handle.read(MAX_IMAGE_BYTES + 1)
                total_bytes += len(content)
                if len(content) > MAX_IMAGE_BYTES or total_bytes > MAX_CAPTURE_BYTES:
                    raise ValueError("Capture exceeds the 25 MiB/image or 512 MiB/capture limit")
                with warnings.catch_warnings():
                    warnings.simplefilter("error", Image.DecompressionBombWarning)
                    with Image.open(BytesIO(content)) as image:
                        if image.format not in {"JPEG", "PNG"}:
                            raise ValueError("Only JPEG and PNG contents are supported")
                        if image.width * image.height > MAX_PIXELS:
                            raise ValueError("Images must not exceed 50 megapixels")
                        image.load()
                        width, height = image.size
                        suffix = "jpg" if image.format == "JPEG" else "png"
                name = f"{index:04d}.{suffix}"
                (staging / name).write_bytes(content)
                frames.append(
                    Frame(
                        key=f"{prefix}/{name}",
                        sha256=hashlib.sha256(content).hexdigest(),
                        width=width,
                        height=height,
                    )
                )
            if len({frame.sha256 for frame in frames}) < 3:
                raise ValueError("At least three distinct image files are required")
            capture = Capture(id=capture_id, frames=tuple(frames))
            (staging / "manifest.json").write_text(capture.model_dump_json(indent=2))
            destination = self._path(prefix)
            destination.parent.mkdir(parents=True, exist_ok=True)
            staging.replace(destination)
        return capture

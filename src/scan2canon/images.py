"""Decode capture images into upright model inputs without changing the originals."""

import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from io import BytesIO

import pillow_heif
from PIL import Image, ImageCms, ImageOps

SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".heic", ".heif"}
MAX_PIXELS = 50_000_000
NORMALIZATION_VERSION = "upright-srgb-jpeg/1"


@dataclass(frozen=True)
class PreparedImage:
    jpeg: bytes
    width: int
    height: int
    source_suffix: str


def _check_size(size: tuple[int, int]) -> None:
    if size[0] * size[1] > MAX_PIXELS:
        raise ValueError("Images must not exceed 50 megapixels")


@contextmanager
def _open_image(content: bytes) -> Iterator[tuple[Image.Image, str]]:
    if pillow_heif.is_supported(content):
        heif = pillow_heif.open_heif(content, convert_hdr_to_8bit=True)
        _check_size(heif.size)
        # libheif applies container transforms. to_pillow resets EXIF orientation,
        # preventing a second rotation when ImageOps handles JPEG/PNG below.
        with heif.to_pillow() as image:
            yield image, "heic" if "heic" in heif.mimetype else "heif"
    else:
        with Image.open(BytesIO(content)) as image:
            if image.format not in {"JPEG", "PNG"}:
                raise ValueError("Only JPEG, PNG, HEIC and HEIF contents are supported")
            _check_size(image.size)
            yield image, "jpg" if image.format == "JPEG" else "png"


def _to_rgb(image: Image.Image) -> Image.Image:
    icc = image.info.get("icc_profile")
    if image.mode in {"RGBA", "LA", "P"}:
        with (
            image.convert("RGBA") as rgba,
            Image.new("RGBA", image.size, "white") as background,
            Image.alpha_composite(background, rgba) as composite,
        ):
            image = composite.convert("RGB")
        try:
            return _apply_profile(image, icc)
        finally:
            image.close()
    return _apply_profile(image, icc)


def _apply_profile(image: Image.Image, icc: bytes | None) -> Image.Image:
    if not icc:
        # Unprofiled RGB is assumed sRGB. NCLX-only HDR color is not tone-mapped here.
        return image.convert("RGB")
    try:
        converted = ImageCms.profileToProfile(
            image,
            BytesIO(icc),
            ImageCms.createProfile("sRGB"),
            outputMode="RGB",
        )
    except ImageCms.PyCMSError as error:
        raise ValueError("Cannot convert the embedded color profile to sRGB") from error
    assert converted is not None
    return converted


def prepare_image(content: bytes) -> PreparedImage:
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        with _open_image(content) as (image, suffix):
            image.load()
            with ImageOps.exif_transpose(image) as upright, _to_rgb(upright) as rgb:
                output = BytesIO()
                # Original metadata remains in the source asset. Derived images
                # contain pixels only, so consumers cannot apply EXIF rotation twice.
                rgb.info.clear()
                rgb.save(output, format="JPEG", quality=95, subsampling=0)
                return PreparedImage(output.getvalue(), rgb.width, rgb.height, suffix)

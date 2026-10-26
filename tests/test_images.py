from io import BytesIO

import pillow_heif
import pytest
from PIL import Image, ImageCms

from scan2canon import images
from scan2canon.images import prepare_image


def encoded_image(format: str, orientation: int = 1) -> bytes:
    image = Image.new("RGB", (64, 32), "red")
    image.paste("blue", (32, 0, 64, 32))
    exif = Image.Exif()
    exif[274] = orientation
    output = BytesIO()
    if format == "HEIF":
        # Encode unrotated pixels with a container transform, as phone captures do.
        pillow_heif.from_bytes(image.mode, image.size, image.tobytes()).save(
            output,
            quality=100,
            exif=exif.tobytes(),
        )
    else:
        image.save(output, format=format, exif=exif)
    image.close()
    return output.getvalue()


@pytest.mark.parametrize("format", ["JPEG", "PNG", "HEIF"])
def test_portrait_orientation_is_applied_once(format: str) -> None:
    prepared = prepare_image(encoded_image(format, orientation=6))
    assert (prepared.width, prepared.height) == (32, 64)
    with Image.open(BytesIO(prepared.jpeg)) as image:
        image.load()
        assert image.format == "JPEG"
        assert image.mode == "RGB"
        assert image.getexif().get(274) is None
        top = image.getpixel((16, 8))
        bottom = image.getpixel((16, 56))
        assert isinstance(top, tuple) and top[0] > 200 and top[2] < 50
        assert isinstance(bottom, tuple) and bottom[2] > 200 and bottom[0] < 50


@pytest.mark.parametrize("format", ["JPEG", "PNG", "HEIF"])
def test_upright_dimensions_are_unchanged(format: str) -> None:
    prepared = prepare_image(encoded_image(format))
    assert (prepared.width, prepared.height) == (64, 32)


@pytest.mark.parametrize("format", ["JPEG", "PNG", "HEIF"])
def test_pixel_limit_applies_to_all_decoders(format: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(images, "MAX_PIXELS", 100)
    with pytest.raises(ValueError, match="megapixels"):
        prepare_image(encoded_image(format))


def test_transparency_is_composited_on_white() -> None:
    output = BytesIO()
    Image.new("RGBA", (32, 32), (255, 0, 0, 0)).save(output, format="PNG")
    prepared = prepare_image(output.getvalue())
    with Image.open(BytesIO(prepared.jpeg)) as image:
        assert image.getpixel((16, 16)) == (255, 255, 255)


def test_embedded_color_profile_is_processed() -> None:
    output = BytesIO()
    profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    Image.new("RGB", (32, 32), "red").save(output, format="PNG", icc_profile=profile)
    prepared = prepare_image(output.getvalue())
    with Image.open(BytesIO(prepared.jpeg)) as image:
        red = image.getpixel((16, 16))
        assert isinstance(red, tuple) and red[0] > 250
        assert "exif" not in image.info


def test_invalid_color_profile_is_not_silently_ignored() -> None:
    output = BytesIO()
    Image.new("RGB", (32, 32), "red").save(output, format="PNG", icc_profile=b"invalid")
    with pytest.raises(ValueError, match="color profile"):
        prepare_image(output.getvalue())

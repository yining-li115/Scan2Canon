"""Only checksum-bound, known-camera diagnostic renders may regain mesh registration."""

import shutil
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from experiments.artifacts import read_json, sha256, write_json  # noqa: E402
from experiments.canonicalization.oa_proposals import load_predictions  # noqa: E402
from experiments.orientation.bind_probe import bind  # noqa: E402
from tests.canonical_fixtures import bundles as bundles  # noqa: E402


def test_binding_preserves_logits_and_rejects_changed_registration(
    bundles: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    views, inference = bundles
    images = tmp_path / "images"
    images.mkdir()
    manifest = read_json(views / "manifest.json")
    rows = []
    for v in manifest["views"]:
        v["camera_position"] = [0, -2, 1]
    write_json(views / "manifest.json", manifest)
    for v in manifest["views"]:
        shutil.copyfile(views / v["image"], images / v["image"])
        rows.append(
            {
                **v,
                "view_id": v["id"],
                "registration": "known_render_camera",
                "source_manifest_sha256": sha256(views / "manifest.json"),
                "mesh_sha256": manifest["mesh_sha256"],
            }
        )
    inputs = {"schema_version": 1, "kind": "oa_image_probe", "images": rows}
    write_json(images / "images.json", inputs)
    probe = read_json(inference / "predictions.json")
    probe.update(kind="oa_image_probe", input_manifest_sha256=sha256(images / "images.json"))
    write_json(inference / "probe.json", probe)
    out = bind(views, images, inference, tmp_path / "bound")
    load_predictions(views, out)
    with np.load(out / "logits.npz") as actual, np.load(inference / "logits.npz") as expected:
        for key in expected.files:
            np.testing.assert_array_equal(actual[key], expected[key])
    with pytest.raises(FileExistsError):
        bind(views, images, inference, out)
    # Even rehashed metadata cannot promote an unregistered photo into a render.
    rows[0]["registration"] = "unregistered_photo"
    write_json(images / "images.json", inputs)
    probe["input_manifest_sha256"] = sha256(images / "images.json")
    write_json(inference / "probe.json", probe)
    with pytest.raises(ValueError, match="exact registered"):
        bind(views, images, inference, tmp_path / "failed")
    assert read_json(tmp_path / "failed/predictions.json")["status"] == "failed"
    assert not (tmp_path / "failed/logits.npz").exists()

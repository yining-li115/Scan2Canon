"""Recover registered renders from a mixed diagnostic without registering photographs."""

import argparse
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from experiments.artifacts import read_json, sha256, write_json
from experiments.orientation.common import config, verify_views
from experiments.orientation.image_probe import verify_images


def bind(views: Path, images: Path, probe: Path, output: Path) -> Path:
    output.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {"schema_version": 1, "status": "running"}
    write_json(output / "predictions.json", report)
    try:
        manifest = verify_views(views)
        verify_images(images)
        inputs = read_json(images / "images.json")
        source = read_json(probe / "probe.json")
        checkpoint = read_json(probe / "checkpoint-manifest.json")
        if (
            source.get("kind") != "oa_image_probe"
            or source.get("status") != "succeeded"
            or source.get("schema_version") != 1
            or source.get("settings") != config()
            or checkpoint.get("settings") != config()
            or source.get("input_manifest_sha256") != sha256(images / "images.json")
            or source.get("logits_sha256") != sha256(probe / "logits.npz")
        ):
            raise ValueError("Incomplete, modified or mismatched diagnostic probe")
        predictions = {p["id"]: p for p in source["predictions"]}
        if (
            len(source["predictions"]) != len(predictions)
            or len(predictions) != len(inputs["images"])
            or set(predictions) != {r["id"] for r in inputs["images"]}
        ):
            raise ValueError("Probe must cover every input exactly once")
        rows, mapping, values = [], [], {}
        with np.load(probe / "logits.npz", allow_pickle=False) as logits:
            for view in manifest["views"]:
                matches = [
                    r
                    for r in inputs["images"]
                    if (
                        r.get("registration") == "known_render_camera"
                        and r.get("source_manifest_sha256") == sha256(views / "manifest.json")
                        and r.get("mesh_sha256") == manifest["mesh_sha256"]
                        and r.get("view_id") == view["id"]
                        and r.get("sha256") == view["sha256"]
                        and r.get("camera_to_world") == view["camera_to_world"]
                        and r.get("camera_position") == view["camera_position"]
                    )
                ]
                if len(matches) != 1:
                    raise ValueError("Each render must have one exact registered probe input")
                identifier = matches[0]["id"]
                value = logits[f"view_{identifier:02d}"]
                if value.shape != (900,) or not np.isfinite(value).all():
                    raise ValueError("Invalid probe logits")
                values[f"view_{view['id']:02d}"] = value
                rows.append({**predictions[identifier], "id": view["id"]})
                mapping.append({"view_id": view["id"], "probe_id": identifier})
        np.savez_compressed(output / "logits.npz", **values)
        shutil.copyfile(probe / "checkpoint-manifest.json", output / "checkpoint-manifest.json")
        report.update(
            status="succeeded",
            settings=config(),
            predictions=rows,
            views_manifest_sha256=sha256(views / "manifest.json"),
            logits_sha256=sha256(output / "logits.npz"),
            source_probe_sha256=sha256(probe / "probe.json"),
            source_images_sha256=sha256(images / "images.json"),
            mapping=mapping,
            binder_sha256=sha256(Path(__file__)),
            scope="Exact saved render association only; no photo registration or new inference",
        )
    except BaseException as error:
        report.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        write_json(output / "predictions.json", report)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("views", "images", "probe", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    print(bind(args.views, args.images, args.probe, args.output))


if __name__ == "__main__":
    main()

"""Run pinned OA V2 single-image inference on every rendered view."""

import argparse
import importlib
import importlib.metadata
import sys
import time
from pathlib import Path
from typing import Any

from experiments.appearance import describe, require_color
from experiments.artifacts import sha256, write_json
from experiments.orientation.common import CONFIG_PATH, config, verify_source, verify_views
from experiments.orientation.download import verify_checkpoint


def predict(
    views: Path, source: Path, checkpoint: Path, output: Path, verify_official: bool = False
) -> dict[str, Any]:
    manifest = verify_views(views)
    images = [(view["id"], views / view["image"]) for view in manifest["views"]]
    return predict_images(images, source, checkpoint, output, verify_official)


def predict_images(
    images: list[tuple[int, Path]],
    source: Path,
    checkpoint: Path,
    output: Path,
    verify_official: bool = False,
) -> dict[str, Any]:
    """Shared forward pass; callers validate their distinct image/mesh contracts."""
    import numpy as np
    import torch
    from PIL import Image

    if not torch.cuda.is_available():
        raise RuntimeError("The upstream OA V2 forward pass requires CUDA")
    sys.path.insert(0, str(source))
    model_class = importlib.import_module("vision_tower").VGGT_OriAny_Ref
    helpers = importlib.import_module("utils.app_utils")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model = model_class(dtype=dtype, out_dim=900, nopretrain=True)
    model.load_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True
    )
    model.eval().to("cuda")
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - started
    raw_outputs: dict[str, Any] = {}
    predictions = []
    started = time.perf_counter()
    with torch.inference_mode():
        for image_id, image_path in images:
            with Image.open(image_path) as image:
                rgba = image.convert("RGBA")
                inputs = helpers.preprocess_images([rgba], mode="pad")
            inputs = inputs.unsqueeze(0).to("cuda")
            captured: list[Any] = []

            def capture(_module: Any, _args: Any, result: Any, sink: list[Any] = captured) -> None:
                sink.append(result.detach().float().cpu().numpy().reshape(-1))

            hook = model.register_forward_hook(capture)
            try:
                answer = helpers.inf_single_batch(model, inputs)
                if verify_official:
                    official = helpers.inf_single_case(model, rgba, None)
                    np.testing.assert_allclose(captured[0], captured[1], rtol=1e-5, atol=1e-5)
                    for key in ("ref_az_pred", "ref_el_pred", "ref_ro_pred", "ref_alpha_pred"):
                        if float(answer[key]) != float(official[key]):
                            raise ValueError(f"Official single-case decoding differs: {key}")
            finally:
                hook.remove()
            logits = captured[0]
            if logits.shape != (900,) or not np.isfinite(logits).all():
                raise ValueError("OA returned invalid orientation logits")
            raw_outputs[f"view_{image_id:02d}"] = logits
            angles = [float(answer[f"ref_{name}_pred"]) for name in ("az", "el", "ro")]
            alpha = int(answer["ref_alpha_pred"])
            if not np.isfinite(angles).all() or alpha not in (0, 1, 2, 4):
                raise ValueError("OA returned invalid angles or symmetry fit")
            predictions.append({"id": image_id, "angles_degrees": angles, "symmetry_fit": alpha})
            print(f"Image {image_id:02d}: angles={angles}, symmetry_fit={alpha}", flush=True)
            write_json(output / "partial-predictions.json", {"predictions": predictions})
    torch.cuda.synchronize()
    np.savez_compressed(output / "logits.npz", **raw_outputs)
    return {
        "predictions": predictions,
        "official_single_case_parity": "passed" if verify_official else "not_checked",
        "metrics": {
            "load_seconds": load_seconds,
            "inference_seconds": time.perf_counter() - started,
            "gpu": torch.cuda.get_device_name(0),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        },
    }


def run(
    views: Path,
    output: Path,
    source: Path,
    checkpoint: Path,
    verify_official: bool = False,
    require_appearance: bool = False,
) -> Path:
    views, output, source, checkpoint = (p.resolve() for p in (views, output, source, checkpoint))
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "settings": config(),
        "dataset_approval": "pending_review",
        "preprocessing": "Official pad-to-518; alpha composited onto white; no rembg call",
        "symmetry_note": "Fit 0 is unresolved azimuth, not proof of physical symmetry",
        "runner_sha256": sha256(Path(__file__)),
        "verify_official": verify_official,
        "environment_lock_sha256": sha256(CONFIG_PATH.with_name("uv.lock")),
    }
    write_json(output / "predictions.json", report)
    try:
        verify_source(source)
        manifest = verify_views(views)
        appearance = describe(views / manifest["mesh"])
        if manifest.get("appearance") is not None and manifest["appearance"] != appearance:
            raise ValueError("Render appearance inventory does not match its mesh")
        if require_appearance:
            require_color(appearance)
        report["appearance"] = appearance
        report["require_appearance"] = require_appearance
        checkpoint_file, checkpoint_manifest = verify_checkpoint(checkpoint)
        report["views_manifest_sha256"] = sha256(views / "manifest.json")
        write_json(output / "checkpoint-manifest.json", checkpoint_manifest)
        report["packages"] = {
            d.metadata["Name"]: d.version for d in importlib.metadata.distributions()
        }
        report.update(predict(views, source, checkpoint_file, output, verify_official))
        report["logits_sha256"] = sha256(output / "logits.npz")
        report["status"] = "succeeded"
    except BaseException as error:
        report.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        write_json(output / "predictions.json", report)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("views", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source-dir", type=Path, default=Path("data/tools/Orient-Anything-V2"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("weights/orientation"))
    parser.add_argument("--verify-official", action="store_true")
    parser.add_argument("--require-appearance", action="store_true")
    args = parser.parse_args()
    print(
        run(
            args.views,
            args.output,
            args.source_dir,
            args.checkpoint_dir,
            args.verify_official,
            args.require_appearance,
        )
    )


if __name__ == "__main__":
    main()

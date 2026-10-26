"""Run pinned OA V2 single-image inference on every rendered view."""

import argparse
import importlib
import importlib.metadata
import sys
import time
from pathlib import Path
from typing import Any

from experiments.artifacts import sha256, write_json
from experiments.orientation.common import CONFIG_PATH, config, verify_source, verify_views
from experiments.orientation.download import verify_checkpoint


def predict(views: Path, source: Path, checkpoint: Path, output: Path) -> dict[str, Any]:
    import numpy as np
    import torch
    from PIL import Image

    manifest = verify_views(views)
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
        for view in manifest["views"]:
            with Image.open(views / view["image"]) as image:
                inputs = helpers.preprocess_images([image.convert("RGBA")], mode="pad")
            inputs = inputs.unsqueeze(0).to("cuda")
            captured: list[Any] = []

            def capture(_module: Any, _args: Any, result: Any, sink: list[Any] = captured) -> None:
                sink.append(result.detach().float().cpu().numpy().reshape(-1))

            hook = model.register_forward_hook(capture)
            try:
                answer = helpers.inf_single_batch(model, inputs)
            finally:
                hook.remove()
            logits = captured[0]
            if logits.shape != (900,) or not np.isfinite(logits).all():
                raise ValueError("OA returned invalid orientation logits")
            raw_outputs[f"view_{view['id']:02d}"] = logits
            angles = [float(answer[f"ref_{name}_pred"]) for name in ("az", "el", "ro")]
            alpha = int(answer["ref_alpha_pred"])
            if not np.isfinite(angles).all() or alpha not in (0, 1, 2, 4):
                raise ValueError("OA returned invalid angles or symmetry fit")
            predictions.append({"id": view["id"], "angles_degrees": angles, "symmetry_fit": alpha})
            print(f"View {view['id']:02d}: angles={angles}, symmetry_fit={alpha}", flush=True)
            write_json(output / "partial-predictions.json", {"predictions": predictions})
    torch.cuda.synchronize()
    np.savez_compressed(output / "logits.npz", **raw_outputs)
    return {
        "predictions": predictions,
        "metrics": {
            "load_seconds": load_seconds,
            "inference_seconds": time.perf_counter() - started,
            "gpu": torch.cuda.get_device_name(0),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        },
    }


def run(views: Path, output: Path, source: Path, checkpoint: Path) -> Path:
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
        "environment_lock_sha256": sha256(CONFIG_PATH.with_name("uv.lock")),
    }
    write_json(output / "predictions.json", report)
    try:
        verify_source(source)
        verify_views(views)
        checkpoint_file, checkpoint_manifest = verify_checkpoint(checkpoint)
        report["views_manifest_sha256"] = sha256(views / "manifest.json")
        write_json(output / "checkpoint-manifest.json", checkpoint_manifest)
        report["packages"] = {
            d.metadata["Name"]: d.version for d in importlib.metadata.distributions()
        }
        report.update(predict(views, source, checkpoint_file, output))
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
    args = parser.parse_args()
    print(run(args.views, args.output, args.source_dir, args.checkpoint_dir))


if __name__ == "__main__":
    main()

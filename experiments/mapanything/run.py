"""Run the pinned image-only baseline and preserve provenance and failures."""

import argparse
import importlib.metadata
import os
import platform
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from experiments.artifacts import read_json, sha256, verified_inputs, write_json
from experiments.mapanything.common import CONFIG_PATH, config, verify_dinov2

INFERENCE_OPTIONS = {
    "memory_efficient_inference": True,
    "minibatch_size": 1,
    "use_amp": True,
    "amp_dtype": "bf16",
    "apply_mask": True,
    "mask_edges": True,
}


def infer(paths: list[Path], checkpoint: Path, output: Path) -> dict[str, Any]:
    os.environ["TORCH_HOME"] = str(checkpoint / "torch")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import numpy as np
    import torch
    import trimesh
    from mapanything.models import MapAnything
    from mapanything.utils.geometry import depthmap_to_world_frame
    from mapanything.utils.image import load_images

    if not torch.cuda.is_available():
        raise RuntimeError("This experiment requires an NVIDIA CUDA GPU; no CPU/MPS fallback")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The pinned baseline requires BF16 support")
    torch.manual_seed(0)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model = MapAnything.from_pretrained(str(checkpoint / "model"), local_files_only=True)
    model = model.eval().to("cuda")
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - started
    views = load_images([str(path) for path in paths], verbose=True)
    if len(views) != len(paths):
        raise ValueError("Upstream image loader skipped one or more manifest frames")
    started = time.perf_counter()
    with torch.inference_mode():
        predictions = model.infer(views, **INFERENCE_OPTIONS)
    torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - started
    if len(predictions) != len(paths):
        raise ValueError("Prediction count does not match the input manifest")

    all_points, all_colors, valid_counts = [], [], []
    for index, prediction in enumerate(predictions):
        depth = prediction["depth_z"][0].squeeze(-1)
        intrinsics = prediction["intrinsics"][0]
        pose = prediction["camera_poses"][0]
        # Match the official demo's pinhole reconstruction from predicted depth.
        points, valid_depth = depthmap_to_world_frame(depth, intrinsics, pose)
        arrays = {
            "depth_z": depth.float().cpu().numpy(),
            "intrinsics": intrinsics.float().cpu().numpy(),
            "camera_to_world": pose.float().cpu().numpy(),
            "world_points": points.float().cpu().numpy(),
            "image_rgb": prediction["img_no_norm"][0].float().cpu().numpy(),
            "confidence": prediction["conf"][0].float().cpu().numpy(),
        }
        if not all(np.isfinite(arrays[key]).all() for key in ("intrinsics", "camera_to_world")):
            raise ValueError(f"Non-finite camera parameters in view {index}")
        mask = (prediction["mask"][0].squeeze(-1).bool() & valid_depth).cpu().numpy()
        mask &= np.isfinite(arrays["world_points"]).all(axis=-1)
        mask &= np.isfinite(arrays["image_rgb"]).all(axis=-1)
        arrays["valid_mask"] = mask
        valid_counts.append(int(mask.sum()))
        np.savez_compressed(output / f"view-{index:04d}.npz", **arrays)
        all_points.append(arrays["world_points"][mask])
        all_colors.append(arrays["image_rgb"][mask])
    if not sum(valid_counts):
        raise ValueError("No valid reconstructed points; inspect per-view predictions")
    points_np = np.concatenate(all_points)
    colors_np = (np.clip(np.concatenate(all_colors), 0, 1) * 255).astype(np.uint8)
    # Bound preview size only; NPZ files retain full per-view predictions.
    indices = np.linspace(0, len(points_np) - 1, min(200_000, len(points_np)), dtype=int)
    cloud = trimesh.points.PointCloud(points_np[indices], colors=colors_np[indices])
    cloud.export(output / "points.ply")
    trimesh.Scene(cloud).export(output / "preview.glb")
    return {
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "torch_cuda": torch.version.cuda,
        "model_load_seconds": load_seconds,
        "inference_seconds": inference_seconds,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "valid_points_per_view": valid_counts,
        "preview_point_count": len(indices),
    }


def run(inputs: Path, checkpoint: Path, output: Path) -> Path:
    inputs, checkpoint, output = inputs.resolve(), checkpoint.resolve(), output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "started_at": datetime.now(UTC).isoformat(),
        "settings": config(),
        "inference_options": INFERENCE_OPTIONS,
        "preprocessing": {"resolution_set": 518, "resize_mode": "fixed_mapping", "norm": "dinov2"},
        "seed": 0,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "runner_sha256": sha256(Path(__file__)),
        "environment_lock_sha256": sha256(CONFIG_PATH.with_name("uv.lock")),
        "packages": {
            dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions()
        },
        "semantic_canonicalization": "not_performed",
        "object_segmentation": "not_performed",
        "metric_scale": "predicted_not_verified",
    }
    write_json(output / "run.json", report)
    try:
        paths = verified_inputs(inputs)
        bundle = read_json(checkpoint / "manifest.json")
        if any(bundle.get(key) != value for key, value in config().items()):
            raise ValueError("Checkpoint bundle does not match the experiment configuration")
        for name in ("config.json", "model.safetensors"):
            if sha256(checkpoint / "model" / name) != bundle["files"][name]:
                raise ValueError(f"Checkpoint checksum mismatch: {name}")
        verify_dinov2(checkpoint / "torch" / "hub" / "facebookresearch_dinov2_main")
        write_json(output / "input-manifest.json", read_json(inputs / "manifest.json"))
        write_json(output / "checkpoint-manifest.json", bundle)
        driver = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=True)
        (output / "nvidia-smi.txt").write_text(driver.stdout)
        report["metrics"] = infer(paths, checkpoint, output)
        report["status"] = "succeeded"
    except BaseException as error:
        report["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        report["finished_at"] = datetime.now(UTC).isoformat()
        write_json(output / "run.json", report)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, help="Prepared capture with manifest.json")
    parser.add_argument("output", type=Path, help="New experiment output directory")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("weights/mapanything"))
    args = parser.parse_args()
    print(run(args.inputs, args.checkpoint_dir, args.output))


if __name__ == "__main__":
    main()

"""Run official SAM 3D inference; export generated assets without claiming scan accuracy."""

import argparse
import importlib
import importlib.metadata
import os
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from experiments.artifacts import read_json, sha256, write_json
from experiments.sam3d.common import CONFIG_PATH, config, verify_source
from experiments.sam3d.download import verify_checkpoint
from experiments.sam3d.prepare import verify_inputs


def infer(inputs: Path, checkpoint: Path, source: Path, output: Path, seed: int) -> dict[str, Any]:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    # The upstream notebook reads CONDA_PREFIX solely to set CUDA_HOME.
    os.environ.setdefault("CONDA_PREFIX", os.environ.get("CUDA_HOME", "/usr/local/cuda"))
    sys.path[:0] = [str(source / "notebook"), str(source)]
    import numpy as np
    import torch
    from PIL import Image

    Inference = importlib.import_module("inference").Inference
    if not torch.cuda.is_available():
        raise RuntimeError("SAM 3D requires an NVIDIA CUDA GPU")
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model = Inference(str(checkpoint / "checkpoints" / "pipeline.yaml"), compile=False)
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - started
    with Image.open(inputs / "image.png") as image, Image.open(inputs / "mask.png") as mask:
        rgb = np.asarray(image.convert("RGB"))
        foreground = np.asarray(mask) == 255
    started = time.perf_counter()
    with torch.inference_mode():
        result = model(rgb, foreground, seed=seed)
    torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - started

    raw_mesh = result["mesh"][0]
    vertices = raw_mesh.vertices.detach().float().cpu().numpy()
    faces = raw_mesh.faces.detach().cpu().numpy()
    if not len(vertices) or not len(faces) or not np.isfinite(vertices).all():
        raise ValueError("Model returned an empty or non-finite mesh")
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError("Model returned invalid mesh face indices")
    np.savez_compressed(output / "raw-mesh.npz", vertices=vertices, faces=faces)
    if result.get("glb") is None or result.get("gs") is None:
        raise ValueError("Official pipeline did not return mesh and Gaussian assets")
    result["glb"].export(str(output / "mesh.glb"))
    result["gs"].save_ply(str(output / "gaussians.ply"))
    pose = {
        name: result[name].detach().float().cpu().numpy()
        for name in ("rotation", "translation", "scale")
        if name in result
    }
    if not all(np.isfinite(array).all() for array in pose.values()):
        raise ValueError("Non-finite predicted object layout")
    np.savez_compressed(output / "predicted-layout.npz", **pose)
    return {
        "gpu": torch.cuda.get_device_name(0),
        "torch_cuda": torch.version.cuda,
        "model_load_seconds": load_seconds,
        "inference_seconds": inference_seconds,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "raw_vertices": len(vertices),
        "raw_faces": len(faces),
        "layout_fields": sorted(pose),
    }


def run(inputs: Path, checkpoint: Path, source: Path, output: Path, seed: int = 42) -> Path:
    inputs, checkpoint, source, output = (
        path.resolve() for path in (inputs, checkpoint, source, output)
    )
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "settings": config(),
        "seed": seed,
        "started_at": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "runner_sha256": sha256(Path(__file__)),
        "environment_lock_sha256": sha256(CONFIG_PATH.with_name("uv.lock")),
        "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
        "geometry_origin": "single_image_generative_prediction",
        "semantic_canonicalization": "not_performed",
        "metric_scale": "not_verified",
        "held_out_view_validation": "not_performed",
        "dataset_approval": "pending_review",
        "inference_options": {
            "compile": False,
            "with_mesh_postprocess": False,
            "with_texture_baking": False,
            "with_layout_postprocess": False,
            "use_vertex_color": True,
        },
    }
    write_json(output / "run.json", report)
    try:
        verify_inputs(inputs)
        verify_source(source)
        checkpoint_manifest = verify_checkpoint(checkpoint)
        write_json(output / "input-manifest.json", read_json(inputs / "manifest.json"))
        write_json(output / "checkpoint-manifest.json", checkpoint_manifest)
        report["metrics"] = infer(inputs, checkpoint, source, output, seed)
        report["artifacts"] = {
            name: sha256(output / name)
            for name in ("mesh.glb", "gaussians.ply", "raw-mesh.npz", "predicted-layout.npz")
        }
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
    parser.add_argument("inputs", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("weights/sam3d"))
    parser.add_argument("--source-dir", type=Path, default=Path("data/tools/sam-3d-objects"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(run(args.inputs, args.checkpoint_dir, args.source_dir, args.output, args.seed))


if __name__ == "__main__":
    main()

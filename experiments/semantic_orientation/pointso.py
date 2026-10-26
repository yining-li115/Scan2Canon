"""Run pinned PointSO with single-instruction official inference and raw votes."""

import argparse
import importlib
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from experiments.artifacts import sha256, write_json
from experiments.semantic_orientation.assets import (
    config,
    verify_case,
    verify_source,
    verify_weights,
)
from experiments.semantic_orientation.contracts import direction_error


def run(cases: list[Path], output: Path, source: Path, weights: Path) -> Path:
    import torch

    source, weights = source.resolve(), weights.resolve()
    verify_source(source)
    weight_manifest = verify_weights(weights, "pointso")
    prepared = [(path, verify_case(path)) for path in cases]
    output.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "method": "pointso-small",
        "settings": config(),
        "weights": weight_manifest,
        "cases": [],
        "approval": "pending_review",
        "runner_sha256": sha256(Path(__file__)),
        "inference": "Official pred_orientation, one instruction per call, 12 votes, FP32",
        "seed": 17,
    }
    write_json(output / "predictions.json", report)
    started = time.perf_counter()
    previous = Path.cwd()
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required for this experiment")
        sys.path.insert(0, str(source))
        upstream: Any = importlib.import_module("serve.pointso")

        upstream.CFG_PATH = str(source / "orientation/cfgs/train/small.yaml")
        upstream.CHECKPOINT_PATH = str(weights / config()["pointso_file"])
        os.chdir(source)
        model = upstream.get_model()
        os.chdir(previous)
        clip_path = Path.home() / ".cache/clip/ViT-B-32.pt"
        report["clip_sha256"] = sha256(clip_path)
        torch.cuda.synchronize()
        report["load_seconds"] = time.perf_counter() - started
        torch.cuda.reset_peak_memory_stats()
        for path, case in prepared:
            points = np.load(path / case["points"], allow_pickle=False)
            if points.shape != (30000, 6) or not np.isfinite(points).all():
                raise ValueError("Expected 30,000 finite XYZRGB points")
            case_start = time.perf_counter()
            entry: dict[str, Any] = {
                "case": path.name,
                "case_sha256": sha256(path / "case.json"),
                "queries": {},
            }
            for axis in ("up", "front"):
                instruction = case["rule"][f"pointso_{axis}"]
                random.seed(17)
                np.random.seed(17)
                torch.manual_seed(17)
                torch.cuda.manual_seed_all(17)
                raw: list[Any] = []

                def capture(
                    _model: Any, _inputs: Any, result: Any, collected: list[Any] = raw
                ) -> None:
                    collected.append(result.detach().float().cpu().numpy())

                hook = model.register_forward_hook(capture)
                try:
                    direction = upstream.pred_orientation(model, points.copy(), [instruction])[0]
                finally:
                    hook.remove()
                if len(raw) != 1 or raw[0].shape != (12, 3):
                    raise ValueError("Unexpected official inference vote layout")
                votes = raw[0]
                entry["queries"][axis] = {
                    "instruction": instruction,
                    "direction": direction,
                    "raw_votes": votes.tolist(),
                    "vote_deviation_degrees": [direction_error(v, direction) for v in votes],
                }
            torch.cuda.synchronize()
            entry["seconds"] = time.perf_counter() - case_start
            report["cases"].append(entry)
            write_json(output / "predictions.json", report)
        report.update(status="succeeded", peak_memory_bytes=torch.cuda.max_memory_allocated())
    except BaseException as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        os.chdir(previous)
        report["elapsed_seconds"] = time.perf_counter() - started
        write_json(output / "predictions.json", report)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("cases", type=Path, nargs="+")
    parser.add_argument("--source", type=Path, default=Path("data/tools/SoFar"))
    parser.add_argument("--weights", type=Path, default=Path("weights/semantic-pointso"))
    args = parser.parse_args()
    print(run(args.cases, args.output, args.source, args.weights))

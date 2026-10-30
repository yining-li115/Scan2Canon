"""CPU resampling diagnostic for OA marginal-fusion top-1; never an acceptance gate."""

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from experiments.artifacts import read_json, sha256, write_json
from experiments.canonicalization.checkpoints import execute_stage
from experiments.canonicalization.oa_proposals import load_predictions
from experiments.orientation import distribution, geometry
from experiments.orientation.distribution import SETTINGS, fuse
from experiments.orientation.geometry import angular_error, mesh_to_oa, rotation, symmetry_error
from scan2canon.worker import worker_lock

SEED = 20261030


def subsets(count: int) -> dict[str, list[int]]:
    """Six seeded complementary halves plus leave-one-out, using positions not view IDs."""
    if count < 8 or count > 24 or count % 2:
        raise ValueError("Use an even number of 8–24 registered views")
    all_ids = set(range(count))
    halves = [tuple(range(0, count, 2))]
    rng = np.random.default_rng(SEED)
    while len(halves) < 6:
        half = set(rng.choice(count, count // 2, replace=False).tolist())
        if 0 not in half:
            half = all_ids - half
        key = tuple(sorted(half))
        if key not in halves:
            halves.append(key)
    result = {"all": list(range(count))}
    for i, half_tuple in enumerate(halves):
        result[f"split-{i}-a"] = list(half_tuple)
        result[f"split-{i}-b"] = sorted(all_ids - set(half_tuple))
    for i in range(count):
        result[f"omit-{i}"] = sorted(all_ids - {i})
    return result


def differences(first: Any, second: Any) -> dict[str, float]:
    a, b = rotation(first), rotation(second)
    return {
        "frame_degrees": angular_error(a, b),
        "up_degrees": symmetry_error(a, b, 0),
        "front_degrees": float(np.degrees(np.arccos(np.clip(np.dot(a[0], b[0]), -1, 1)))),
    }


def statistics(rows: list[dict[str, float]]) -> dict[str, Any]:
    return {
        key: {"median": float(np.median(v)), "max": float(np.max(v))}
        for key in ("frame_degrees", "up_degrees", "front_degrees")
        if (v := [r[key] for r in rows])
    }


def summarize(fits: dict[str, dict[str, Any]]) -> dict[str, Any]:
    tops = {k: v["candidates"][0]["mesh_to_oa"] for k, v in fits.items() if v["candidates"]}
    pairs: list[dict[str, Any]] = []
    for i in range(6):
        a, b = f"split-{i}-a", f"split-{i}-b"
        pairs.append(
            {
                "pair": i,
                "difference": differences(tops[a], tops[b]) if a in tops and b in tops else None,
            }
        )
    omissions: list[dict[str, Any]] = [
        {"subset": name, "difference": differences(top, tops["all"])}
        for name, top in tops.items()
        if name.startswith("omit-") and "all" in tops
    ]
    return {
        "disjoint_halves": pairs,
        "disjoint_statistics": statistics([p["difference"] for p in pairs if p["difference"]]),
        "leave_one_out_vs_all": omissions,
        "leave_one_out_statistics": statistics([r["difference"] for r in omissions]),
        "missing_top1_subsets": [k for k in fits if k not in tops],
        "semantic_accuracy_measured": False,
        "acceptance_threshold_calibrated": False,
        "note": "Descriptive resampling stability, not accuracy. Splits share views across "
        "trials; leave-one-out overlaps the all-view fit. No independent confidence interval.",
    }


def evaluate(views: Path, predictions: Path, output: Path, resume: bool = False) -> Path:
    manifest, inference = load_predictions(views, predictions)
    plan = subsets(len(manifest["views"]))
    request = {
        "schema_version": 1,
        "method": "oa-fused-top1-stability-v1",
        "views_manifest_sha256": sha256(views / "manifest.json"),
        "predictions_sha256": sha256(predictions / "predictions.json"),
        "logits_sha256": sha256(predictions / "logits.npz"),
        "checkpoint_manifest_sha256": sha256(predictions / "checkpoint-manifest.json"),
        "mesh_sha256": manifest["mesh_sha256"],
        "source_view_ids": [v["id"] for v in manifest["views"]],
        "subsets_by_position": plan,
        "seed": SEED,
        "settings": SETTINGS,
        "implementation": {
            p.name: sha256(p)
            for p in (
                Path(__file__),
                Path(distribution.__file__),
                Path(geometry.__file__),
            )
        },
        "development": True,
        "ground_truth_used_for_fitting": False,
    }
    if not resume:
        output.mkdir(parents=True, exist_ok=False)
    with worker_lock(output):
        if resume:
            if read_json(output / "request.json") != request:
                raise ValueError("Stability inputs, protocol or implementation changed")
            state = read_json(output / "state.json")
        else:
            write_json(output / "request.json", request)
            state = {"status": "running", "tools": {}}
        cameras = np.array([v["camera_to_world"] for v in manifest["views"]])
        with np.load(predictions / "logits.npz", allow_pickle=False) as archive:
            logits = np.array([archive[f"view_{v['id']:02d}"] for v in manifest["views"]])
        fits = {}
        try:
            for name, positions in plan.items():

                def compute(path: Path, positions: list[int] = positions) -> Path:
                    path.mkdir()
                    result = fuse(cameras[positions], logits[positions])
                    result["source_view_ids"] = [manifest["views"][i]["id"] for i in positions]
                    write_json(path / "fit.json", result)
                    return path

                path = execute_stage(output, state, name, compute)
                fits[name] = read_json(path / "fit.json")
            summary = summarize(fits)
            # Official decoded per-view outputs stay separate from our fused top-1.
            by_id = {v["id"]: v for v in manifest["views"]}
            decoded = [
                {
                    "id": p["id"],
                    "symmetry_fit": p["symmetry_fit"],
                    "mesh_to_oa": mesh_to_oa(
                        p["angles_degrees"], by_id[p["id"]]["camera_to_world"]
                    ).tolist(),
                }
                for p in inference["predictions"]
            ]
            summary["decoded_views"] = decoded
            summary["decoded_yaw_note"] = (
                "Fit zero does not establish yaw; matrices are representatives."
            )
            summary["decoded_pairwise_up_statistics"] = statistics(
                [
                    differences(a["mesh_to_oa"], b["mesh_to_oa"])
                    for i, a in enumerate(decoded)
                    for b in decoded[i + 1 :]
                ]
            )["up_degrees"]
            write_json(output / "summary.json", summary)
            state["status"] = "succeeded"
        except Exception:
            state["status"] = "failed"
            raise
        finally:
            write_json(output / "state.json", state)
    return output / "summary.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("views", type=Path)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    print(evaluate(args.views, args.predictions, args.output, args.resume))


if __name__ == "__main__":
    main()

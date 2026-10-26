"""Compare semantic predictions without treating agreement as ground truth."""

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from experiments.artifacts import read_json, sha256, write_json
from experiments.orientation.geometry import angular_error, rotation
from experiments.semantic_orientation.assets import verify_case
from experiments.semantic_orientation.contracts import direction_error, frame


def oa_medoid(frames: list[Any]) -> Any:
    matrices = [rotation(f) for f in frames]
    costs = [sum(angular_error(a, b) for b in matrices) for a in matrices]
    return matrices[int(np.argmin(costs))]


def methods(case: dict[str, Any], pointso: dict[str, Any], vlm: dict[str, Any]) -> dict[str, Any]:
    raw = oa_medoid(case["oa_rotations"])
    up = case["support"]["up_in_input_frame"]
    queries = pointso["queries"]
    estimates: dict[str, Any] = {"oa_raw_medoid": raw}
    for name, up_axis, front_axis in (
        ("oa_support", up, raw[2]),
        ("pointso_raw", queries["up"]["direction"], queries["front"]["direction"]),
        ("pointso_support", up, queries["front"]["direction"]),
    ):
        try:
            estimates[name] = frame(up_axis, front_axis)
        except ValueError:
            estimates[name] = None
    candidate = vlm["consistent_candidate_id"]
    estimates["vlm_support"] = next(
        (rotation(c["rotation"]) for c in case["candidates"] if c["id"] == candidate), None
    )
    return estimates


def evaluate(
    cases: list[Path], pointso_run: Path, vlm_run: Path, controls: Path, output: Path
) -> Path:
    pointso = read_json(pointso_run / "predictions.json")
    vlm = read_json(vlm_run / "predictions.json")
    control = read_json(controls / "controls.json")
    if any(r["status"] != "succeeded" for r in (pointso, vlm, control)):
        raise ValueError("Evaluation requires completed runs and controls")
    output.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "cases": [],
        "pairs": [],
        "approval": "pending_review",
        "controls_sha256": sha256(controls / "controls.json"),
        "pointso_sha256": sha256(pointso_run / "predictions.json"),
        "vlm_sha256": sha256(vlm_run / "predictions.json"),
        "note": "Two objects are integration diagnostics, not a generalization benchmark. "
        "Chair has an authored frame; punch has no accepted front ground truth. "
        "No symmetry discount. All support variants share OA-derived up evidence.",
    }
    recovered: dict[str, dict[str, Any]] = {}
    try:
        for path in cases:
            case = verify_case(path)
            p = next(c for c in pointso["cases"] if c["case"] == path.name)
            v = next(c for c in vlm["cases"] if c["case"] == path.name)
            c = next(c for c in control["cases"] if c["name"] == path.name)
            if (
                p["case_sha256"] != sha256(path / "case.json")
                or v["case_sha256"] != sha256(path / "case.json")
                or case["mesh_sha256"] != c["mesh_sha256"]
            ):
                raise ValueError("Mismatched cases, model results, or ground truth")
            estimates = methods(case, p, v)
            recovered[path.name] = estimates
            entry: dict[str, Any] = {
                "case": path.name,
                "methods": {},
                "vlm": v,
                "pointso_axis_separation_degrees": direction_error(
                    p["queries"]["up"]["direction"], p["queries"]["front"]["direction"]
                ),
                "pointso_front_support_up_separation_degrees": direction_error(
                    case["support"]["up_in_input_frame"], p["queries"]["front"]["direction"]
                ),
            }
            scene = trimesh.load_scene(path / case["mesh"], process=False)
            for name, estimate in estimates.items():
                result: dict[str, Any] = {"rotation": None, "decision": "needs_evidence"}
                if estimate is not None:
                    transform = np.eye(4)
                    transform[:3, :3] = estimate
                    transform[:3, 3] = -estimate @ scene.bounds.mean(axis=0)
                    moved = scene.copy()
                    moved.apply_transform(transform)
                    filename = f"{path.name}-{name}.glb"
                    moved.export(output / filename)
                    result.update(
                        rotation=estimate.tolist(),
                        decision="needs_review",
                        mesh=filename,
                        mesh_sha256=sha256(output / filename),
                    )
                    if c["canonical_ground_truth"] is not None:
                        truth = rotation(c["canonical_ground_truth"])
                        result.update(
                            rotation_error_degrees=angular_error(estimate, truth),
                            up_error_degrees=direction_error(estimate[1], truth[1]),
                            front_error_degrees=direction_error(estimate[2], truth[2]),
                        )
                entry["methods"][name] = result
            report["cases"].append(entry)
        for c in control["cases"]:
            if not c["name"].endswith("-rotated") or c["name"] not in recovered:
                continue
            base_name = c["name"].replace("-rotated", "-base")
            if base_name not in recovered:
                continue
            q = rotation(c["applied_rotation"])
            for method, estimate in recovered[c["name"]].items():
                base = recovered[base_name][method]
                item: dict[str, Any] = {"pair": base_name, "method": method, "available": False}
                if estimate is not None and base is not None:
                    returned = estimate @ q
                    item.update(
                        available=True,
                        rotation_difference_degrees=angular_error(returned, base),
                        up_difference_degrees=direction_error(returned[1], base[1]),
                        front_difference_degrees=direction_error(returned[2], base[2]),
                    )
                report["pairs"].append(item)
        report["status"] = "succeeded"
    except BaseException as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        write_json(output / "evaluation.json", report)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("pointso", "vlm", "controls", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("cases", nargs="+", type=Path)
    args = parser.parse_args()
    print(evaluate(args.cases, args.pointso, args.vlm, args.controls, args.output))

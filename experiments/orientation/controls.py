"""Prepare known-pose controls and measure orientation accuracy and equivariance."""

import argparse
from pathlib import Path
from typing import Any

from experiments.artifacts import read_json, sha256, write_json
from experiments.orientation.common import verify_views


def chair_fixture() -> Any:
    """An explicitly authored control: feet down (+Y up), seat opening toward +Z."""
    import trimesh

    scene = trimesh.Scene()
    parts = [("seat", [1.0, 0.16, 1.0], [0, 0.85, 0], [180, 105, 50, 255])]
    parts.append(("back", [1.0, 0.9, 0.14], [0, 1.38, -0.43], [180, 105, 50, 255]))
    for x in (-0.38, 0.38):
        for z in (-0.38, 0.38):
            parts.append((f"leg-{x}-{z}", [0.13, 0.8, 0.13], [x, 0.4, z], [55, 60, 65, 255]))
    for name, extents, center, color in parts:
        mesh = trimesh.creation.box(extents=extents)
        mesh.apply_translation(center)
        mesh.visual.vertex_colors = color
        scene.add_geometry(mesh, node_name=name, geom_name=name)
    scene.apply_translation(-scene.bounds.mean(axis=0))
    return scene


def prepare(mesh: Path, output: Path, blender: str) -> Path:
    import numpy as np
    import trimesh

    from experiments.orientation.geometry import axis_rotation
    from experiments.orientation.render import render

    output.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "source_mesh_sha256": sha256(mesh),
        "note": "Procedural chair is a diagnostic fixture, not a real capture or benchmark asset",
        "cases": [],
    }
    write_json(output / "controls.json", report)
    try:
        q = axis_rotation(0, 41) @ axis_rotation(1, 67) @ axis_rotation(2, -23)
        chair = chair_fixture()
        punch = trimesh.load_scene(mesh, process=False)
        punch.apply_translation(-punch.bounds.mean(axis=0))
        for name, scene, known_rotation, canonical in (
            ("chair-base", chair, np.eye(3), True),
            ("chair-rotated", chair, q, True),
            ("punch-base", punch, np.eye(3), False),
            ("punch-rotated", punch, q, False),
        ):
            moved = scene.copy()
            transform = np.eye(4)
            transform[:3, :3] = known_rotation
            moved.apply_transform(transform)
            path = output / f"{name}.glb"
            moved.export(path)
            render(path, output / name, blender, views=12)
            report["cases"].append(
                {
                    "name": name,
                    "applied_rotation": known_rotation.tolist(),
                    "canonical_ground_truth": known_rotation.T.tolist() if canonical else None,
                    "mesh_sha256": sha256(path),
                    "views_manifest_sha256": sha256(output / name / "manifest.json"),
                }
            )
        report["status"] = "succeeded"
    except BaseException as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        write_json(output / "controls.json", report)
    return output


def evaluate(controls: Path, predictions: Path, output: Path) -> Path:
    import numpy as np

    from experiments.orientation.geometry import OA_TO_CANONICAL, angular_error, mesh_to_oa

    if output.exists():
        raise FileExistsError(output)
    settings = read_json(controls / "controls.json")
    if settings["status"] != "succeeded":
        raise ValueError("Control preparation is incomplete")
    recovered: dict[str, dict[int, Any]] = {}
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "succeeded",
        "controls_sha256": sha256(controls / "controls.json"),
        "cases": [],
        "pairs": [],
        "note": "Accuracy uses the authored chair convention; equivariance is not accuracy",
    }
    for case in settings["cases"]:
        name = case["name"]
        views = controls / name
        manifest = verify_views(views)
        run = predictions / name
        inference = read_json(run / "predictions.json")
        digest = sha256(views / "manifest.json")
        if (
            inference["status"] != "succeeded"
            or digest != case["views_manifest_sha256"]
            or digest != inference["views_manifest_sha256"]
            or sha256(run / "logits.npz") != inference["logits_sha256"]
        ):
            raise ValueError(f"Invalid or mismatched control inference: {name}")
        cameras = {v["id"]: v["camera_to_world"] for v in manifest["views"]}
        estimates = inference["predictions"]
        if len(estimates) != len(cameras) or {p["id"] for p in estimates} != set(cameras):
            raise ValueError("Predictions must cover each control view exactly once")
        recovered[name] = {
            p["id"]: OA_TO_CANONICAL @ mesh_to_oa(p["angles_degrees"], cameras[p["id"]])
            for p in estimates
        }
        entry: dict[str, Any] = {
            "name": name,
            "predictions_sha256": sha256(run / "predictions.json"),
            "official_parity": inference.get("official_single_case_parity", "not_checked"),
        }
        if case["canonical_ground_truth"] is not None:
            truth = np.asarray(case["canonical_ground_truth"])
            errors = [
                {
                    "view": view,
                    "rotation_error_degrees": angular_error(rotation, truth),
                    "up_error_degrees": float(
                        np.degrees(np.arccos(np.clip(rotation[1] @ truth[1], -1, 1)))
                    ),
                }
                for view, rotation in recovered[name].items()
            ]
            entry["per_view"] = errors
            entry["median_rotation_error_degrees"] = float(
                np.median([e["rotation_error_degrees"] for e in errors])
            )
            entry["median_up_error_degrees"] = float(
                np.median([e["up_error_degrees"] for e in errors])
            )
        report["cases"].append(entry)
    for category in ("chair", "punch"):
        base_name, changed_name = f"{category}-base", f"{category}-rotated"
        q = np.asarray(
            next(c for c in settings["cases"] if c["name"] == changed_name)["applied_rotation"]
        )
        pairs = []
        for view, base in recovered[base_name].items():
            returned = recovered[changed_name][view] @ q
            pairs.append(
                {
                    "view": view,
                    "rotation_difference_degrees": angular_error(returned, base),
                    "up_difference_degrees": float(
                        np.degrees(np.arccos(np.clip(returned[1] @ base[1], -1, 1)))
                    ),
                }
            )
        report["pairs"].append(
            {
                "category": category,
                "per_view": pairs,
                "median_rotation_difference_degrees": float(
                    np.median([p["rotation_difference_degrees"] for p in pairs])
                ),
                "median_up_difference_degrees": float(
                    np.median([p["up_difference_degrees"] for p in pairs])
                ),
                "symmetry": "Full-frame comparison; no predicted symmetry discount",
            }
        )
    write_json(output, report)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("mesh", type=Path)
    prep.add_argument("output", type=Path)
    prep.add_argument("--blender", default="blender")
    check = commands.add_parser("evaluate")
    check.add_argument("controls", type=Path)
    check.add_argument("predictions", type=Path)
    check.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        print(prepare(args.mesh, args.output, args.blender))
    else:
        print(evaluate(args.controls, args.predictions, args.output))


if __name__ == "__main__":
    main()

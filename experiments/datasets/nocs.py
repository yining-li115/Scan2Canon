"""Audit source axes and prepare deterministic, shape-only NOCS rotation inputs."""

import argparse
import hashlib
import re
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation  # type: ignore[import-untyped]

from experiments.artifacts import read_json, sha256, write_json
from experiments.orientation.common import checked_file, verify_views
from experiments.orientation.render import render


def samples(catalog: Path) -> list[dict[str, Any]]:
    selected = [s for s in read_json(catalog)["samples"] if s["source"] == "nocs"]
    if not selected or len({s["id"] for s in selected}) != len(selected):
        raise ValueError("Expected distinct NOCS instances")
    if len({s["mesh"]["sha256"] for s in selected}) != len(selected):
        raise ValueError("Repeated mesh content must not count as a distinct instance")
    return selected


def load_shape(sample: dict[str, Any], root: Path) -> Any:
    asset = sample["mesh"]
    path = checked_file(root, asset["path"], asset["sha256"])
    mesh = trimesh.load_mesh(path, process=False)
    if (
        not isinstance(mesh, trimesh.Trimesh)
        or not len(mesh.vertices)
        or not len(mesh.faces)
        or not np.isfinite(mesh.vertices).all()
        or mesh.faces.min() < 0
        or mesh.faces.max() >= len(mesh.vertices)
        or float(np.linalg.norm(mesh.extents)) < 1e-8
    ):
        raise ValueError("Expected nonempty finite triangle geometry")
    # Missing MTL references must not become synthetic appearance evidence.
    mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=[175, 175, 175, 255])
    return mesh


def random_rotations(object_id: str, count: int, seed: int) -> Any:
    digest = hashlib.sha256(f"nocs-rotations-v1:{seed}:{object_id}".encode()).digest()
    rng = np.random.Generator(np.random.PCG64(int.from_bytes(digest[:16], "little")))
    quaternions = rng.normal(size=(count, 4))
    quaternions /= np.linalg.norm(quaternions, axis=1, keepdims=True)
    return Rotation.from_quat(quaternions).as_matrix()


def audit(catalog: Path, root: Path, output: Path, blender: str) -> Path:
    selected = samples(catalog)
    output.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "catalog_sha256": sha256(catalog),
        "appearance": "Uniform gray; source textures unavailable",
        "labels": "Camera positions in source coordinates; not semantic front labels",
        "objects": [],
    }
    write_json(output / "audit.json", report)
    try:
        for sample in selected:
            mesh = load_shape(sample, root)
            path = output / f"{sample['id']}.glb"
            mesh.export(path)
            views = output / sample["id"]
            render(path, views, blender, views=7, preset="axes")
            verify_views(views)
            report["objects"].append(
                {
                    "object_id": sample["id"],
                    "category": sample["category"],
                    "partition": sample["partition"],
                    "source_sha256": sample["mesh"]["sha256"],
                    "mesh_sha256": sha256(path),
                    "views": sample["id"],
                    "views_manifest_sha256": sha256(views / "manifest.json"),
                }
            )
            write_json(output / "audit.json", report)
            print(f"Audited views: {sample['id']}", flush=True)
        report["status"] = "succeeded"
    except BaseException as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        write_json(output / "audit.json", report)
    return output


def prepare(catalog: Path, root: Path, output: Path, count: int = 5, seed: int = 17) -> Path:
    if not 1 <= count <= 20 or seed < 0:
        raise ValueError("Expected 1–20 rotations and a nonnegative seed")
    selected = samples(catalog)
    output.mkdir(parents=True, exist_ok=False)
    (output / "inputs").mkdir()
    references: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "catalog_sha256": sha256(catalog),
        "runner_sha256": sha256(Path(__file__)),
        "numpy_version": np.__version__,
        "trimesh_version": trimesh.__version__,
        "seed": seed,
        "rotations_per_object": count,
        "sampling": "Per-object SHA-256 seeded PCG64; normalized Gaussian xyzw quaternions",
        "reference_scope": "Known source-frame recovery; semantic conventions stored separately",
        "cases": [],
    }
    inputs: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "appearance": "Uniform gray; no texture evidence",
        "cases": [],
    }
    write_json(output / "references.json", references)
    write_json(output / "inputs/index.json", inputs)
    try:
        for sample in selected:
            source = load_shape(sample, root)
            center = source.bounds.mean(axis=0)
            for index, q in enumerate(random_rotations(sample["id"], count, seed)):
                key = f"nocs-input-v1:{seed}:{sample['id']}:{index}"
                case_id = "case-" + hashlib.sha256(key.encode()).hexdigest()[:16]
                path = output / "inputs" / f"{case_id}.glb"
                mesh = source.copy()
                # Bake into vertices: a scene-node rotation would expose the answer.
                mesh.vertices = (source.vertices - center) @ q.T
                mesh.export(path)
                restored = trimesh.load_mesh(path, process=False)
                if not np.array_equal(restored.faces, source.faces) or not np.allclose(
                    restored.vertices @ q + center, source.vertices, atol=1e-6, rtol=0
                ):
                    raise ValueError("GLB export changed geometry beyond float32 tolerance")
                entry = {
                    "case_id": case_id,
                    "partition": sample["partition"],
                    "category": sample["category"],
                    "mesh": path.name,
                    "mesh_sha256": sha256(path),
                }
                inputs["cases"].append(entry)
                references["cases"].append(
                    {
                        **entry,
                        "object_id": sample["id"],
                        "source_mesh_sha256": sample["mesh"]["sha256"],
                        "rotation_index": index,
                        "source_center": center.tolist(),
                        "source_to_input_rotation": q.tolist(),
                        "input_to_source_rotation": q.T.tolist(),
                    }
                )
            write_json(output / "references.json", references)
            write_json(output / "inputs/index.json", inputs)
        references["status"] = inputs["status"] = "succeeded"
    except BaseException as error:
        references.update(status="failed", error=str(error))
        inputs.update(status="failed", error=str(error))
        raise
    finally:
        write_json(output / "references.json", references)
        write_json(output / "inputs/index.json", inputs)
    return output


def render_inputs(prepared: Path, output: Path, blender: str, views: int = 12) -> Path:
    index = read_json(prepared / "inputs/index.json")
    if index.get("status") != "succeeded":
        raise ValueError("Rotation inputs are incomplete")
    if not 3 <= views <= 64:
        raise ValueError("Expected 3–64 views")
    ids = [c["case_id"] for c in index["cases"]]
    if (
        not ids
        or len(ids) != len(set(ids))
        or any(re.fullmatch(r"case-[0-9a-f]{16}", name) is None for name in ids)
    ):
        raise ValueError("Expected unique safe case identifiers")
    output.mkdir(parents=True, exist_ok=True)
    progress_path = output / "index.json"
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "input_index_sha256": sha256(prepared / "inputs/index.json"),
        "views_per_case": views,
        "renderer_sha256": sha256(Path(__file__).parents[1] / "orientation/blender_render.py"),
        "cases": [],
    }
    if progress_path.exists():
        previous = read_json(progress_path)
        if any(
            previous.get(k) != report[k]
            for k in ("input_index_sha256", "views_per_case", "renderer_sha256")
        ):
            raise ValueError("Cannot reuse a render directory with different inputs/settings")
    write_json(progress_path, report)
    try:
        for case in index["cases"]:
            mesh = checked_file(prepared / "inputs", case["mesh"], case["mesh_sha256"])
            target = output / case["case_id"]
            if not target.exists():
                render(mesh, target, blender, views=views)
            existing = verify_views(target)
            if (
                existing["mesh_sha256"] != case["mesh_sha256"]
                or len(existing["views"]) != views
                or existing["preset"] != "sphere"
                or existing["renderer_sha256"] != report["renderer_sha256"]
            ):
                raise ValueError("Existing render does not match this input/settings")
            report["cases"].append(
                {
                    **case,
                    "views_directory": case["case_id"],
                    "views_manifest_sha256": sha256(target / "manifest.json"),
                }
            )
            write_json(progress_path, report)
            print(
                f"Rendered {len(report['cases'])}/{len(index['cases'])}: {case['case_id']}",
                flush=True,
            )
        report["status"] = "succeeded"
    except BaseException as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        write_json(progress_path, report)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("audit", "prepare"):
        command = commands.add_parser(name)
        command.add_argument("catalog", type=Path)
        command.add_argument("output", type=Path)
        command.add_argument("--root", type=Path, default=Path.cwd())
        if name == "audit":
            command.add_argument("--blender", default="blender")
        else:
            command.add_argument("--count", type=int, default=5)
            command.add_argument("--seed", type=int, default=17)
    command = commands.add_parser("render")
    command.add_argument("prepared", type=Path)
    command.add_argument("output", type=Path)
    command.add_argument("--blender", default="blender")
    command.add_argument("--views", type=int, default=12)
    args = parser.parse_args()
    if args.command == "audit":
        print(audit(args.catalog, args.root, args.output, args.blender))
    elif args.command == "prepare":
        print(prepare(args.catalog, args.root, args.output, args.count, args.seed))
    else:
        print(render_inputs(args.prepared, args.output, args.blender, args.views))


if __name__ == "__main__":
    main()

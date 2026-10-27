"""Validate visual-agent decisions, route bounded tools, and export rigid canonical meshes."""

import argparse
from pathlib import Path
from typing import Any, Literal

import numpy as np
import trimesh
from pydantic import BaseModel, ConfigDict, Field

from experiments.artifacts import read_json, sha256, write_json
from experiments.canonicalization.front import prepare as prepare_front
from experiments.canonicalization.semantic import RULES, descriptor, verify
from experiments.canonicalization.upright import prepare as prepare_up
from experiments.orientation.geometry import rotation
from experiments.semantic_orientation.contracts import Selection, direction_error, parse_selection


class Decision(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    schema_version: Literal[1]
    actor: Literal["interactive_visual_agent", "local_vlm"]
    case_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    rules_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    model_calls: int = Field(ge=1, le=2)
    selection: Selection


def chosen(case_path: Path, decision_path: Path) -> tuple[dict[str, Any], Decision, Any]:
    case, images = verify(case_path)
    decision = Decision.model_validate(read_json(decision_path))
    if decision.case_sha256 != sha256(descriptor(case_path)) or decision.rules_sha256 != sha256(
        RULES
    ):
        raise ValueError("Decision belongs to different evidence or conventions")
    selection = parse_selection(decision.selection.model_dump_json(), {key for key, _ in images})
    if selection.decision != "selected":
        return case, decision, None
    if selection.candidate_id not in selection.evidence_view_ids:
        raise ValueError("The selected pose must be cited as visual evidence")
    candidate = next(c for c in case["candidates"] if c["id"] == selection.candidate_id)
    return case, decision, rotation(candidate["rotation"])


def export(mesh_path: Path, matrix: Any, output: Path) -> dict[str, Any]:
    from scipy.spatial import cKDTree  # type: ignore[import-untyped]

    matrix = rotation(matrix)
    original = trimesh.load_scene(mesh_path, process=False)
    mesh = original.to_mesh()
    if not len(mesh.faces) or not np.isfinite(mesh.vertices).all():
        raise ValueError("Expected finite triangle geometry")
    center = mesh.bounds.mean(axis=0)
    transform = np.eye(4)
    transform[:3, :3] = matrix
    transform[:3, 3] = -matrix @ center
    moved_vertices = mesh.vertices @ matrix.T + transform[:3, 3]
    transform[1, 3] -= moved_vertices[:, 1].min()
    moved = original.copy()
    moved.apply_transform(transform)
    if output.exists():
        raise FileExistsError(output)
    moved.export(output)
    loaded = trimesh.load_scene(output, process=False).to_mesh()
    if len(mesh.vertices) != len(loaded.vertices) or len(mesh.faces) != len(loaded.faces):
        raise ValueError("Export changed mesh topology counts")
    restored = (loaded.vertices - transform[:3, 3]) @ matrix
    error = max(
        float(cKDTree(mesh.vertices).query(restored)[0].max()),
        float(cKDTree(restored).query(mesh.vertices)[0].max()),
    )
    tolerance = max(float(mesh.extents.max()) * 2e-6, 1e-7)
    if error > tolerance:
        raise ValueError("Export failed inverse-transform geometry verification")
    return {
        "mesh": output.name,
        "sha256": sha256(output),
        "transform": transform.tolist(),
        "inverse_transform": np.linalg.inv(transform).tolist(),
        "inverse_vertex_error": error,
        "tolerance": tolerance,
        "vertices": len(mesh.vertices),
        "faces": len(mesh.faces),
        "scale": "unchanged",
        "origin": "Bounding-box XZ center; lowest vertex at Y=0",
    }


def next_front(case_path: Path, decision_path: Path, output: Path, blender: str) -> Path:
    case, decision, matrix = chosen(case_path, decision_path)
    if case.get("task", "up") != "up" or matrix is None:
        raise ValueError("Front preparation requires a selected upright candidate")
    rule = read_json(RULES)["categories"][case["category"]]
    if rule["front"] is None:
        raise ValueError("This convention leaves yaw unspecified; no front call is needed")
    prepare_front(case_path / "poses", case["category"], matrix, output, blender)
    path = output / "front.json"
    record = read_json(path)
    record.update(
        up_case_sha256=sha256(descriptor(case_path)),
        up_decision_sha256=sha256(decision_path),
        up_model_calls=decision.model_calls,
    )
    write_json(path, record)
    return output


def finish(
    up_case: Path,
    up_decision: Path,
    output: Path,
    front_case: Path | None = None,
    front_decision: Path | None = None,
) -> Path:
    case, decision, matrix = chosen(up_case, up_decision)
    if case.get("task", "up") != "up":
        raise ValueError("Expected an upright case")
    rule = read_json(RULES)["categories"][case["category"]]
    record: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "category": case["category"],
        "input_mesh_sha256": case["mesh_sha256"],
        "rules_sha256": sha256(RULES),
        "up_case_sha256": sha256(descriptor(up_case)),
        "up_decision_sha256": sha256(up_decision),
        "decisions": [decision.model_dump()],
        "model_calls": decision.model_calls,
        "budget": 4,
        "dataset_approval": "not_granted",
        "semantic_validation": "Visual agent proposal; not a correctness certificate",
        "coordinate_convention": "+Y semantic up, +Z reference front, right-handed",
        "runner_sha256": sha256(Path(__file__)),
    }
    if bool(front_case) != bool(front_decision):
        raise ValueError("Both front evidence and decision are required")
    scope = "up_only" if rule["front"] is None else "up_only_front_pending"
    if front_case is not None and front_decision is not None:
        if rule["front"] is None:
            raise ValueError("This convention leaves yaw unspecified")
        front, choice, selected = chosen(front_case, front_decision)
        if (
            matrix is None
            or front.get("task") != "front"
            or front["mesh_sha256"] != case["mesh_sha256"]
            or front["category"] != case["category"]
            or front["up_case_sha256"] != sha256(descriptor(up_case))
            or front["up_decision_sha256"] != sha256(up_decision)
        ):
            raise ValueError("Front result is not bound to the selected upright evidence")
        record["model_calls"] += choice.model_calls
        if record["model_calls"] > record["budget"]:
            raise ValueError("Model call budget exhausted")
        record["decisions"].append(choice.model_dump())
        record.update(
            front_case_sha256=sha256(descriptor(front_case)),
            front_decision_sha256=sha256(front_decision),
        )
        if selected is not None:
            if direction_error(matrix[1], selected[1]) > 1e-3:
                raise ValueError("Front decision changes the established up direction")
            matrix, scope = selected, "full_frame"
    output.mkdir(parents=True, exist_ok=False)
    try:
        if matrix is None:
            record.update(
                decision="needs_evidence",
                scope="unresolved",
                missing_evidence=decision.selection.missing_evidence,
            )
        else:
            record["artifact"] = export(
                up_case / "poses/source.glb", matrix, output / "canonical.glb"
            )
            record.update(
                decision="automatic_proposal",
                scope=scope,
                yaw="established_by_convention" if scope == "full_frame" else "unspecified",
            )
        record["status"] = "succeeded"
    except BaseException as error:
        record.update(status="failed", error=str(error))
        raise
    finally:
        write_json(output / "canonicalization.json", record)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    prepare_command = sub.add_parser("prepare")
    prepare_command.add_argument("mesh", type=Path)
    prepare_command.add_argument("category")
    prepare_command.add_argument("output", type=Path)
    prepare_command.add_argument("--blender", default="blender")
    for action in ("front", "finish"):
        command = sub.add_parser(action)
        for name in ("case", "decision", "output"):
            command.add_argument(name, type=Path)
        if action == "front":
            command.add_argument("--blender", default="blender")
        else:
            command.add_argument("--front-case", type=Path)
            command.add_argument("--front-decision", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        print(prepare_up(args.mesh, args.category, args.output, args.blender))
    elif args.action == "front":
        print(next_front(args.case, args.decision, args.output, args.blender))
    else:
        print(finish(args.case, args.decision, args.output, args.front_case, args.front_decision))

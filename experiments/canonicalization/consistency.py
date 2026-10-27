"""Evaluate repeated input rotations after every semantic decision has been saved."""

import argparse
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from experiments.artifacts import read_json, sha256, write_json
from experiments.canonicalization.evaluate import errors
from experiments.orientation.common import checked_file
from experiments.orientation.geometry import angular_error, rotation
from experiments.semantic_orientation.contracts import direction_error


def common_frame(predicted: Any, source_to_input: Any) -> Any:
    """p_input = Q p_source; p_canonical = R p_input, hence common rotation R Q."""
    return rotation(rotation(predicted) @ rotation(source_to_input))


def comparisons(rows: list[dict[str, Any]], tolerance: float = 15.0) -> dict[str, Any]:
    if not np.isfinite(tolerance) or not 0 < tolerance < 90:
        raise ValueError("Expected a finite angle threshold between 0 and 90")
    if len(rows) < 2 or len({r["case_id"] for r in rows}) != len(rows):
        raise ValueError("Expected at least two distinct rotation cases")
    proposed = [r for r in rows if r["common_rotation"] is not None]
    pairs = []
    for a, b in combinations(proposed, 2):
        first, second = rotation(a["common_rotation"]), rotation(b["common_rotation"])
        full = a["scope"] == b["scope"] == "full_frame"
        pairs.append(
            {
                "first": a["case_id"],
                "second": b["case_id"],
                "up_degrees": direction_error(first[1], second[1]),
                "frame_degrees": angular_error(first, second) if full else None,
            }
        )
    up = [p["up_degrees"] for p in pairs]
    full_errors = [p["frame_degrees"] for p in pairs if p["frame_degrees"] is not None]
    complete_up = len(proposed) == len(rows)
    complete_full = all(r["scope"] == "full_frame" for r in rows)
    return {
        "cases": len(rows),
        "up_proposals": len(proposed),
        "full_frame_proposals": sum(r["scope"] == "full_frame" for r in rows),
        "decision_scopes_consistent": len({r["scope"] for r in rows}) == 1,
        "up_pair_count": len(up),
        "full_frame_pair_count": len(full_errors),
        "max_up_difference_degrees": max(up) if up else None,
        "max_frame_difference_degrees": max(full_errors) if full_errors else None,
        "all_rotations_up_consistent": bool(complete_up and up and max(up) <= tolerance),
        "all_rotations_frame_consistent": bool(
            complete_full and full_errors and max(full_errors) <= tolerance
        ),
        "measurement": "pairwise" if pairs else "not_measurable",
        "pairs": pairs,
    }


def run(protocol_path: Path, prepared: Path, output: Path) -> Path:
    if output.exists():
        raise FileExistsError(output)
    protocol = read_json(protocol_path)
    cases = protocol["cases"]
    ids = [c["case_id"] for c in cases]
    if not cases or len(ids) != len(set(ids)):
        raise ValueError("Protocol must list distinct cases")
    # Check all decisions exist before reading any answer-bearing reference file.
    saved = []
    for case in cases:
        path = (
            Path(case["prior_result"])
            if case["prior_result"]
            else (protocol_path.parent / case["case_id"] / "canonicalization.json")
        )
        if case["prior_result"] and sha256(path) != case["prior_result_sha256"]:
            raise ValueError("A reused decision changed after protocol freeze")
        result = read_json(path)
        if result["status"] != "succeeded" or result["input_mesh_sha256"] != case["mesh_sha256"]:
            raise ValueError("Incomplete or mismatched canonicalization result")
        if result["category"] != case["category"]:
            raise ValueError("Category differs from the frozen protocol")
        if result["scope"] not in ("full_frame", "up_only", "up_only_front_pending", "unresolved"):
            raise ValueError("Unknown canonicalization scope")
        matrix = None
        if result["scope"] != "unresolved":
            artifact = result["artifact"]
            checked_file(path.parent, artifact["mesh"], artifact["sha256"])
            transform = np.asarray(artifact["transform"], dtype=float)
            if (
                transform.shape != (4, 4)
                or not np.isfinite(transform).all()
                or not np.allclose(transform[3], [0, 0, 0, 1])
            ):
                raise ValueError("Invalid exported transform")
            matrix = rotation(transform[:3, :3])
        elif "artifact" in result:
            raise ValueError("Unresolved case unexpectedly carries a canonical artifact")
        saved.append((case, path, result, matrix))

    for filename, expected in protocol["files"].items():
        if sha256(Path(filename)) != expected:
            raise ValueError(f"Frozen selection tool or convention changed: {filename}")

    known_path, reference_path = prepared / "references.json", prepared / "scoring-references.json"
    known, reference = read_json(known_path), read_json(reference_path)
    if known["status"] != "succeeded" or reference["status"] != "succeeded":
        raise ValueError("Incomplete rotation references")
    if reference["known_rotations_sha256"] != sha256(known_path):
        raise ValueError("Scoring references belong to different rotation inputs")
    rotations = {c["case_id"]: c for c in known["cases"]}
    scoring = {c["case_id"]: c for c in reference["cases"]}
    if len(rotations) != len(known["cases"]) or len(scoring) != len(reference["cases"]):
        raise ValueError("Duplicate reference case IDs")
    rows = []
    for case, path, result, matrix in saved:
        key = case["case_id"]
        applied, target = rotations[key], scoring[key]
        if any(
            r["mesh_sha256"] != case["mesh_sha256"]
            or r["category"] != case["category"]
            or r["partition"] != case["partition"]
            for r in (applied, target)
        ):
            raise ValueError("Reference does not match the frozen case")
        if applied["object_id"] != target["object_id"]:
            raise ValueError("Reference object IDs disagree")
        checked_file(prepared / "inputs", applied["mesh"], applied["mesh_sha256"])
        row: dict[str, Any] = {
            "case_id": key,
            "object_id": applied["object_id"],
            "category": case["category"],
            "rotation_index": applied["rotation_index"],
            "scope": result["scope"],
            "reused_previous_decision": bool(case["prior_result"]),
            "result": str(path),
            "result_sha256": sha256(path),
            "common_rotation": None,
        }
        if matrix is not None:
            row["common_rotation"] = common_frame(
                matrix, applied["source_to_input_rotation"]
            ).tolist()
            row.update(errors(matrix, target))
            if result["scope"] != "full_frame":
                row["frame_error_degrees"] = None
        rows.append(row)
    groups = []
    tolerance = protocol["angle_tolerance_degrees"]
    for obj in sorted({r["object_id"] for r in rows}):
        group = sorted(
            [r for r in rows if r["object_id"] == obj], key=lambda r: r["rotation_index"]
        )
        if len({r["rotation_index"] for r in group}) != len(group):
            raise ValueError("Repeated rotation index within one object")
        up_errors = [r["up_error_degrees"] for r in group if "up_error_degrees" in r]
        frame_errors = [
            r["frame_error_degrees"] for r in group if r.get("frame_error_degrees") is not None
        ]
        groups.append(
            {
                "object_id": obj,
                "category": group[0]["category"],
                **comparisons(group, tolerance),
                "up_within_reference_tolerance": sum(e <= tolerance for e in up_errors),
                "full_frame_within_reference_tolerance": sum(e <= tolerance for e in frame_errors),
                "max_up_reference_error_degrees": max(up_errors) if up_errors else None,
                "max_frame_reference_error_degrees": max(frame_errors) if frame_errors else None,
            }
        )
    write_json(
        output,
        {
            "schema_version": 1,
            "status": "succeeded",
            "protocol_sha256": sha256(protocol_path),
            "known_rotations_sha256": sha256(known_path),
            "scoring_references_sha256": sha256(reference_path),
            "runner_sha256": sha256(Path(__file__)),
            "angle_tolerance_degrees": tolerance,
            "formula": "common_rotation = predicted_input_to_canonical @ source_to_input_rotation",
            "scope": (
                "Rotational equivariance; translation is not compared. Up-only yaw is unspecified."
            ),
            "accuracy_note": (
                "Agreement can be consistently wrong; reference accuracy is reported separately."
            ),
            "history": protocol["history"],
            "objects": groups,
            "cases": rows,
        },
    )
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("protocol", "prepared", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    print(run(args.protocol, args.prepared, args.output))

"""Score saved canonical proposals after inference; never called by the selection tools."""

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from experiments.artifacts import read_json, sha256, write_json
from experiments.orientation.geometry import angular_error, rotation
from experiments.semantic_orientation.contracts import direction_error


def errors(matrix: Any, reference: dict[str, Any]) -> dict[str, float | None]:
    matrix = rotation(matrix)
    full = reference["metric"] == "full_frame"
    return {
        "up_error_degrees": direction_error(matrix[1], reference["up_in_input_frame"]),
        "frame_error_degrees": angular_error(
            matrix, rotation(reference["input_to_project_rotation"])
        )
        if full
        else None,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    up = [r["up_error_degrees"] for r in rows if r.get("up_error_degrees") is not None]
    full = [r["frame_error_degrees"] for r in rows if r.get("frame_error_degrees") is not None]
    return {
        "cases": len(rows),
        "up_proposals": len(up),
        "full_frame_proposals": len(full),
        "up_within_15_degrees": sum(e <= 15 for e in up),
        "full_frame_within_15_degrees": sum(e <= 15 for e in full),
        "median_up_error_degrees": float(np.median(up)) if up else None,
        "median_frame_error_degrees": float(np.median(full)) if full else None,
    }


def run(references: Path, results: Path, output: Path, baseline: bool = False) -> Path:
    if output.exists():
        raise FileExistsError(output)
    refs = {c["case_id"]: c for c in read_json(references)["cases"]}
    rows = []
    if baseline:
        evidence = read_json(results / "evidence.json")
        if evidence["status"] != "succeeded":
            raise ValueError("Incomplete evidence collection")
        for case in evidence["cases"]:
            ref = refs[case["case_id"]]
            if case["mesh_sha256"] != ref["mesh_sha256"]:
                raise ValueError("Reference mesh mismatch")
            for method in ("raw_rotation", "support_rotation"):
                rows.append(
                    {
                        "case_id": case["case_id"],
                        "object_id": ref["object_id"],
                        "category": case["category"],
                        "partition": ref["partition"],
                        "method": method,
                        **errors(case["evidence"][method], ref),
                    }
                )
    else:
        for path in sorted(results.glob("case-*/canonicalization.json")):
            report = read_json(path)
            ref = refs[path.parent.name]
            if report["status"] != "succeeded" or report["input_mesh_sha256"] != ref["mesh_sha256"]:
                raise ValueError("Incomplete proposal or reference mismatch")
            row: dict[str, Any] = {
                "case_id": path.parent.name,
                "object_id": ref["object_id"],
                "category": ref["category"],
                "partition": ref["partition"],
                "scope": report["scope"],
                "decision": report["decision"],
                "proposal_sha256": sha256(path),
            }
            if "artifact" in report:
                artifact = report["artifact"]
                if sha256(path.parent / artifact["mesh"]) != artifact["sha256"]:
                    raise ValueError("Canonical artifact changed")
                matrix = np.asarray(artifact["transform"])[:3, :3]
                row.update(errors(matrix, ref))
                if report["scope"] != "full_frame":
                    row["frame_error_degrees"] = None
            rows.append(row)
    if not rows:
        raise ValueError("No completed cases to evaluate")
    groups = sorted({r.get("method", "visual_agent") for r in rows})
    report = {
        "schema_version": 1,
        "status": "succeeded",
        "references_sha256": sha256(references),
        "reference_scope": "Audited dataset coordinates, not independent human labels",
        "warning": "Repeated rotations are correlated; count distinct objects, not only cases",
        "distinct_objects": len({r["object_id"] for r in rows}),
        "summaries": {
            g: summarize([r for r in rows if r.get("method", "visual_agent") == g]) for g in groups
        },
        "cases": rows,
        "runner_sha256": sha256(Path(__file__)),
    }
    write_json(output, report)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("references", "results", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--baseline", action="store_true")
    args = parser.parse_args()
    print(run(args.references, args.results, args.output, args.baseline))

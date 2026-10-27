"""Combine audited source conventions with known rotations, outside model inputs."""

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from experiments.artifacts import read_json, sha256, write_json
from experiments.orientation.common import checked_file
from experiments.orientation.geometry import rotation


def build(prepared: Path, conventions: Path, output: Path) -> Path:
    if output.exists():
        raise FileExistsError(output)
    if output.resolve().is_relative_to((prepared / "inputs").resolve()):
        raise ValueError("Scoring references must stay outside model inputs")
    known = read_json(prepared / "references.json")
    rules = read_json(conventions)
    if (
        known.get("status") != "succeeded"
        or rules.get("status") != "source_convention_checked"
        or known["catalog_sha256"] != rules["catalog_sha256"]
    ):
        raise ValueError("Incomplete preparation or mismatched source conventions")
    objects = {obj["object_id"]: obj for obj in rules["objects"]}
    if len(objects) != len(rules["objects"]):
        raise ValueError("Repeated reference object")
    cases = []
    for case in known["cases"]:
        obj = objects[case["object_id"]]
        if (
            obj["status"] != "source_convention_checked"
            or obj["source_mesh_sha256"] != case["source_mesh_sha256"]
            or obj["category"] != case["category"]
        ):
            raise ValueError("Reference does not match source mesh/category")
        checked_file(prepared / "inputs", case["mesh"], case["mesh_sha256"])
        rule = rules["categories"][obj["category"]]
        convention = rotation(rule["source_to_project_rotation"])
        applied = rotation(case["source_to_input_rotation"])
        inverse = rotation(case["input_to_source_rotation"])
        if not np.allclose(applied @ inverse, np.eye(3), atol=1e-7):
            raise ValueError("Recorded inverse does not undo the applied rotation")
        if not np.allclose(convention @ rule["source_up"], [0, 1, 0], atol=1e-7):
            raise ValueError("Source up is inconsistent with the convention")
        metric = rule["metric"]
        if metric not in ("up_only", "full_frame"):
            raise ValueError("Unsupported reference metric")
        if metric == "full_frame" and not np.allclose(
            convention @ rule["source_reference_front"], [0, 0, 1], atol=1e-7
        ):
            raise ValueError("Source front is inconsistent with the convention")
        expected = convention @ inverse
        cases.append(
            {
                "case_id": case["case_id"],
                "object_id": case["object_id"],
                "category": case["category"],
                "partition": case["partition"],
                "mesh_sha256": case["mesh_sha256"],
                "metric": metric,
                "up_in_input_frame": expected[1].tolist(),
                "input_to_project_rotation": expected.tolist() if metric == "full_frame" else None,
                "reference_front_in_input_frame": (
                    expected[2].tolist() if metric == "full_frame" else None
                ),
            }
        )
    if not cases or len({c["case_id"] for c in cases}) != len(cases):
        raise ValueError("Expected distinct scoring cases")
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "succeeded",
        "known_rotations_sha256": sha256(prepared / "references.json"),
        "conventions_sha256": sha256(conventions),
        "reference_scope": "Audited dataset coordinates, not measured physical poses",
        "rotation_formula": "input_to_project = source_to_project @ source_to_input.T",
        "dataset_approval": "not_granted",
        "cases": cases,
    }
    write_json(output, report)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared", "conventions", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    print(build(args.prepared, args.conventions, args.output))

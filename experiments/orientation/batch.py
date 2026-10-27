"""Bounded, resumable OA inference over one partition of prepared render bundles."""

import argparse
import re
from pathlib import Path
from typing import Any

from experiments.artifacts import read_json, sha256, write_json
from experiments.orientation import infer
from experiments.orientation.common import checked_file, verify_views


def evidence(views: Path, predictions: Path) -> dict[str, Any]:
    manifest = verify_views(views)
    result = read_json(predictions / "predictions.json")
    if (
        result.get("status") != "succeeded"
        or result["views_manifest_sha256"] != sha256(views / "manifest.json")
        or result["logits_sha256"] != sha256(predictions / "logits.npz")
    ):
        raise ValueError("Incomplete or mismatched orientation evidence")
    ids = [p["id"] for p in result["predictions"]]
    if len(ids) != len(set(ids)) or set(ids) != {v["id"] for v in manifest["views"]}:
        raise ValueError("Predictions must cover every view exactly once")
    return result


def run(
    renders: Path, output: Path, partition: str, source: Path, checkpoint: Path, budget: int = 30
) -> Path:
    if partition not in ("development", "holdout") or not 1 <= budget <= 60:
        raise ValueError("Expected a named partition and a budget of 1–60 model calls")
    index = read_json(renders / "index.json")
    if index.get("status") != "succeeded":
        raise ValueError("Incomplete render index")
    cases = [c for c in index["cases"] if c["partition"] == partition]
    ids = [c["case_id"] for c in cases]
    if (
        not ids
        or len(ids) != len(set(ids))
        or any(not re.fullmatch(r"case-[a-f0-9]{16}", key) for key in ids)
    ):
        raise ValueError("Expected distinct safe case IDs")
    identity = {
        "render_index_sha256": sha256(renders / "index.json"),
        "partition": partition,
        "runner_sha256": sha256(Path(infer.__file__)),
    }
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "batch.json"
    report: dict[str, Any] = {**identity, "schema_version": 1, "cases": {}}
    if report_path.exists():
        report = read_json(report_path)
        if any(report.get(k) != v for k, v in identity.items()):
            raise ValueError("Cannot resume a different input, partition or model runner")
    calls = 0
    report["status"] = "running"
    write_json(report_path, report)
    try:
        for case in cases:
            key = case["case_id"]
            if case["views_directory"] != key:
                raise ValueError("Render directory must match the case ID")
            views = renders / key
            checked_file(renders, f"{key}/manifest.json", case["views_manifest_sha256"])
            manifest = verify_views(views)
            if manifest["mesh_sha256"] != case["mesh_sha256"]:
                raise ValueError("Render mesh does not match index")
            previous = report["cases"].get(key)
            if previous is not None:
                folder = output / previous["directory"]
                if not folder.resolve().is_relative_to(output.resolve()):
                    raise ValueError("Prediction path escapes batch")
                evidence(views, folder)
                if sha256(folder / "predictions.json") != previous["predictions_sha256"]:
                    raise ValueError("Completed predictions changed")
                continue
            if calls >= budget:
                break
            attempt = 1
            while (output / f"{key}-attempt-{attempt:02d}").exists():
                attempt += 1
            folder = output / f"{key}-attempt-{attempt:02d}"
            calls += 1
            infer.run(views, folder, source, checkpoint)
            evidence(views, folder)
            report["cases"][key] = {
                "directory": folder.name,
                "predictions_sha256": sha256(folder / "predictions.json"),
            }
            write_json(report_path, report)
            print(f"Completed {len(report['cases'])}/{len(cases)}: {key}", flush=True)
        report["status"] = "succeeded" if len(report["cases"]) == len(cases) else "budget_exhausted"
    except BaseException as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        report["calls_this_invocation"] = calls
        write_json(report_path, report)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("renders", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--partition", choices=["development", "holdout"], required=True)
    parser.add_argument("--budget", type=int, default=30)
    parser.add_argument("--source-dir", type=Path, default=Path("data/tools/Orient-Anything-V2"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("weights/orientation"))
    args = parser.parse_args()
    print(
        run(
            args.renders,
            args.output,
            args.partition,
            args.source_dir,
            args.checkpoint_dir,
            args.budget,
        )
    )

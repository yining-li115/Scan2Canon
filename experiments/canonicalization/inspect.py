"""Collect mesh evidence for one completed OA partition, without reference access."""

import argparse
from pathlib import Path

from experiments.artifacts import read_json, sha256, write_json
from experiments.canonicalization.evidence import collect
from experiments.orientation.common import checked_file


def run(renders: Path, predictions: Path, output: Path) -> Path:
    if output.exists():
        raise FileExistsError(output)
    batch = read_json(predictions / "batch.json")
    index = read_json(renders / "index.json")
    if batch["render_index_sha256"] != sha256(renders / "index.json"):
        raise ValueError("Batch does not match render index")
    output.mkdir(parents=True)
    result = {
        "schema_version": 1,
        "status": "running",
        "partition": batch["partition"],
        "batch_sha256": sha256(predictions / "batch.json"),
        "cases": [],
        "runner_sha256": sha256(Path(__file__).with_name("evidence.py")),
    }
    try:
        for case in index["cases"]:
            key = case["case_id"]
            if key not in batch["cases"]:
                continue
            entry = batch["cases"][key]
            path = checked_file(
                predictions, f"{entry['directory']}/predictions.json", entry["predictions_sha256"]
            )
            checked_file(renders, f"{key}/manifest.json", case["views_manifest_sha256"])
            facts = collect(renders / key, path.parent)
            result["cases"].append({**case, "evidence": facts})
            write_json(output / "evidence.json", result)
            print(
                f"{key}: up={facts['up_votes']}/12, front={facts['front_cluster']['count']}/12",
                flush=True,
            )
        result["status"] = "succeeded"
    except BaseException:
        result["status"] = "failed"
        raise
    finally:
        write_json(output / "evidence.json", result)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("renders", "predictions", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    print(run(args.renders, args.predictions, args.output))

"""Summarize a job's execution trace without dispatching models or tools."""

import argparse
from pathlib import Path
from typing import Any

from experiments.artifacts import read_json, write_json
from experiments.canonicalization.trace import read_events


def summarize(root: Path) -> dict[str, Any]:
    events, malformed = read_events(root)
    state = read_json(root / "state.json") if (root / "state.json").exists() else {}
    completions = [e for e in events if e["event"] == "run_completed"]
    report_path = (
        root / completions[-1]["result"] / "canonicalization.json" if completions else None
    )
    report = read_json(report_path) if report_path else {}
    observations = {e["artifact"]: e for e in events if e["event"] == "agent_observed"}
    switches = {e["turn"]: e for e in events if e["event"] == "model_switched"}
    return {
        "schema_version": 1,
        "run": str(root.resolve()),
        "status": state.get("status", "not_started"),
        "model_calls": len(state.get("calls", {})),
        "model_switches": list(switches.values()),
        "decision_policy": report.get("decision_policy"),
        "model_seconds": sum(
            e["elapsed_seconds"]
            for e in events
            if e["event"] in ("model_response_saved", "model_failed")
        ),
        "tool_seconds": sum(
            e["elapsed_seconds"]
            for e in events
            if e["event"] in ("stage_completed", "stage_failed")
        ),
        "executed_evidence_actions": [
            {k: e[k] for k in ("turn", "action_kind", "image_ids", "artifact")}
            for e in observations.values()
        ],
        "failures": [
            e
            for e in events
            if e["event"]
            in ("run_failed", "agent_decision_rejected", "model_reply_rejected", "stage_failed")
        ],
        "direction_assessment": report.get("direction_assessment"),
        "direction_checks": report.get("direction_checks"),
        "scope": report.get("scope"),
        "stop_reason": report.get("stop_reason"),
        "pending_capture": report.get("pending_capture"),
        "result": str(report_path.resolve()) if report_path else None,
        "malformed_trace_lines": malformed,
        "dataset_approved": False,
    }


def write_report(root: Path) -> Path:
    summary = summarize(root)
    write_json(root / "trace-summary.json", summary)
    events, _ = read_events(root)
    lines = [
        "# Execution trace",
        "",
        f"Status: {summary['status']}; scope: {summary['scope']}; "
        f"model calls dispatched: {summary['model_calls']}.",
        "",
        "Model proposals and successful exports are not independent semantic validation.",
        "",
        "| Time (UTC) | Event | Details |",
        "| --- | --- | --- |",
    ]
    for e in events:
        details = []
        for key in (
            "call",
            "stage",
            "turn",
            "action_kind",
            "reason",
            "scope",
            "status",
            "error_type",
            "failure_kind",
            "model",
            "from_model",
            "to_model",
        ):
            if key in e:
                details.append(f"{key}={e[key]}")
        if e["event"] == "agent_proposed":
            action = e["action"]
            details += [f"action={action['kind']}", action["reason"], action["expected_evidence"]]
        if e["event"] == "direction_checked":
            details.extend(
                f"{c['hypothesis_id']}@{c['image_id']}: {c['status']} ({c['reason']})"
                for c in e["report"]["checks"]
            )
        if e["event"] == "initial_evidence":
            details.append("original photos=" + ", ".join(e["photo_ids"]))
            details.append(f"mesh color evidence={e['appearance']['has_color_evidence']}")
        if "image_ids" in e:
            details.append("images=" + ", ".join(e["image_ids"]))
        if "elapsed_seconds" in e:
            details.append(f"{e['elapsed_seconds']:.2f}s")
        detail = "; ".join(details).replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {e['time']} | {e['event']} | {detail} |")
    path = root / "trace.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    if not (args.run / "state.json").is_file():
        parser.error("Run has no state.json")
    print(write_report(args.run))


if __name__ == "__main__":
    main()

"""Import a completed evidence-agent job into the local object library, without inference."""

import argparse
import json
from pathlib import Path
from uuid import NAMESPACE_URL, UUID, uuid5

from experiments.artifacts import read_json, sha256
from experiments.canonicalization.agent_contracts import ORIENTATION_REFERENCE, AgentTurn
from experiments.orientation.common import checked_file
from scan2canon.library import Assessment, DirectionClaim, EvidenceRef, IdentityClaim, ObjectRecord
from scan2canon.ports import ObjectLibrary
from scan2canon.runtime import build_runtime


def import_job(
    library: ObjectLibrary, job: Path, name: str, object_id: UUID | None = None
) -> Assessment:
    job = job.resolve()
    state = read_json(job / "state.json")
    if state.get("status") != "succeeded":
        raise ValueError("Only completed jobs can be imported; pending/failed jobs remain on disk")
    files: dict[str, Path] = {}
    reports = []
    for stage, attempts in state["tools"].items():
        record = attempts[-1]
        if record["status"] != "succeeded":
            continue
        for relative, digest in record["files"].items():
            path = checked_file(job, str(Path(record["path"]) / relative), digest)
            files.setdefault(digest, path)
            if stage.startswith("agent-export-") and relative == "canonicalization.json":
                reports.append(path)
    if len(reports) != 1:
        raise ValueError("Expected one completed final agent export")
    report_path = reports[0]
    report = read_json(report_path)
    if (
        report.get("route") not in {"oa-evidence-agent-v3", "oa-evidence-agent-v4"}
        or report.get("stop_reason") != "agent_finished"
    ):
        raise ValueError("Unsupported or nonterminal evidence-agent result")
    request_path = checked_file(job, "request.json", report["request_sha256"])
    call_name = f"agent-{report['agent_turns']}"
    call_state = state["calls"][call_name]
    if call_state["status"] != "responded":
        raise ValueError("Final model response is unavailable")
    call_path = checked_file(job, f"calls/{call_name}.json", call_state["sha256"])
    call = read_json(call_path)
    if report["route"] == "oa-evidence-agent-v4" and (
        report.get("orientation_reference") != ORIENTATION_REFERENCE
        or json.loads(call["request"]["prompt"].rsplit("\n", 1)[-1]).get("orientation_reference")
        != ORIENTATION_REFERENCE
    ):
        raise ValueError("OA reference must match the recorded model request and route")
    turn = AgentTurn.model_validate_json(call["response"]["raw_text"])
    if (
        turn.action.kind != "finish"
        or turn.choice is None
        or report["semantics"] != turn.semantics.model_dump(mode="json")
        or report["direction_assessment"] != turn.directions.model_dump(mode="json")
        or report["decision"] != turn.choice.model_dump(mode="json")
        or report["scope"] != turn.choice.scope
        or report["decision_policy"] != call["request"]["policy"]
    ):
        raise ValueError("Final report does not match its recorded model proposal")
    inputs = state["tools"]["inputs"][-1]
    manifest_path = checked_file(
        job,
        str(Path(inputs["path"]) / "views/manifest.json"),
        inputs["files"]["views/manifest.json"],
    )
    manifest = read_json(manifest_path)
    mesh = checked_file(manifest_path.parent, manifest["mesh"], manifest["mesh_sha256"])
    mesh_digest = sha256(mesh)
    object_id = object_id or uuid5(NAMESPACE_URL, "scan2canon:mesh:" + mesh_digest)
    report_digest = sha256(report_path)
    refs = [
        EvidenceRef(id="source-mesh", location=str(mesh), sha256=mesh_digest, kind="mesh"),
        EvidenceRef(
            id="source-report", location=str(report_path), sha256=report_digest, kind="report"
        ),
        EvidenceRef(
            id="source-request",
            location=str(request_path),
            sha256=sha256(request_path),
            kind="request",
        ),
        EvidenceRef(
            id="source-reply", location=str(call_path), sha256=sha256(call_path), kind="model_reply"
        ),
    ]
    for image in call["request"]["images"]:
        image_path = files.get(image["sha256"])
        if image_path is None:
            raise ValueError("A cited model image is missing from the checked job artifacts")
        refs.append(
            EvidenceRef(
                id=image["id"], location=str(image_path), sha256=image["sha256"], kind="image"
            )
        )
    policy = report["decision_policy"]
    assessment = Assessment(
        id=uuid5(object_id, report_digest),
        object_id=object_id,
        producer=policy["backend"],
        model=policy["model"],
        implementation_sha256=policy["implementation_sha256"],
        report_sha256=report_digest,
        evidence=tuple(refs),
        identities=tuple(
            IdentityClaim(
                functional_identity=h.functional_identity,
                appearance=h.appearance_description,
                parts=tuple(h.functional_parts),
                evidence_ids=tuple(h.evidence_ids),
            )
            for h in turn.semantics.hypotheses
        ),
        directions=tuple(DirectionClaim(**h.model_dump()) for h in turn.directions.hypotheses),
        semantic_status=turn.semantics.status,
        pose_scope=turn.choice.scope,
        selected_up=turn.directions.selected_up,
        selected_front=turn.directions.selected_front,
    )
    library.ingest(
        ObjectRecord(id=object_id, name=name, source_mesh_sha256=mesh_digest), assessment
    )
    return assessment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job", type=Path)
    parser.add_argument("--name", required=True, help="Display name, never a category label")
    parser.add_argument(
        "--object-id", type=UUID, help="Explicit instance association; never inferred by name"
    )
    parser.add_argument("--data-dir", type=Path)
    args = parser.parse_args()
    try:
        result = import_job(
            build_runtime(args.data_dir).library, args.job, args.name, args.object_id
        )
        print(result.model_dump_json(indent=2))
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        parser.exit(1, f"Library import failed: {error}\n")


if __name__ == "__main__":
    main()

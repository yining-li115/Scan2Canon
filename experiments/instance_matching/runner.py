"""Bounded, recoverable pair matching with verified library evidence and proposal writes."""

import json
import shutil
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid5

from experiments.artifacts import read_json, sha256, write_json
from experiments.canonicalization.model_execution import call_policy
from experiments.canonicalization.model_policy import InvalidPolicyReply, ModelPolicy, PolicyRequest
from experiments.canonicalization.trace import event
from experiments.instance_matching.contracts import PROMPT, PairReply
from experiments.orientation.common import checked_file
from scan2canon.library import Assessment, EvidenceRef, MatchProposal
from scan2canon.ports import ObjectLibrary
from scan2canon.worker import worker_lock


def verified_images(assessment: Assessment) -> list[EvidenceRef]:
    for ref in assessment.evidence:
        path = Path(ref.location)
        if not path.is_file() or sha256(path) != ref.sha256:
            raise ValueError("Library evidence is missing or modified; refusing model dispatch")
    images = [e for e in assessment.evidence if e.kind == "image"]
    # Spread source views around the object; keep available original photographs.
    sources = [e for e in images if e.id.startswith("source:")][::2][:6]
    photos = [e for e in images if e.id.startswith("photo:")][:2]
    selected = sources + photos
    if not sources:
        raise ValueError("Pair matching requires source views, not only canonical candidate images")
    return selected


def validate_reply(
    reply: PairReply, left: Assessment, right: Assessment, images: list[tuple[str, Path]]
) -> None:
    available = {label for label, _ in images}
    for side, ids in (("left", reply.left_evidence_ids), ("right", reply.right_evidence_ids)):
        if len(ids) != len(set(ids)) or any(
            i not in available or not i.startswith(side + ":") for i in ids
        ):
            raise ValueError("Cite distinct visible image IDs from the matching instance")
    for pair in reply.correspondences:
        for source, identity in ((left, pair.left_direction_id), (right, pair.right_direction_id)):
            if not any(d.id == identity and d.axis == pair.axis for d in source.directions):
                raise ValueError("Correspondences must reference existing matching-axis directions")


def run(
    library: ObjectLibrary,
    policy: ModelPolicy,
    left_id: UUID,
    right_id: UUID,
    root: Path,
    *,
    resume: bool = False,
    max_calls: int = 2,
) -> Path:
    if not 1 <= max_calls <= 3:
        raise ValueError("Pair matching permits one to three total calls")
    left, right = library.get_assessment(left_id), library.get_assessment(right_id)
    if left.object_id == right.object_id:
        raise ValueError("Choose two different object instances")
    selected = {"left": verified_images(left), "right": verified_images(right)}
    descriptor = {
        "route": "instance-matching-v1",
        "left": left.model_dump(mode="json"),
        "right": right.model_dump(mode="json"),
        "policy": policy.identity,
        "max_calls": max_calls,
        "implementation": {
            p.name: sha256(p)
            for p in (
                Path(__file__),
                Path(__file__).with_name("contracts.py"),
                Path(__file__).parents[1] / "canonicalization/model_execution.py",
            )
        },
    }
    root = root.resolve()
    if root.exists() and not resume:
        raise ValueError("Use a new output directory or explicitly resume this job")
    if not root.exists() and resume:
        raise ValueError("Cannot resume a missing pair job")
    root.mkdir(parents=True, exist_ok=True)
    with worker_lock(root):
        request_path = root / "request.json"
        if request_path.exists():
            if read_json(request_path) != descriptor:
                raise ValueError("Pair inputs, policy, code or budget changed; start a new job")
        else:
            write_json(request_path, descriptor)
        images = []
        directory = root / "images"
        directory.mkdir(exist_ok=True)
        for side, refs in selected.items():
            for index, ref in enumerate(refs):
                name = f"{side}-{index}{Path(ref.location).suffix}"
                target = directory / name
                if not target.exists():
                    temporary = target.with_suffix(target.suffix + ".tmp")
                    shutil.copyfile(ref.location, temporary)
                    temporary.replace(target)
                checked_file(directory, name, ref.sha256)
                images.append((side + ":" + ref.id, target))
        state: dict[str, Any] = (
            read_json(root / "state.json")
            if (root / "state.json").exists()
            else {"calls": {}, "status": "running"}
        )
        if state.get("status") == "succeeded":
            result = checked_file(root, state["result"], state["result_sha256"])
            library.propose_match(MatchProposal.model_validate(read_json(result)["proposal"]))
            return result
        history: list[dict[str, Any]] = []
        for turn in range(1, max_calls + 1):
            context = {
                "remaining_calls": max_calls - turn + 1,
                "left_directions": [
                    d.model_dump(mode="json", exclude={"evidence_ids"}) for d in left.directions
                ],
                "right_directions": [
                    d.model_dump(mode="json", exclude={"evidence_ids"}) for d in right.directions
                ],
                "visible_images": [i for i, _ in images],
                "history": history,
            }
            request = PolicyRequest(
                "instance_matching",
                PROMPT + "\n" + json.dumps(context),
                tuple(images),
                PairReply.model_json_schema(),
            )
            try:
                reply = call_policy(
                    root, state, f"pair-{turn}", request, PairReply, max_calls, policy
                )
                validate_reply(reply, left, right, images)
                result = root / f"proposal-{turn}.json"
                write_json(
                    result,
                    {
                        "reply": reply.model_dump(mode="json"),
                        "request_sha256": sha256(request_path),
                        "call_sha256": sha256(root / f"calls/pair-{turn}.json"),
                        "semantic_correctness_verified": False,
                    },
                )
                proposal = MatchProposal(
                    id=uuid5(
                        NAMESPACE_URL, f"scan2canon:pair:{left_id}:{right_id}:{sha256(result)}"
                    ),
                    left_assessment_id=left_id,
                    right_assessment_id=right_id,
                    identity_relation=reply.identity_relation,
                    shared_frame=reply.shared_frame,
                    correspondences=tuple(reply.correspondences),
                    evidence=EvidenceRef(
                        id="pair-report", location=str(result), sha256=sha256(result), kind="report"
                    ),
                    producer=policy.identity["backend"] + ":" + policy.identity["model"],
                    rationale=reply.rationale,
                )
                library.propose_match(proposal)
            except (InvalidPolicyReply, ValueError) as error:
                if state["calls"].get(f"pair-{turn}", {}).get("status") != "responded":
                    raise  # Unconfirmed dispatches must never cause an automatic retry.
                feedback = (
                    str(error)
                    if not isinstance(error, InvalidPolicyReply)
                    else "Invalid JSON contract; correct required fields and types"
                )
                history.append({"turn": turn, "validation_error": feedback})
                event(root, "pair_rejected", turn=turn, reason=feedback)
                if turn == max_calls:
                    state["status"] = "failed"
                    write_json(root / "state.json", state)
                    raise ValueError("Pair matching exhausted its correction budget") from None
                continue
            # Persist the replayable proposal separately; its evidence remains an immutable report.
            final = root / "result.json"
            write_json(
                final,
                {
                    "proposal": proposal.model_dump(mode="json"),
                    "semantic_correctness_verified": False,
                    "alignment_verified": False,
                    "model_calls": len(state["calls"]),
                },
            )
            state.update(status="succeeded", result=final.name, result_sha256=sha256(final))
            write_json(root / "state.json", state)
            event(
                root,
                "pair_completed",
                identity_relation=reply.identity_relation,
                shared_axes=[c.axis for c in reply.correspondences],
                proposal_id=str(proposal.id),
            )
            return final
    raise RuntimeError("No pair result")

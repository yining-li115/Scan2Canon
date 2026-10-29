"""Bounded open-semantic policy over verified OA V2 pose proposals."""

import json
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

from experiments.appearance import describe, require_color
from experiments.artifacts import read_json, sha256, write_json
from experiments.canonicalization.checkpoints import execute_stage
from experiments.canonicalization.controller import versions
from experiments.canonicalization.model_execution import call_policy
from experiments.canonicalization.model_policy import (
    ModelPolicy,
    PolicyRequest,
)
from experiments.canonicalization.oa_proposals import load_predictions
from experiments.canonicalization.open_contracts import (
    IDENTITY_PROMPT,
    POSE_PROMPT,
    ObjectSemantics,
    OpenPoseChoice,
)
from experiments.canonicalization.open_tools import OpenGeometryTools, image_list, photographs
from experiments.canonicalization.trace import event, failure_kind
from experiments.canonicalization.workflow import export
from experiments.orientation.common import CONFIG_PATH, verify_views
from scan2canon.worker import worker_lock

Reply = TypeVar("Reply", bound=BaseModel)


def evidence_ids(ids: list[str], request: PolicyRequest) -> None:
    if len(ids) != len(set(ids)) or not set(ids) <= {label for label, _ in request.images}:
        raise ValueError("Evidence must cite unique supplied image labels")


def validate_choice(
    choice: OpenPoseChoice,
    semantics: ObjectSemantics,
    candidates: dict[str, Any],
    request: PolicyRequest,
) -> None:
    evidence_ids(choice.evidence_ids, request)
    if (
        any(i.startswith("photo:") for i in choice.evidence_ids)
        and not choice.mesh_correspondence.strip()
    ):
        raise ValueError("Photographic evidence requires visible mesh correspondence")
    if choice.scope == "unresolved":
        return
    rows = {c["id"]: c for c in candidates["candidates"]}
    if choice.candidate_id not in rows:
        raise ValueError("Policy selected an unavailable OA candidate")
    primary = f"candidate:{choice.candidate_id}"
    alternatives = {f"candidate:{i}" for i in rows if i != choice.candidate_id}
    alternatives.add(f"opposite:{choice.candidate_id}")
    if primary not in choice.evidence_ids or not alternatives.intersection(choice.evidence_ids):
        raise ValueError("Pose choice must compare the selected candidate with another view")
    refinement = rows[choice.candidate_id].get("refinement")
    if refinement and f"candidate:{refinement['parent_candidate_id']}" not in choice.evidence_ids:
        raise ValueError("A refined pose must cite its original parent candidate for comparison")
    if choice.scope == "full_frame" and (
        semantics.status != "resolved"
        or semantics.front_mode != "unique"
        or not rows[choice.candidate_id]["front_available"]
    ):
        raise ValueError("A full frame requires a unique semantic front and an OA yaw candidate")


def pose_request(
    semantics: ObjectSemantics,
    candidates: dict[str, Any],
    images: list[tuple[str, Path]],
) -> PolicyRequest:
    # Do not carry references to identity images absent from this request.
    context = semantics.model_dump()
    context["hypotheses"] = [
        {key: value for key, value in h.items() if key not in ("evidence_ids", "reasoning")}
        for h in context["hypotheses"]
    ]
    schema = OpenPoseChoice.model_json_schema()
    schema["properties"]["evidence_ids"]["items"] = {
        "type": "string",
        "enum": [label for label, _ in images],
    }
    schema["properties"]["candidate_id"] = {
        "anyOf": [
            {"type": "integer", "enum": [c["id"] for c in candidates["candidates"]]},
            {"type": "null"},
        ]
    }
    return PolicyRequest(
        task="open_pose",
        prompt=POSE_PROMPT
        + "\n"
        + json.dumps(
            {
                "semantics": context,
                "candidates": [
                    {k: v for k, v in c.items() if k != "rotation"}
                    for c in candidates["candidates"]
                ],
                "omitted_hypotheses": candidates["omitted_hypotheses"],
            }
        ),
        images=tuple(images),
        response_schema=schema,
    )


class OpenController:
    route = "oa-open-semantics-v1"
    allowed_budgets: tuple[int, ...] = (2, 3)

    def __init__(
        self, policy: ModelPolicy, tools: OpenGeometryTools, *, allow_shape_only: bool = True
    ) -> None:
        self.policy, self.tools = policy, tools
        self.allow_shape_only = allow_shape_only

    @property
    def policy_identity(self) -> dict[str, str]:
        return self.policy.identity

    def run(
        self,
        views: Path,
        predictions: Path,
        output: Path,
        observations: Path | None = None,
        resume: bool = False,
        max_calls: int = 3,
    ) -> Path:
        if max_calls not in self.allowed_budgets:
            raise ValueError(f"Allowed model call budgets: {self.allowed_budgets}")
        manifest, _ = load_predictions(views, predictions)
        appearance = describe(views / manifest["mesh"])
        if not self.allow_shape_only:
            require_color(appearance)
        if manifest.get("appearance") is not None and manifest["appearance"] != appearance:
            raise ValueError("Render appearance inventory does not match its mesh")
        photographs(observations, manifest["mesh_sha256"])
        if len(manifest["views"]) > 12:
            raise ValueError("Prepare 3–12 source views for the bounded semantic request")
        if output.exists() and not resume:
            raise FileExistsError(output)
        output.mkdir(parents=True, exist_ok=True)
        with worker_lock(output):
            expected = {
                "route": self.route,
                "allow_shape_only": self.allow_shape_only,
                "appearance": appearance,
                "views": sha256(views / "manifest.json"),
                "predictions": sha256(predictions / "predictions.json"),
                "checkpoint": sha256(predictions / "checkpoint-manifest.json"),
                "observations": sha256(observations / "manifest.json") if observations else None,
                "policy": self.policy_identity,
                "max_calls": max_calls,
                "versions": versions(),
                "oa_config": sha256(CONFIG_PATH),
                "oa_bundle_validation": sha256(CONFIG_PATH.with_name("common.py")),
                "blender": self.tools.blender,
                "proposal_method": self.tools.proposal_method,
            }
            request_file = output / "request.json"
            if request_file.exists() and read_json(request_file) != expected:
                raise ValueError("Resume inputs, policy, budget or versions changed")
            write_json(request_file, expected)
            state_file = output / "state.json"
            state = read_json(state_file) if state_file.exists() else {"tools": {}, "calls": {}}
            event(
                output,
                "run_started",
                route=self.route,
                resume=resume,
                max_calls=max_calls,
                policy=self.policy_identity,
            )
            try:
                result = self._run(views, predictions, observations, output, state, max_calls)
            except Exception as error:
                event(
                    output,
                    "run_failed",
                    error_type=type(error).__name__,
                    failure_kind=failure_kind(error),
                    model_calls=len(state["calls"]),
                )
                state["status"] = "failed"
                write_json(state_file, state)
                raise
            state["status"] = read_json(result / "canonicalization.json")["status"]
            write_json(state_file, state)
            event(
                output,
                "run_completed",
                status=state["status"],
                model_calls=len(state["calls"]),
                result=str(result.relative_to(output)),
            )
            return result

    def _call(
        self,
        root: Path,
        state: dict[str, Any],
        name: str,
        request: PolicyRequest,
        reply_type: type[Reply],
        budget: int,
        policy: ModelPolicy | None = None,
    ) -> Reply:
        return call_policy(root, state, name, request, reply_type, budget, policy or self.policy)

    def _run(
        self,
        views: Path,
        predictions: Path,
        observations: Path | None,
        root: Path,
        state: dict[str, Any],
        budget: int,
    ) -> Path:
        snapshot = execute_stage(
            root,
            state,
            "inputs",
            lambda out: self.tools.snapshot(views, predictions, observations, out),
        )
        manifest = verify_views(snapshot / "views")
        photos = photographs(
            snapshot / "observations" if observations else None, manifest["mesh_sha256"]
        )
        identity_request = PolicyRequest(
            task="open_identity",
            prompt=IDENTITY_PROMPT,
            images=tuple(image_list(snapshot / "views", "source") + photos),
            response_schema=ObjectSemantics.model_json_schema(),
        )
        semantics = self._call(root, state, "identity", identity_request, ObjectSemantics, budget)
        for hypothesis in semantics.hypotheses:
            evidence_ids(hypothesis.evidence_ids, identity_request)
        candidates: dict[str, Any] | None = None
        choice: OpenPoseChoice | None = None
        if semantics.status == "resolved":
            prepared = execute_stage(
                root, state, "candidates", lambda out: self.tools.prepare(snapshot, out)
            )
            candidates = read_json(prepared / "candidates.json")
            primary = image_list(prepared / "primary", "candidate")
            for attempt in range(1, budget):
                images = primary + photos
                if attempt == 2:
                    opposite = execute_stage(
                        root,
                        state,
                        "opposite",
                        lambda out: self.tools.opposite(snapshot, prepared, out),
                    )
                    images = primary + image_list(opposite / "opposite", "opposite") + photos
                request = pose_request(semantics, candidates, images)
                choice = self._call(root, state, f"pose-{attempt}", request, OpenPoseChoice, budget)
                validate_choice(choice, semantics, candidates, request)
                if choice.scope != "unresolved":
                    break
        return execute_stage(
            root,
            state,
            "export",
            lambda out: self._export(snapshot, root, state, semantics, candidates, choice, out),
        )

    def _export(
        self,
        snapshot: Path,
        root: Path,
        state: dict[str, Any],
        semantics: ObjectSemantics,
        candidates: dict[str, Any] | None,
        choice: OpenPoseChoice | None,
        output: Path,
    ) -> Path:
        output.mkdir()
        scope = choice.scope if choice else "unresolved"
        report: dict[str, Any] = {
            "schema_version": 1,
            "route": self.route,
            "status": "needs_evidence" if scope == "unresolved" else "succeeded",
            "scope": scope,
            "semantic_status": "instance_proposal_not_registered_as_category_convention",
            "semantics": semantics.model_dump(),
            "decision": choice.model_dump() if choice else None,
            "candidates": candidates,
            "policy": self.policy_identity,
            "model_calls": len(state["calls"]),
            "request_sha256": sha256(root / "request.json"),
            "yaw_status": "proposed" if scope == "full_frame" else "unspecified",
            "missing_evidence": choice.missing_evidence if choice else semantics.missing_evidence,
            "dataset_approved": False,
            "appearance": read_json(snapshot / "appearance.json"),
            "artifact": None,
        }
        if choice is not None and choice.candidate_id is not None and candidates is not None:
            candidate = next(c for c in candidates["candidates"] if c["id"] == choice.candidate_id)
            manifest = verify_views(snapshot / "views")
            report["artifact"] = export(
                snapshot / "views" / manifest["mesh"],
                candidate["rotation"],
                output / "canonical.glb",
            )
        write_json(output / "canonicalization.json", report)
        return output

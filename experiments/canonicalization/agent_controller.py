"""One visual agent choosing evidence actions under the canonicalization harness."""

import json
from pathlib import Path
from typing import Any

from experiments.artifacts import read_json, sha256, write_json
from experiments.canonicalization.agent_contracts import AGENT_PROMPT, AgentTurn, EvidenceAction
from experiments.canonicalization.agent_tools import (
    action_key,
    execute_evidence,
    read_evidence,
    validate_action,
)
from experiments.canonicalization.checkpoints import execute_stage
from experiments.canonicalization.compute import copy_observations
from experiments.canonicalization.direction_checks import evaluate, render_checks, validate_export
from experiments.canonicalization.model_policy import (
    InvalidPolicyReply,
    ModelPolicy,
    PolicyRateLimited,
    PolicyRequest,
)
from experiments.canonicalization.open_controller import (
    OpenController,
    evidence_ids,
    validate_choice,
)
from experiments.canonicalization.open_tools import OpenGeometryTools, image_list, photographs
from experiments.canonicalization.trace import event
from experiments.orientation.common import checked_file, verify_views
from scan2canon.worker import worker_lock


def agent_request(
    images: list[tuple[str, Path]],
    candidates: dict[str, Any],
    photo_ids: list[str],
    history: list[dict[str, Any]],
    remaining: int,
) -> PolicyRequest:
    labels = [label for label, _ in images]
    ids = [c["id"] for c in candidates["candidates"]]
    schema = AgentTurn.model_json_schema()
    definitions = schema["$defs"]
    # Image labels grow after evidence actions. Keep them in context rather than
    # multiplying large enums inside nested arrays in the provider's grammar.
    # Membership is still enforced locally before any action or export.
    definitions["OpenPoseChoice"]["properties"]["candidate_id"] = {
        "anyOf": [{"type": "integer", "enum": ids}, {"type": "null"}]
    }
    properties = definitions["EvidenceAction"]["properties"]
    properties["candidate_ids"] = {
        "anyOf": [
            {
                "type": "array",
                "items": {"type": "integer", "enum": ids},
                "minItems": 3,
                "maxItems": 3,
            },
            {"type": "null"},
        ]
    }
    context = {
        "remaining_calls_including_this_one": remaining,
        "available_photos": photo_ids,
        "visible_image_ids": labels,
        "mesh_appearance": candidates["appearance"],
        "candidates": [
            {k: v for k, v in c.items() if k != "rotation"} for c in candidates["candidates"]
        ],
        "omitted_hypotheses": candidates["omitted_hypotheses"],
        "history": history,
    }
    return PolicyRequest(
        "canonicalization_agent", AGENT_PROMPT + "\n" + json.dumps(context), tuple(images), schema
    )


class AgentController(OpenController):
    route = "oa-evidence-agent-v3"
    allowed_budgets = (2, 3, 4)

    def __init__(
        self,
        policy: ModelPolicy,
        tools: OpenGeometryTools,
        *,
        allow_shape_only: bool = False,
        fallback_policies: tuple[ModelPolicy, ...] = (),
    ) -> None:
        super().__init__(policy, tools, allow_shape_only=allow_shape_only)
        self.policies = (policy, *fallback_policies)

    @property
    def policy_identity(self) -> dict[str, str]:
        return {
            **self.policy.identity,
            "fallback_policies": json.dumps(
                [p.identity for p in self.policies[1:]], sort_keys=True
            ),
        }

    @staticmethod
    def _reject(
        root: Path, history: list[dict[str, Any]], turn: int, budget: int, error: ValueError
    ) -> None:
        # Only contract failures reach here; artifact/transport/tool failures do not.
        feedback = {
            "turn": turn,
            "validation_error": str(error),
            "instruction": "Correct the rejected response using visible evidence; do not invent "
            "citations or treat rejection as an executed action. Return unresolved if unsupported.",
        }
        event(root, "agent_decision_rejected", turn=turn, reason=str(error))
        history.append(feedback)
        if turn == budget:
            raise error

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
        photo_pool = dict(
            photographs(
                snapshot / "observations" if observations else None, manifest["mesh_sha256"]
            )
        )
        prepared = execute_stage(
            root, state, "candidates", lambda out: self.tools.prepare(snapshot, out)
        )
        candidates = read_json(prepared / "candidates.json")

        def assess(out: Path) -> Path:
            import trimesh

            from experiments.canonicalization.up_refinement import assess_up

            mesh = trimesh.load_scene(
                snapshot / "views" / manifest["mesh"], process=False
            ).to_mesh()
            out.mkdir()
            write_json(out / "assessment.json", assess_up(mesh, candidates))
            return out

        assessment = execute_stage(root, state, "support-assessment", assess)
        checks = read_json(assessment / "assessment.json")
        by_id = {a["candidate_id"]: a for a in checks["assessments"]}
        candidates = {
            **candidates,
            "support_assessment": checks,
            "candidates": [{**c, **by_id.get(c["id"], {})} for c in candidates["candidates"]],
        }
        primary = image_list(prepared / "primary", "candidate")
        initial_photos = list(photo_pool.items())[:2]
        base = image_list(snapshot / "views", "source") + primary + initial_photos
        event(
            root,
            "initial_evidence",
            appearance=candidates["appearance"],
            photo_ids=[i for i, _ in initial_photos],
            original_photos_registered=False,
        )
        extra: dict[str, Path] = {}
        history: list[dict[str, Any]] = []
        completed: list[dict[str, Any]] = []
        previous_checks: list[dict[str, Any]] = []
        policy_index = 0
        for turn in range(1, budget + 1):
            # Keep the primary evidence stable. Only supplementary views are evicted.
            supplementary = [(i, p) for i, p in extra.items() if i not in dict(base)]
            images = base + supplementary[-min(8, 32 - len(base)) :]
            request = agent_request(
                images, candidates, list(photo_pool), history, budget - turn + 1
            )
            try:
                reply = self._call(
                    root,
                    state,
                    f"agent-{turn}",
                    request,
                    AgentTurn,
                    budget,
                    policy=self.policies[policy_index],
                )
            except PolicyRateLimited:
                if turn == budget or policy_index + 1 == len(self.policies):
                    event(
                        root,
                        "model_fallback_stopped",
                        turn=turn,
                        reason="call_budget_exhausted"
                        if turn == budget
                        else "model_list_exhausted",
                    )
                    raise
                event(
                    root,
                    "model_switched",
                    turn=turn,
                    reason="quota_or_rate_limit",
                    from_model=self.policies[policy_index].identity.get("model"),
                    to_model=self.policies[policy_index + 1].identity.get("model"),
                )
                policy_index += 1
                continue
            except InvalidPolicyReply as error:
                self._reject(root, history, turn, budget, error)
                continue
            action = reply.action
            event(
                root,
                "agent_proposed",
                turn=turn,
                action=action.model_dump(),
                identity=[h.functional_identity for h in reply.semantics.hypotheses],
                semantic_status=reply.semantics.status,
                choice=reply.choice.model_dump() if reply.choice else None,
            )
            direction_report: dict[str, Any] | None = None
            try:
                for hypothesis in reply.semantics.hypotheses:
                    evidence_ids(hypothesis.evidence_ids, request)
                validate_action(action, candidates, dict(images), photo_pool, completed)
                try:
                    direction_report = evaluate(
                        reply.directions, candidates, dict(images), previous_checks
                    )
                except ValueError as error:
                    if reply.choice is not None:
                        raise
                    # An evidence action has no export. Invalid interim pose claims
                    # cannot authorize a pose or block an otherwise valid observation.
                    direction_report = {
                        "method": "projected_part_relation_v1",
                        "checks": [],
                        "validation_status": "deferred_invalid_interim_assessment",
                        "validation_error": str(error),
                        "semantic_correctness_verified": False,
                        "symmetry_verified": False,
                    }
                    event(root, "interim_direction_deferred", turn=turn, reason=str(error))
                previous_checks.extend(direction_report["checks"])
                checks_path = root / "direction-checks" / f"agent-{turn}.json"
                checks_path.parent.mkdir(exist_ok=True)
                write_json(checks_path, direction_report)
                render_checks(direction_report, dict(images), checks_path.with_suffix(""))
                event(root, "direction_checked", turn=turn, report=direction_report)
                if reply.choice:
                    validate_choice(reply.choice, reply.semantics, candidates, request)
                    if reply.choice.candidate_id is not None:
                        chosen = next(
                            c
                            for c in candidates["candidates"]
                            if c["id"] == reply.choice.candidate_id
                        )
                        reviewed = {
                            i
                            for a in completed
                            if a["kind"] == "refine_up"
                            for i in a["candidate_ids"]
                        }
                        if (
                            chosen.get("up_refinement_review_required")
                            and "refinement" not in chosen
                            and chosen["id"] not in reviewed
                        ):
                            raise ValueError(
                                "Inspect the available support-plane correction with refine_up "
                                "before choosing this tilted original candidate. You may retain "
                                "the original after comparing it against the tool result."
                            )
                        refined_alternatives = {
                            f"candidate:{c['id']}"
                            for c in candidates["candidates"]
                            if c.get("refinement", {}).get("parent_candidate_id") == chosen["id"]
                        }
                        if (
                            chosen.get("up_refinement_review_required")
                            and "refinement" not in chosen
                            and refined_alternatives
                            and not refined_alternatives.intersection(reply.choice.evidence_ids)
                        ):
                            raise ValueError(
                                "Retaining a tilted original must compare its refined alternative"
                            )
                    validate_export(
                        reply.directions,
                        direction_report,
                        reply.choice.scope,
                        reply.choice.candidate_id,
                    )
            except ValueError as error:
                if direction_report is not None:
                    history.append({"turn": turn, "direction_check_feedback": direction_report})
                self._reject(root, history, turn, budget, error)
                continue
            event(root, "agent_decision_validated", turn=turn, action_kind=action.kind)
            if action.kind == "finish":
                assert reply.choice is not None
                return self._finish(
                    snapshot, root, state, reply, candidates, history, turn, "agent_finished"
                )
            if action.kind == "request_capture":
                submission = state.get("submissions", {}).get(str(turn))
                if submission is None:
                    return self._finish(
                        snapshot,
                        root,
                        state,
                        reply,
                        candidates,
                        history,
                        turn,
                        "awaiting_capture" if turn < budget else "budget_exhausted",
                    )
                directory = root / submission["path"]
                checked_file(directory, "manifest.json", submission["manifest_sha256"])
                supplied = photographs(directory, manifest["mesh_sha256"])
                added = [
                    (f"photo:capture-{turn}-{label.removeprefix('photo:')}", path)
                    for label, path in supplied
                ]
                event(
                    root,
                    "capture_observed",
                    turn=turn,
                    image_ids=[label for label, _ in added],
                    request_id=submission["request_id"],
                )
                photo_pool.update(added)
                extra.update(added)
                history.append(
                    {
                        "turn": turn,
                        "action": action.model_dump(),
                        "result": {
                            "submitted_images": [label for label, _ in added],
                            "manifest_sha256": submission["manifest_sha256"],
                            "registration": "unregistered",
                        },
                    }
                )
                continue
            if turn == budget:
                return self._finish(
                    snapshot, root, state, reply, candidates, history, turn, "budget_exhausted"
                )

            def perform(
                out: Path,
                action: EvidenceAction = action,
                images: list[tuple[str, Path]] = images,
                turn: int = turn,
                candidates: dict[str, Any] = candidates,
            ) -> Path:
                return execute_evidence(
                    action,
                    snapshot,
                    candidates,
                    dict(images),
                    photo_pool,
                    out,
                    self.tools.blender,
                    turn,
                )

            output = execute_stage(root, state, f"evidence-{turn}", perform)
            observation, additions = read_evidence(output)
            new_candidates = observation.get("new_candidates", [])
            if new_candidates:
                # Replay merges the checksummed tool result, never model-supplied rotations.
                existing = {c["id"] for c in candidates["candidates"]}
                incoming = {c["id"] for c in new_candidates}
                if existing & incoming or len(incoming) != len(new_candidates):
                    raise ValueError("Refinement candidate IDs must be new and unique")
                if len(new_candidates) > 3 or action.kind != "refine_up":
                    raise ValueError("Only the bounded refinement tool can add candidates")
                candidates = {**candidates, "candidates": candidates["candidates"] + new_candidates}
            event(
                root,
                "agent_observed",
                turn=turn,
                action_kind=action.kind,
                image_ids=[label for label, _ in additions],
                result=observation["result"],
                artifact=str(output.relative_to(root)),
            )
            # Re-insertion makes an explicitly retrieved image recent again.
            for label, path in additions:
                extra.pop(label, None)
                extra[label] = path
            completed.append(action_key(action))
            history.append(
                {
                    "turn": turn,
                    **observation,
                    "direction_assessment": reply.directions.model_dump(),
                    "direction_checks": direction_report,
                    "evidence_sha256": sha256(output / "evidence.json"),
                }
            )
        raise RuntimeError("Agent exhausted its loop without recording a terminal state")

    def _finish(
        self,
        snapshot: Path,
        root: Path,
        state: dict[str, Any],
        reply: AgentTurn,
        candidates: dict[str, Any],
        history: list[dict[str, Any]],
        turn: int,
        reason: str,
    ) -> Path:
        pending = None
        if reason == "awaiting_capture":
            pending = {
                "turn": turn,
                "question": reply.action.question,
                "expected_evidence": reply.action.expected_evidence,
                "mesh_sha256": candidates["mesh_sha256"],
                "request_id": sha256(root / "calls" / f"agent-{turn}.json"),
            }

        def finish(output: Path) -> Path:
            self._export(snapshot, root, state, reply.semantics, candidates, reply.choice, output)
            report = read_json(output / "canonicalization.json")
            report.update(
                direction_assessment=reply.directions.model_dump(),
                direction_checks=read_json(root / "direction-checks" / f"agent-{turn}.json"),
                stop_reason=reason,
                decision_policy=read_json(root / "calls" / f"agent-{turn}.json")["request"][
                    "policy"
                ],
                action_history=history,
                last_action=reply.action.model_dump(),
                pending_capture=pending,
                agent_turns=turn,
            )
            if reply.choice is None:
                report["missing_evidence"] = reply.action.question or reply.action.expected_evidence
            write_json(output / "canonicalization.json", report)
            return output

        result = execute_stage(root, state, f"agent-export-{turn}", finish)
        state["pending_capture"] = pending
        event(
            root,
            "agent_stopped",
            turn=turn,
            reason=reason,
            scope=reply.choice.scope if reply.choice else "unresolved",
            pending_capture=pending,
        )
        return result


def submit_capture(root: Path, bundle: Path) -> None:
    """Append verified user evidence to a pending task without changing its call budget."""
    with worker_lock(root):
        state = read_json(root / "state.json")
        pending = state.get("pending_capture")
        if not pending or state.get("status") == "succeeded":
            raise ValueError("This job is not waiting for a user capture")
        if len(state["calls"]) != pending["turn"]:
            raise ValueError("The agent has already continued beyond this capture request")
        request = read_json(root / "request.json")
        if request["route"] != AgentController.route or len(state["calls"]) >= request["max_calls"]:
            raise ValueError("No remaining agent budget for submitted evidence")
        turn = str(pending["turn"])
        checked_file(root / "calls", f"agent-{turn}.json", pending["request_id"])
        photographs(bundle, pending["mesh_sha256"])
        manifest = read_json(bundle / "manifest.json")
        submissions = state.setdefault("submissions", {})
        if turn in submissions:
            if submissions[turn]["manifest_sha256"] != sha256(bundle / "manifest.json"):
                raise ValueError("This capture request already has a different submission")
            return

        def snapshot(out: Path) -> Path:
            copy_observations(
                bundle,
                out,
                {"mesh_sha256": pending["mesh_sha256"], "category": manifest.get("category")},
            )
            return out

        output = execute_stage(root, state, f"capture-{turn}", snapshot)
        # Also catches a crash between a successful copy and recording its submission.
        if sha256(output / "manifest.json") != sha256(bundle / "manifest.json"):
            raise ValueError("Submitted capture changed during recovery")
        submissions[turn] = {
            "path": str(output.relative_to(root)),
            "manifest_sha256": sha256(output / "manifest.json"),
            "request_id": pending["request_id"],
        }
        state["status"] = "needs_evidence"
        write_json(root / "state.json", state)
        event(
            root,
            "capture_submitted",
            turn=pending["turn"],
            request_id=pending["request_id"],
            manifest_sha256=submissions[turn]["manifest_sha256"],
        )

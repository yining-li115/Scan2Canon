"""Journal one bounded model dispatch and validate the reply before tool execution."""

from pathlib import Path
from time import perf_counter
from typing import Any

from pydantic import BaseModel, ValidationError

from experiments.artifacts import read_json, sha256, write_json
from experiments.canonicalization.model_policy import (
    InvalidPolicyReply,
    ModelPolicy,
    PolicyRateLimited,
    PolicyRequest,
)
from experiments.canonicalization.trace import event, failure_kind
from experiments.orientation.common import checked_file


def validation_feedback(error: ValueError, reply_type: type[BaseModel]) -> str:
    """Return bounded schema diagnostics without copying model-supplied values or keys."""
    message = "Response does not match the required JSON schema."
    if not isinstance(error, ValidationError):
        return message
    schema = reply_type.model_json_schema()
    models = [schema, *schema.get("$defs", {}).values()]
    fields = {name for model in models for name in model.get("properties", {})}
    details = []
    for issue in error.errors(include_input=False, include_context=False, include_url=False)[:5]:
        path = (
            ".".join(
                str(part) if isinstance(part, int) or part in fields else "<unknown field>"
                for part in issue["loc"]
            )
            or "<response>"
        )
        details.append(f"{path}: {issue['type']}")
        if issue["msg"] == "Value error, An export requires resolved instance semantics":
            details.append(
                "choice.scope exports a pose but semantics.status is uncertain. "
                "If identity and up are supported, front uncertainty belongs in front_mode "
                "and directions, and an up_only choice is possible. If identity is uncertain, "
                "choose unresolved. Do not change evidence claims just to pass validation."
            )
    return message + " " + "; ".join(details)


def call_policy[Reply: BaseModel](
    root: Path,
    state: dict[str, Any],
    name: str,
    request: PolicyRequest,
    reply_type: type[Reply],
    budget: int,
    policy: ModelPolicy,
) -> Reply:
    calls = state["calls"]
    directory = root / "calls"
    directory.mkdir(exist_ok=True)
    path = directory / f"{name}.json"
    descriptor = {
        "task": request.task,
        "prompt": request.prompt,
        "images": [{"id": label, "sha256": sha256(p)} for label, p in request.images],
        "response_schema": request.response_schema,
        "policy": policy.identity,
    }
    if name in calls:
        record = calls[name]
        if record["status"] == "rate_limited":
            checked_file(directory, path.name, record["sha256"])
            if read_json(path)["request"] != descriptor:
                raise ValueError("Saved model request changed")
            raise PolicyRateLimited("Saved quota/rate limit (HTTP 429); no retry dispatched")
        if record["status"] != "responded":
            raise RuntimeError(
                "A dispatched call has no saved response; automatic retry is disabled"
            )
        checked_file(directory, path.name, record["sha256"])
        saved = read_json(path)
        if saved["request"] != descriptor:
            raise ValueError("Saved model request changed")
        event(root, "model_response_reused", call=name, response_sha256=record["sha256"])
    else:
        if len(calls) >= budget:
            raise RuntimeError("Model call budget exhausted")
        calls[name] = {"status": "started"}
        write_json(root / "state.json", state)
        event(
            root,
            "model_dispatched",
            call=name,
            task=request.task,
            image_ids=[label for label, _ in request.images],
            budget_remaining=budget - len(calls),
            model=policy.identity.get("model"),
        )
        started = perf_counter()
        try:
            response = policy.select(request)
        except Exception as error:
            if isinstance(error, PolicyRateLimited):
                write_json(path, {"request": descriptor, "failure_kind": "rate_limit"})
                calls[name] = {"status": "rate_limited", "sha256": sha256(path)}
                write_json(root / "state.json", state)
            event(
                root,
                "model_failed",
                call=name,
                error_type=type(error).__name__,
                failure_kind=failure_kind(error),
                elapsed_seconds=perf_counter() - started,
            )
            raise
        saved = {"request": descriptor, "response": response}
        write_json(path, saved)
        calls[name] = {"status": "responded", "sha256": sha256(path)}
        write_json(root / "state.json", state)
        event(
            root,
            "model_response_saved",
            call=name,
            response_sha256=sha256(path),
            elapsed_seconds=perf_counter() - started,
            usage=response.get("usage"),
            resolved_model=response.get("resolved_model"),
        )
    try:
        reply = reply_type.model_validate_json(saved["response"]["raw_text"])
    except ValueError as error:
        event(root, "model_reply_rejected", call=name, error_type=type(error).__name__)
        raise InvalidPolicyReply(validation_feedback(error, reply_type)) from None
    event(root, "model_reply_parsed", call=name)
    return reply

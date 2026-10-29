"""Journal one bounded model dispatch and validate the reply before tool execution."""

from pathlib import Path
from time import perf_counter
from typing import Any

from pydantic import BaseModel

from experiments.artifacts import read_json, sha256, write_json
from experiments.canonicalization.model_policy import (
    InvalidPolicyReply,
    ModelPolicy,
    PolicyRateLimited,
    PolicyRequest,
)
from experiments.canonicalization.trace import event, failure_kind
from experiments.orientation.common import checked_file


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
        raise InvalidPolicyReply(
            "Response does not match the required JSON schema; check field types, "
            "required arguments and semantic/pose consistency."
        ) from None
    event(root, "model_reply_parsed", call=name)
    return reply

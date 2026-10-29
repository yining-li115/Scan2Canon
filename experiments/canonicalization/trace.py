"""Append-only execution observations, separate from authoritative job checkpoints."""

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def event(root: Path, kind: str, **fields: Any) -> None:
    record = {"schema_version": 1, "time": datetime.now(UTC).isoformat(), "event": kind, **fields}
    with (root / "events.jsonl").open("a") as handle:
        handle.write(json.dumps(record, allow_nan=False) + "\n")
        handle.flush()


def read_events(root: Path) -> tuple[list[dict[str, Any]], int]:
    path = root / "events.jsonl"
    records, malformed = [], 0
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                record = json.loads(line)
                if not isinstance(record, dict) or record.get("schema_version") != 1:
                    raise ValueError("Unsupported trace record")
                records.append(record)
            except (ValueError, TypeError):
                malformed += 1
    return records, malformed


def failure_kind(error: Exception) -> str:
    # Never persist arbitrary exception strings: a transport can include credentials.
    message = str(error).lower()
    if "http 429" in message:
        return "rate_limit"
    code = re.search(r"\bhttp (\d{3})\b", message)
    if code:
        return "http_" + code.group(1)
    if "did not return one completed candidate" in message:
        return "incomplete_response"
    if "api blocked the request" in message:
        return "provider_blocked"
    if isinstance(error, TimeoutError) or "transport failed" in message:
        return "transport"
    if isinstance(error, ValueError):
        return "validation"
    return "execution"

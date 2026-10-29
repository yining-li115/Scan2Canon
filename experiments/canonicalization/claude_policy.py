"""Bounded Claude vision transport with a free token-count preflight and no retries."""

import base64
import io
import json
import re
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from PIL import Image

from experiments.artifacts import sha256
from experiments.canonicalization.model_policy import PolicyRateLimited, PolicyReply, PolicyRequest


class ClaudePolicy:
    max_input_tokens = 50_000
    max_output_tokens = 8192
    api_version = "2026-10-21"

    def __init__(self, model: str, api_key: str, timeout: float = 60, *, stop_at: str = "") -> None:
        if not re.fullmatch(r"claude-[a-z0-9-]+", model) or not api_key.strip():
            raise ValueError("A Claude model and ANTHROPIC_API_KEY are required")
        self.model, self._key, self.timeout = model, api_key, timeout
        self.stop_at = datetime.fromisoformat(stop_at) if stop_at else None
        if self.stop_at is not None and self.stop_at.utcoffset() is None:
            raise ValueError("ANTHROPIC_STOP_AT must include a timezone")

    @property
    def identity(self) -> dict[str, str]:
        return {
            "backend": "anthropic_messages",
            "model": self.model,
            "implementation_sha256": sha256(Path(__file__)),
            "api_version": self.api_version,
            "response_format": "schema_prompt_local_validation",
            "max_input_tokens": str(self.max_input_tokens),
            "max_output_tokens": str(self.max_output_tokens),
            "timeout_seconds": str(self.timeout),
            "stop_at": self.stop_at.isoformat() if self.stop_at else "",
        }

    def _check_deadline(self) -> None:
        if self.stop_at is not None and datetime.now(UTC) >= self.stop_at:
            raise RuntimeError("Claude local credit deadline reached; no request dispatched")

    def _post(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        self._check_deadline()
        encoded = json.dumps(payload).encode()
        if len(encoded) > 20_000_000:
            raise ValueError("Claude request exceeded the size limit")
        http = urllib.request.Request(
            "https://api.anthropic.com/v1/messages" + endpoint,
            data=encoded,
            headers={
                "x-api-key": self._key,
                "anthropic-version": self.api_version,
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(http, timeout=self.timeout) as response:
                raw = response.read(2_000_001)
        except urllib.error.HTTPError as error:
            if error.code == 429:
                raise PolicyRateLimited(
                    "Claude API returned HTTP 429; no retry dispatched"
                ) from None
            raise RuntimeError(
                f"Claude API returned HTTP {error.code}; no retry dispatched"
            ) from None
        except (urllib.error.URLError, TimeoutError):
            raise RuntimeError("Claude API transport failed; request remains counted") from None
        if len(raw) > 2_000_000:
            raise ValueError("Claude response exceeded the size limit")
        try:
            body = json.loads(raw)
        except (ValueError, UnicodeError):
            raise ValueError("Claude returned malformed JSON") from None
        if not isinstance(body, dict):
            raise ValueError("Claude returned a non-object response")
        return body

    def select(self, request: PolicyRequest) -> dict[str, Any]:
        self._check_deadline()
        if not 3 <= len(request.images) <= 32:
            raise ValueError("Expected 3–32 labeled images")
        # Native constrained decoding rejects the full agent contract as grammar-too-large.
        # Supply that exact contract as instructions; controllers still validate every reply.
        schema = request.response_schema or PolicyReply.model_json_schema()
        prompt = (
            request.prompt + "\nReturn exactly one JSON object matching the following JSON Schema. "
            "Do not use Markdown fences or text outside JSON. All bounds and required "
            "fields are enforced locally before any action executes.\n" + json.dumps(schema)
        )
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for label, path in request.images:
            with Image.open(path) as source:
                image = Image.new("RGBA", source.size, "white")
                image.alpha_composite(source.convert("RGBA"))
                image.thumbnail((768, 768))
                buffer = io.BytesIO()
                image.convert("RGB").save(buffer, format="JPEG", quality=90)
            content.extend(
                [
                    {"type": "text", "text": label},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": base64.b64encode(buffer.getvalue()).decode("ascii"),
                        },
                    },
                ]
            )
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
        }
        counted = self._post("/count_tokens", payload).get("input_tokens")
        if type(counted) is not int or not 0 < counted <= self.max_input_tokens:
            raise RuntimeError(
                "Claude input token preflight failed or exceeded 50000; no generation"
            )
        body = self._post("", {**payload, "max_tokens": self.max_output_tokens})
        if body.get("stop_reason") != "end_turn":
            raise RuntimeError("Claude did not complete the response; no retry dispatched")
        blocks = body.get("content")
        if not isinstance(blocks, list):
            raise ValueError("Claude returned no content blocks")
        texts = [b.get("text") for b in blocks if isinstance(b, dict) and b.get("type") == "text"]
        if len(texts) != 1 or not isinstance(texts[0], str):
            raise ValueError("Expected exactly one Claude output text")
        return {
            "raw_text": texts[0],
            "response_id": body.get("id"),
            "resolved_model": body.get("model"),
            "usage": body.get("usage"),
            "preflight_input_tokens": counted,
        }

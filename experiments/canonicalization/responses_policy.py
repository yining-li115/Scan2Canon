"""One bounded Responses API request; no implicit retries or tool execution."""

import base64
import io
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from PIL import Image

from experiments.artifacts import sha256
from experiments.canonicalization.model_policy import PolicyReply, PolicyRequest


class ResponsesPolicy:
    def __init__(self, model: str, api_key: str, timeout: float = 60) -> None:
        if not model.strip() or not api_key.strip():
            raise ValueError("An explicit model and OPENAI_API_KEY are required")
        self.model, self._key, self.timeout = model, api_key, timeout

    @property
    def identity(self) -> dict[str, str]:
        return {
            "backend": "openai_responses",
            "model": self.model,
            "implementation_sha256": sha256(Path(__file__)),
            "max_output_tokens": "8192",
            "reasoning_effort": "medium",
            "timeout_seconds": str(self.timeout),
        }

    def select(self, request: PolicyRequest) -> dict[str, Any]:
        if not 3 <= len(request.images) <= 32:
            raise ValueError("Expected 3–32 labeled images")
        content: list[dict[str, Any]] = [{"type": "input_text", "text": request.prompt}]
        for label, path in request.images:
            with Image.open(path) as source:
                image = Image.new("RGBA", source.size, "white")
                image.alpha_composite(source.convert("RGBA"))
                image.thumbnail((768, 768))
                buffer = io.BytesIO()
                image.convert("RGB").save(buffer, format="JPEG", quality=90)
            data = base64.b64encode(buffer.getvalue()).decode("ascii")
            content.extend(
                [
                    {"type": "input_text", "text": label},
                    {
                        "type": "input_image",
                        "image_url": "data:image/jpeg;base64," + data,
                        "detail": "high",
                    },
                ]
            )
        schema = PolicyReply.model_json_schema()
        schema["required"] = list(schema["properties"])
        for field in schema["properties"].values():
            field.pop("default", None)
        payload = {
            "model": self.model,
            "store": False,
            "max_output_tokens": 8192,
            "reasoning": {"effort": "medium"},
            "input": [{"role": "user", "content": content}],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "canonical_selection",
                    "strict": True,
                    "schema": schema,
                }
            },
        }
        http = urllib.request.Request(
            "https://api.openai.com/v1/responses",
            data=json.dumps(payload).encode(),
            headers={"Authorization": "Bearer " + self._key, "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(http, timeout=self.timeout) as response:
                data_bytes = response.read(2_000_001)
        except urllib.error.HTTPError as error:
            raise RuntimeError(
                f"Responses API returned HTTP {error.code}; no retry dispatched"
            ) from None
        except (urllib.error.URLError, TimeoutError):
            raise RuntimeError("Responses API transport failed; request remains counted") from None
        if len(data_bytes) > 2_000_000:
            raise ValueError("Responses API response exceeded the size limit")
        body = json.loads(data_bytes)
        if body.get("status") != "completed":
            raise RuntimeError("Responses API did not complete the response")
        parts = [
            part
            for item in body.get("output", [])
            if item.get("type") == "message"
            for part in item.get("content", [])
        ]
        if any(part.get("type") == "refusal" for part in parts):
            raise RuntimeError("Responses API refused the request")
        texts = [part["text"] for part in parts if part.get("type") == "output_text"]
        if len(texts) != 1:
            raise ValueError("Expected exactly one model output text")
        return {
            "raw_text": texts[0],
            "response_id": body.get("id"),
            "resolved_model": body.get("model"),
            "usage": body.get("usage"),
        }

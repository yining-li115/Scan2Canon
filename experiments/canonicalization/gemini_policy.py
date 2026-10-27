"""Bounded Gemini image selection through the native generateContent endpoint."""

import base64
import io
import json
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from PIL import Image

from experiments.artifacts import sha256
from experiments.canonicalization.model_policy import PolicyReply, PolicyRequest


class GeminiPolicy:
    def __init__(self, model: str, api_key: str, timeout: float = 60) -> None:
        if not re.fullmatch(r"gemini-[A-Za-z0-9._-]+", model) or not api_key.strip():
            raise ValueError("A Gemini model ID and GEMINI_API_KEY are required")
        self.model, self._key, self.timeout = model, api_key, timeout

    @property
    def identity(self) -> dict[str, str]:
        return {
            "backend": "gemini_generate_content",
            "model": self.model,
            "implementation_sha256": sha256(Path(__file__)),
            "max_output_tokens": "8192",
            "thinking": "model_default",
            "timeout_seconds": str(self.timeout),
        }

    def select(self, request: PolicyRequest) -> dict[str, Any]:
        if not 3 <= len(request.images) <= 32:
            raise ValueError("Expected 3–32 labeled images")
        parts: list[dict[str, Any]] = [{"text": request.prompt}]
        for label, path in request.images:
            with Image.open(path) as source:
                image = Image.new("RGBA", source.size, "white")
                image.alpha_composite(source.convert("RGBA"))
                image.thumbnail((768, 768))
                buffer = io.BytesIO()
                image.convert("RGB").save(buffer, format="JPEG", quality=90)
            parts.extend(
                [
                    {"text": label},
                    {
                        "inlineData": {
                            "mimeType": "image/jpeg",
                            "data": base64.b64encode(buffer.getvalue()).decode("ascii"),
                        }
                    },
                ]
            )
        schema = PolicyReply.model_json_schema()
        schema["required"] = list(schema["properties"])
        for field in schema["properties"].values():
            field.pop("default", None)
        payload = json.dumps(
            {
                "contents": [{"role": "user", "parts": parts}],
                "generationConfig": {
                    "candidateCount": 1,
                    "maxOutputTokens": 8192,
                    "responseFormat": {"text": {"mimeType": "APPLICATION_JSON", "schema": schema}},
                },
            }
        ).encode()
        if len(payload) >= 20_000_000:
            raise ValueError("Gemini inline request must be smaller than 20 MB")
        http = urllib.request.Request(
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent",
            data=payload,
            headers={"x-goog-api-key": self._key, "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(http, timeout=self.timeout) as response:
                data = response.read(2_000_001)
        except urllib.error.HTTPError as error:
            if error.code == 429:
                raise RuntimeError(
                    "Gemini quota/rate limit reached (HTTP 429); no retry dispatched"
                ) from None
            raise RuntimeError(
                f"Gemini API returned HTTP {error.code}; no retry dispatched"
            ) from None
        except (urllib.error.URLError, TimeoutError):
            raise RuntimeError("Gemini API transport failed; request remains counted") from None
        if len(data) > 2_000_000:
            raise ValueError("Gemini API response exceeded the size limit")
        body = json.loads(data)
        if body.get("promptFeedback", {}).get("blockReason"):
            raise RuntimeError("Gemini API blocked the request")
        candidates = body.get("candidates", [])
        if len(candidates) != 1 or candidates[0].get("finishReason") != "STOP":
            raise RuntimeError("Gemini API did not return one completed candidate")
        texts = [
            part["text"]
            for part in candidates[0].get("content", {}).get("parts", [])
            if "text" in part and not part.get("thought", False)
        ]
        if not texts or not "".join(texts).strip():
            raise ValueError("Gemini API returned no output text")
        return {
            "raw_text": "".join(texts),
            "response_id": body.get("responseId"),
            "resolved_model": body.get("modelVersion"),
            "usage": body.get("usageMetadata"),
        }

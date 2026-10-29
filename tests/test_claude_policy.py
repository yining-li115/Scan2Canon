"""Claude transport safeguards and composition, with no live provider requests."""

import base64
import io
import json
import urllib.error
from email.message import Message
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from experiments.canonicalization.claude_policy import ClaudePolicy
from experiments.canonicalization.model_policy import PolicyRateLimited, PolicyRequest
from experiments.canonicalization.runtime import build_agent_controller


@pytest.fixture
def images(tmp_path: Path) -> PolicyRequest:
    path = tmp_path / "image.png"
    Image.new("RGBA", (1000, 500), (100, 50, 10, 128)).save(path)
    return PolicyRequest("agent", "Choose evidence", tuple((f"view-{i}", path) for i in range(3)))


def test_native_vision_preflight_schema_and_usage(images: PolicyRequest, monkeypatch: Any) -> None:
    calls = []

    def send(request: Any, timeout: float) -> io.BytesIO:
        calls.append(request)
        assert request.get_header("X-api-key") == "test-secret"
        payload = json.loads(request.data)
        assert "test-secret" not in request.data.decode()
        assert timeout == 60
        if request.full_url.endswith("/count_tokens"):
            assert "max_tokens" not in payload
            return io.BytesIO(b'{"input_tokens":1234}')
        assert request.full_url == "https://api.anthropic.com/v1/messages"
        assert payload["max_tokens"] == 8192
        assert "output_config" not in payload
        assert "JSON Schema" in payload["messages"][0]["content"][0]["text"]
        content = payload["messages"][0]["content"]
        assert [b["text"] for b in content if b["type"] == "text"][1:] == [
            label for label, _ in images.images
        ]
        for block in content:
            if block["type"] == "image":
                source = block["source"]
                assert source["media_type"] == "image/jpeg"
                with Image.open(io.BytesIO(base64.b64decode(source["data"]))) as decoded:
                    assert decoded.size == (768, 384)
        return io.BytesIO(
            json.dumps(
                {
                    "stop_reason": "end_turn",
                    "model": "claude-test",
                    "id": "msg-test",
                    "usage": {"input_tokens": 1234, "output_tokens": 100},
                    "content": [{"type": "text", "text": "{}"}],
                }
            ).encode()
        )

    monkeypatch.setattr("urllib.request.urlopen", send)
    policy = ClaudePolicy("claude-test", "test-secret")
    reply = policy.select(images)
    assert len(calls) == 2
    assert reply["usage"]["output_tokens"] == 100
    assert reply["preflight_input_tokens"] == 1234
    assert "test-secret" not in json.dumps([policy.identity, reply])


@pytest.mark.parametrize(
    "failure",
    ["quota", "auth", "timeout", "truncated", "refused", "oversize", "malformed", "count"],
)
def test_fail_closed_without_retries(images: PolicyRequest, monkeypatch: Any, failure: str) -> None:
    calls = []

    def send(request: Any, timeout: float) -> io.BytesIO:
        calls.append(request)
        if request.full_url.endswith("/count_tokens"):
            return io.BytesIO(
                json.dumps({"input_tokens": 50001 if failure == "count" else 100}).encode()
            )
        if failure in ("quota", "auth"):
            raise urllib.error.HTTPError(
                request.full_url, 429 if failure == "quota" else 401, "test-secret", Message(), None
            )
        if failure == "timeout":
            raise TimeoutError("test-secret")
        if failure == "oversize":
            return io.BytesIO(b"x" * 2_000_001)
        if failure == "malformed":
            return io.BytesIO(b"not json")
        return io.BytesIO(
            json.dumps(
                {"stop_reason": "refusal" if failure == "refused" else "max_tokens"}
            ).encode()
        )

    monkeypatch.setattr("urllib.request.urlopen", send)
    with pytest.raises((ValueError, RuntimeError)) as error:
        ClaudePolicy("claude-test", "test-secret").select(images)
    assert "test-secret" not in str(error.value)
    assert len(calls) == (1 if failure == "count" else 2)
    if failure == "quota":
        assert isinstance(error.value, PolicyRateLimited)


def test_deadline_blocks_calls_and_runtime_keeps_provider_explicit(
    images: PolicyRequest, monkeypatch: Any
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("Expired credit guard must block even token counting")

    monkeypatch.setattr("urllib.request.urlopen", forbidden)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-secret")
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-test")
    monkeypatch.setenv("ANTHROPIC_STOP_AT", "2020-01-01T00:00:00+00:00")
    controller = build_agent_controller("claude", "blender")
    assert len(controller.policies) == 1
    assert controller.policy.identity["model"] == "claude-test"
    with pytest.raises(RuntimeError, match="deadline"):
        controller.policy.select(images)
    with pytest.raises(ValueError, match="timezone"):
        ClaudePolicy("claude-test", "secret", stop_at="2026-11-05")


def test_prompt_contract_remains_enforced_locally(images: PolicyRequest, monkeypatch: Any) -> None:
    from experiments.canonicalization.agent_contracts import AgentTurn
    from tests.test_canonicalization_agent import turn

    contract = AgentTurn.model_json_schema()
    invalid = turn()
    invalid["directions"]["measurements"][0]["tail"]["x"] = 1.1

    def send(request: Any, timeout: float) -> io.BytesIO:
        payload = json.loads(request.data)
        prompt = payload["messages"][0]["content"][0]["text"]
        assert json.dumps(contract) in prompt
        if request.full_url.endswith("/count_tokens"):
            return io.BytesIO(b'{"input_tokens":100}')
        return io.BytesIO(
            json.dumps(
                {
                    "stop_reason": "end_turn",
                    "content": [{"type": "text", "text": json.dumps(invalid)}],
                }
            ).encode()
        )

    monkeypatch.setattr("urllib.request.urlopen", send)
    reply = ClaudePolicy("claude-test", "secret").select(
        PolicyRequest(images.task, images.prompt, images.images, contract)
    )
    with pytest.raises(ValueError):
        AgentTurn.model_validate_json(reply["raw_text"])

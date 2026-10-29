"""Gemini transport, configuration and failure behavior without live API calls."""

import base64
import io
import json
import urllib.error
import urllib.request
from email.message import Message
from pathlib import Path

import pytest
from PIL import Image

pytest.importorskip("numpy")

from experiments.canonicalization.gemini_policy import GeminiPolicy  # noqa: E402
from experiments.canonicalization.model_policy import PolicyReply, PolicyRequest  # noqa: E402
from experiments.canonicalization.runtime import build_controller  # noqa: E402
from experiments.canonicalization.worker import main  # noqa: E402


@pytest.fixture
def request_images(tmp_path: Path) -> PolicyRequest:
    image = tmp_path / "image.png"
    Image.new("RGBA", (1000, 500), (100, 50, 10, 128)).save(image)
    return PolicyRequest(
        "up",
        "Choose a supported candidate",
        tuple((f"Primary candidate {i}", image) for i in range(3)),
    )


def test_gemini_native_images_json_schema_and_usage(
    request_images: PolicyRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = json.dumps(
        {
            "selection": {
                "decision": "selected",
                "candidate_id": 0,
                "evidence_view_ids": [0, 1],
                "visible_evidence": "Fixture only",
                "missing_evidence": "",
            },
            "supporting_image_ids": [],
            "mesh_correspondence": "",
        }
    )
    calls = []

    def send(request: urllib.request.Request, timeout: float) -> io.BytesIO:
        calls.append(request)
        assert (
            request.full_url
            == "https://generativelanguage.googleapis.com/v1beta/models/gemini-test:generateContent"
        )
        assert request.get_header("X-goog-api-key") == "test-secret"
        assert "test-secret" not in request.full_url
        assert timeout == 60
        assert isinstance(request.data, bytes)
        payload = json.loads(request.data)
        assert "test-secret" not in request.data.decode()
        config = payload["generationConfig"]
        assert config["candidateCount"] == 1
        assert config["maxOutputTokens"] == 8192
        output = config["responseFormat"]["text"]
        assert output["mimeType"] == "APPLICATION_JSON"
        assert set(output["schema"]["required"]) == set(output["schema"]["properties"])
        parts = payload["contents"][0]["parts"]
        assert [p["text"] for p in parts if "text" in p][1:] == [
            label for label, _ in request_images.images
        ]
        images = [p["inlineData"] for p in parts if "inlineData" in p]
        assert len(images) == 3
        for image in images:
            assert image["mimeType"] == "image/jpeg"
            with Image.open(io.BytesIO(base64.b64decode(image["data"]))) as decoded:
                assert decoded.size == (768, 384)
                assert decoded.mode == "RGB"
        return io.BytesIO(
            json.dumps(
                {
                    "candidates": [
                        {
                            "finishReason": "STOP",
                            "content": {
                                "parts": [
                                    {"text": "Internal thought fixture", "thought": True},
                                    {"text": raw},
                                ]
                            },
                        }
                    ],
                    "modelVersion": "gemini-test-version",
                    "responseId": "fixture-id",
                    "usageMetadata": {"promptTokenCount": 100, "candidatesTokenCount": 50},
                }
            ).encode()
        )

    monkeypatch.setattr("urllib.request.urlopen", send)
    policy = GeminiPolicy("gemini-test", "test-secret")
    result = policy.select(request_images)
    assert PolicyReply.model_validate_json(result["raw_text"]).selection.candidate_id == 0
    assert result["resolved_model"] == "gemini-test-version"
    assert result["usage"]["promptTokenCount"] == 100
    assert len(calls) == 1
    assert "Internal thought" not in result["raw_text"]
    assert "test-secret" not in json.dumps(policy.identity)
    assert "test-secret" not in json.dumps(result)


@pytest.mark.parametrize(
    "failure", ["quota", "auth", "timeout", "blocked", "truncated", "empty", "malformed"]
)
def test_gemini_fails_closed_without_retries(
    request_images: PolicyRequest,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    calls = []

    def send(request: urllib.request.Request, timeout: float) -> io.BytesIO:
        calls.append(request)
        if failure in ("quota", "auth"):
            raise urllib.error.HTTPError(
                request.full_url, 429 if failure == "quota" else 403, "test-secret", Message(), None
            )
        if failure == "timeout":
            raise TimeoutError("test-secret")
        if failure == "malformed":
            return io.BytesIO(b"not JSON")
        if failure == "blocked":
            return io.BytesIO(b'{"promptFeedback": {"blockReason": "SAFETY"}}')
        return io.BytesIO(
            json.dumps(
                {
                    "candidates": [
                        {
                            "finishReason": "MAX_TOKENS" if failure == "truncated" else "STOP",
                            "content": {"parts": []},
                        }
                    ]
                }
            ).encode()
        )

    monkeypatch.setattr("urllib.request.urlopen", send)
    with pytest.raises((RuntimeError, ValueError)) as error:
        GeminiPolicy("gemini-test", "test-secret").select(request_images)
    assert "test-secret" not in str(error.value)
    if failure == "quota":
        from experiments.canonicalization.model_policy import PolicyRateLimited

        assert isinstance(error.value, PolicyRateLimited)
        assert "429" in str(error.value)
    assert len(calls) == 1


def test_agent_model_pool_and_explicit_pin(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from experiments.canonicalization.runtime import build_agent_controller

    monkeypatch.setenv("GEMINI_API_KEY", "test-secret")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.8-flash")
    monkeypatch.delenv("GEMINI_MODELS_FILE", raising=False)
    controller = build_agent_controller("gemini", "blender")
    assert [p.identity["model"] for p in controller.policies] == [
        "gemini-3.8-flash",
        "gemini-3.6-flash",
        "gemini-3.5-flash-lite",
    ]
    assert "test-secret" not in json.dumps(controller.policy_identity)
    pool = tmp_path / "models.json"
    pool.write_text(json.dumps({"models": ["gemini-3.8-flash", "gemini-3.8-flash"]}))
    monkeypatch.setenv("GEMINI_MODELS_FILE", str(pool))
    with pytest.raises(ValueError, match="distinct"):
        build_agent_controller("gemini", "blender")
    assert len(build_agent_controller("gemini", "blender", model="gemini-3.6-flash").policies) == 1


def test_gemini_runtime_reads_its_own_key_and_model_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-secret")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-env")
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-secret")
    assert build_controller("gemini", "blender").policy.identity["model"] == "gemini-env"
    assert (
        build_controller("gemini", "blender", "gemini-cli").policy.identity["model"] == "gemini-cli"
    )
    monkeypatch.delenv("GEMINI_API_KEY")
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        build_controller("gemini", "blender")
    with pytest.raises(ValueError):
        GeminiPolicy("gemini-test/../../other?key=bad", "test-secret")


def test_gemini_cli_missing_key_stops_before_creating_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_MODEL", "gemini-test")
    output = tmp_path / "job"
    monkeypatch.setattr(
        "sys.argv",
        ["worker", str(tmp_path / "mesh.glb"), "mug", str(output), "--backend", "gemini"],
    )
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 1
    assert "GEMINI_API_KEY" in capsys.readouterr().err
    assert not output.exists()


def test_schema_rejection_is_classified_without_leaking_provider_text(
    request_images: PolicyRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from experiments.canonicalization.trace import failure_kind

    calls = []

    def send(request: urllib.request.Request, timeout: float) -> io.BytesIO:
        calls.append(request)
        raise urllib.error.HTTPError(
            request.full_url,
            400,
            "test-secret",
            Message(),
            io.BytesIO(b'{"error":{"message":"Schema produces too many states; test-secret"}}'),
        )

    monkeypatch.setattr("urllib.request.urlopen", send)
    with pytest.raises(RuntimeError, match="schema complexity") as error:
        GeminiPolicy("gemini-test", "test-secret").select(request_images)
    assert failure_kind(error.value) == "http_400"
    assert "test-secret" not in str(error.value)
    assert len(calls) == 1


def test_transport_schema_simplifies_grammar_without_mutating_local_contract() -> None:
    from experiments.canonicalization.agent_contracts import AgentTurn
    from experiments.canonicalization.gemini_policy import transport_schema
    from tests.test_canonicalization_agent import turn

    contract = AgentTurn.model_json_schema()
    adapted = transport_schema(contract)
    assert contract["$defs"]["ImagePoint"]["properties"]["x"]["maximum"] == 1
    assert "maximum" not in adapted["$defs"]["ImagePoint"]["properties"]["x"]
    assert "maxItems" not in adapted["$defs"]["DirectionAssessment"]["properties"]["measurements"]
    assert adapted["required"] == contract["required"]
    invalid = turn()
    invalid["directions"]["measurements"][0]["tail"]["x"] = 1.1
    with pytest.raises(ValueError):
        AgentTurn.model_validate(invalid)

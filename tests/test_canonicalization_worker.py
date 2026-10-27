"""Controller behavior with real geometry/export and explicit render/model fixtures."""

import io
import json
import shutil
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

pytest.importorskip("numpy")
trimesh = pytest.importorskip("trimesh")

from experiments.artifacts import read_json, sha256, write_json  # noqa: E402
from experiments.canonicalization.compute import GeometryTools, copy_observations  # noqa: E402
from experiments.canonicalization.controller import Controller  # noqa: E402
from experiments.canonicalization.model_policy import PolicyRequest  # noqa: E402
from experiments.canonicalization.responses_policy import ResponsesPolicy  # noqa: E402


def reply(candidate: int | None = 0, **extra: Any) -> dict[str, Any]:
    return {
        "raw_text": json.dumps(
            {
                "selection": {
                    "decision": "selected" if candidate is not None else "needs_evidence",
                    "candidate_id": candidate,
                    "evidence_view_ids": [candidate, 1] if candidate is not None else [0, 1],
                    "visible_evidence": "Fixture response, not a semantic measurement",
                    "missing_evidence": "Need a view of the opening" if candidate is None else "",
                },
                "supporting_image_ids": [],
                "mesh_correspondence": "",
                **extra,
            }
        )
    }


class FixturePolicy:
    identity = {"backend": "test_fixture", "model": "scripted-not-inference"}

    def __init__(self, responses: list[Any]) -> None:
        self.responses = iter(responses)
        self.requests: list[PolicyRequest] = []

    def select(self, request: PolicyRequest) -> dict[str, Any]:
        self.requests.append(request)
        value = next(self.responses)
        if isinstance(value, Exception):
            raise value
        result: dict[str, Any] = value
        return result


@pytest.fixture
def mesh(tmp_path: Path) -> Path:
    path = tmp_path / "source.glb"
    trimesh.creation.box(extents=[1, 2, 3]).export(path)
    return path


@pytest.fixture
def renders(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    calls: list[Path] = []

    def render(
        mesh: Path,
        output: Path,
        blender: str,
        *,
        poses: list[Any],
        pose_camera: tuple[float, float, float] = (1, -2, 1),
    ) -> Path:
        calls.append(output)
        output.mkdir(parents=True)
        shutil.copyfile(mesh, output / "source.glb")
        views = []
        for i in range(len(poses)):
            path = output / f"view-{i}.png"
            Image.new("RGB", (16, 16), (i * 10, len(calls), 0)).save(path)
            views.append({"id": i, "image": path.name, "sha256": sha256(path)})
        write_json(output / "poses.json", {"rotations": poses, "camera_direction": pose_camera})
        write_json(
            output / "manifest.json",
            {
                "schema_version": 1,
                "status": "succeeded",
                "mesh": "source.glb",
                "mesh_sha256": sha256(mesh),
                "views": views,
            },
        )
        return output

    for module in ("upright", "front", "compute"):
        monkeypatch.setattr(f"experiments.canonicalization.{module}.render", render)
    return calls


def test_full_frame_is_reversible_and_resume_makes_no_calls(
    mesh: Path,
    renders: list[Path],
    tmp_path: Path,
) -> None:
    policy = FixturePolicy([reply(), reply()])
    controller = Controller(policy, GeometryTools("fixture"))
    root = tmp_path / "job"
    result = controller.run(mesh, "mug", root)
    report = read_json(result / "canonicalization.json")
    assert report["scope"] == "full_frame"
    assert report["dataset_approval"] == "not_granted"
    assert report["artifact"]["inverse_vertex_error"] < report["artifact"]["tolerance"]
    assert report["model_calls"] == 2
    assert [r.task for r in policy.requests] == ["up", "front"]
    assert len(renders) == 2
    assert controller.run(mesh, "mug", root, resume=True) == result
    assert len(policy.requests) == len(renders) == 2
    assert sha256(root / "work/inputs-1/source.glb") == sha256(mesh)
    with pytest.raises(FileExistsError):
        controller.run(mesh, "mug", root)


def test_extra_evidence_rounds_survive_resume_at_exhausted_budget(
    mesh: Path,
    renders: list[Path],
    tmp_path: Path,
) -> None:
    policy = FixturePolicy([reply(None), reply(), reply(None), reply(None)])
    controller = Controller(policy, GeometryTools("fixture"))
    root = tmp_path / "job"
    result = controller.run(mesh, "mug", root)
    state = read_json(root / "state.json")
    assert state["status"] == "needs_evidence"
    assert state["scope"] == "up_only_front_pending"
    assert state["model_calls"] == 4
    assert state["missing_evidence"] == "Need a view of the opening"
    assert (result / "canonical.glb").is_file()
    assert len(policy.requests[1].images) > len(policy.requests[0].images)
    assert any("opposite-candidate" in label for label, _ in policy.requests[1].images)
    assert controller.run(mesh, "mug", root, resume=True) == result
    assert len(policy.requests) == len(renders) == 4


@pytest.mark.parametrize(
    ("category", "choice", "scope", "calls"),
    [
        ("bottle", 0, "up_only", 1),
        ("mug", None, "unresolved", 2),
    ],
)
def test_symmetry_and_abstention_do_not_create_a_full_frame(
    mesh: Path,
    renders: list[Path],
    tmp_path: Path,
    category: str,
    choice: int | None,
    scope: str,
    calls: int,
) -> None:
    policy = FixturePolicy([reply(choice)] * calls)
    result = Controller(policy, GeometryTools("fixture")).run(mesh, category, tmp_path / "job")
    assert read_json(result / "canonicalization.json")["scope"] == scope
    assert (result / "canonical.glb").exists() == (choice is not None)
    assert len(policy.requests) == len(renders) == calls


def test_smaller_budget_preserves_up_and_explains_missing_front(
    mesh: Path,
    renders: list[Path],
    tmp_path: Path,
) -> None:
    policy = FixturePolicy([reply()])
    root = tmp_path / "job"
    Controller(policy, GeometryTools("fixture")).run(mesh, "mug", root, max_calls=1)
    state = read_json(root / "state.json")
    assert state["scope"] == "up_only_front_pending"
    assert "budget exhausted" in state["missing_evidence"]
    assert len(policy.requests) == len(renders) == 1


def test_interrupted_provider_call_is_counted_on_resume(
    mesh: Path,
    renders: list[Path],
    tmp_path: Path,
) -> None:
    policy = FixturePolicy([TimeoutError(), reply()])
    controller = Controller(policy, GeometryTools("fixture"))
    root = tmp_path / "job"
    with pytest.raises(TimeoutError):
        controller.run(mesh, "bottle", root)
    assert read_json(root / "calls/up-1.json")["status"] == "started"
    result = controller.run(mesh, "bottle", root, resume=True)
    assert read_json(result / "canonicalization.json")["model_calls"] == 2
    assert len(policy.requests) == 2
    assert len(renders) == 1


def test_crash_after_export_preserves_artifacts_and_recovers_without_model_calls(
    mesh: Path,
    renders: list[Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = FixturePolicy([reply()])
    tools = GeometryTools("fixture")
    original = tools.export
    controller = Controller(policy, tools)
    root = tmp_path / "job"

    def interrupted(*args: Any, **kwargs: Any) -> Path:
        original(*args, **kwargs)
        raise RuntimeError("Simulated interruption before checkpoint")

    monkeypatch.setattr(tools, "export", interrupted)
    with pytest.raises(RuntimeError, match="interruption"):
        controller.run(mesh, "bottle", root)
    saved = sha256(root / "work/export-1/canonical.glb")
    monkeypatch.setattr(tools, "export", original)
    result = controller.run(mesh, "bottle", root, resume=True)
    assert result.name == "export-2"
    assert sha256(root / "work/export-1/canonical.glb") == saved
    assert len(policy.requests) == len(renders) == 1


@pytest.mark.parametrize(
    "response",
    [
        reply(99),
        reply(rotation=[[1, 0, 0]]),
        {"raw_text": "not JSON"},
        reply(supporting_image_ids=["invented"], mesh_correspondence="guess"),
    ],
)
def test_invalid_model_output_never_exports_or_retries_silently(
    mesh: Path,
    renders: list[Path],
    tmp_path: Path,
    response: dict[str, Any],
) -> None:
    policy = FixturePolicy([response])
    controller = Controller(policy, GeometryTools("fixture"))
    root = tmp_path / "job"
    for resume in (False, True):
        with pytest.raises((ValueError, OSError)):
            controller.run(mesh, "bottle", root, resume=resume)
    assert len(policy.requests) == len(renders) == 1
    assert read_json(root / "state.json")["status"] == "failed"
    assert not list(root.rglob("canonical.glb"))


@pytest.mark.parametrize("change", ["mesh", "budget", "policy", "render", "decision"])
def test_resume_rejects_changed_inputs_or_completed_artifacts(
    mesh: Path,
    renders: list[Path],
    tmp_path: Path,
    change: str,
) -> None:
    policy = FixturePolicy([reply()])
    controller = Controller(policy, GeometryTools("fixture"))
    root = tmp_path / "job"
    controller.run(mesh, "bottle", root)
    budget = 4
    if change == "mesh":
        trimesh.creation.box(extents=[2, 3, 4]).export(mesh)
    elif change == "budget":
        budget = 3
    elif change == "policy":
        policy.identity = {"backend": "different"}
    elif change == "render":
        (root / "work/up-1/poses/view-0.png").write_bytes(b"replaced")
    else:
        (root / "decisions/up-1.json").write_text("{}")
    with pytest.raises(ValueError):
        controller.run(mesh, "bottle", root, resume=True, max_calls=budget)
    assert len(policy.requests) == len(renders) == 1


def test_copy_observations_rejects_absolute_paths_even_inside_source(
    mesh: Path,
    tmp_path: Path,
) -> None:
    source = tmp_path / "photos"
    source.mkdir()
    image = source / "photo.png"
    Image.new("RGB", (16, 16)).save(image)
    case = {"mesh_sha256": sha256(mesh), "category": "bottle"}
    write_json(
        source / "manifest.json",
        {
            "schema_version": 1,
            "status": "succeeded",
            **case,
            "registration": "unregistered",
            "assets": [],
            "images": [{"id": "photo", "image": str(image), "sha256": sha256(image)}],
        },
    )
    with pytest.raises(ValueError, match="relative"):
        copy_observations(source, tmp_path / "copy", case)
    assert image.is_file()


@pytest.fixture
def api_request(tmp_path: Path) -> PolicyRequest:
    path = tmp_path / "image.png"
    Image.new("RGBA", (32, 32), (50, 100, 150, 128)).save(path)
    return PolicyRequest("up", "Select a candidate", tuple((str(i), path) for i in range(3)))


def test_responses_request_uses_strict_schema_images_and_no_key_in_provenance(
    api_request: PolicyRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sent = []

    def send(request: urllib.request.Request, timeout: float) -> io.BytesIO:
        sent.append(request)
        assert timeout == 60
        assert request.full_url == "https://api.openai.com/v1/responses"
        assert isinstance(request.data, bytes)
        payload = json.loads(request.data)
        assert payload["store"] is False
        assert payload["max_output_tokens"] == 8192
        assert payload["reasoning"] == {"effort": "medium"}
        schema = payload["text"]["format"]["schema"]
        assert payload["text"]["format"]["strict"] is True
        assert set(schema["required"]) == set(schema["properties"])
        assert schema["additionalProperties"] is False
        assert schema["$defs"]["Selection"]["additionalProperties"] is False
        images = [c for c in payload["input"][0]["content"] if c["type"] == "input_image"]
        assert len(images) == 3
        assert images[0]["image_url"].startswith("data:image/jpeg;base64,")
        return io.BytesIO(
            json.dumps(
                {
                    "status": "completed",
                    "model": "resolved-model",
                    "output": [
                        {
                            "type": "message",
                            "content": [{"type": "output_text", "text": reply()["raw_text"]}],
                        }
                    ],
                    "usage": {"input_tokens": 100, "output_tokens": 200},
                }
            ).encode()
        )

    monkeypatch.setattr("urllib.request.urlopen", send)
    policy = ResponsesPolicy("test-model", "test-secret")
    result = policy.select(api_request)
    assert result["usage"]["input_tokens"] == 100
    assert result["resolved_model"] == "resolved-model"
    assert len(sent) == 1
    assert "test-secret" not in json.dumps(policy.identity)
    assert "test-secret" not in json.dumps(result)


@pytest.mark.parametrize("failure", ["timeout", "http", "incomplete", "refusal"])
def test_responses_failure_is_not_retried_or_reported_as_success(
    api_request: PolicyRequest,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    sent = []

    def send(request: urllib.request.Request, timeout: float) -> io.BytesIO:
        sent.append(request)
        if failure == "timeout":
            raise TimeoutError("test-secret")
        if failure == "http":
            raise urllib.error.HTTPError(request.full_url, 401, "test-secret", {}, None)  # type: ignore[arg-type]
        body = (
            {"status": "incomplete"}
            if failure == "incomplete"
            else {
                "status": "completed",
                "output": [{"type": "message", "content": [{"type": "refusal"}]}],
            }
        )
        return io.BytesIO(json.dumps(body).encode())

    monkeypatch.setattr("urllib.request.urlopen", send)
    with pytest.raises(RuntimeError) as error:
        ResponsesPolicy("test-model", "test-secret").select(api_request)
    assert "test-secret" not in str(error.value)
    assert len(sent) == 1


def test_response_saved_before_interruption_is_validated_without_another_call(
    mesh: Path,
    renders: list[Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = FixturePolicy([reply()])
    controller = Controller(policy, GeometryTools("fixture"))
    root = tmp_path / "job"
    original = controller._validate_reply

    def interrupted(*args: Any, **kwargs: Any) -> Path:
        raise RuntimeError("Interrupted before validation")

    monkeypatch.setattr(controller, "_validate_reply", interrupted)
    with pytest.raises(RuntimeError, match="Interrupted"):
        controller.run(mesh, "bottle", root)
    assert read_json(root / "calls/up-1.json")["status"] == "responded"
    monkeypatch.setattr(controller, "_validate_reply", original)
    result = controller.run(mesh, "bottle", root, resume=True)
    assert read_json(result / "canonicalization.json")["scope"] == "up_only"
    assert len(policy.requests) == len(renders) == 1


def test_two_failed_calls_exhaust_stage_budget_across_restarts(
    mesh: Path,
    renders: list[Path],
    tmp_path: Path,
) -> None:
    policy = FixturePolicy([TimeoutError(), TimeoutError()])
    controller = Controller(policy, GeometryTools("fixture"))
    root = tmp_path / "job"
    for resume in (False, True):
        with pytest.raises(TimeoutError):
            controller.run(mesh, "bottle", root, resume=resume)
    with pytest.raises(RuntimeError, match="budget exhausted"):
        controller.run(mesh, "bottle", root, resume=True)
    assert len(policy.requests) == 2
    assert len(renders) == 1
    assert not list(root.rglob("canonical.glb"))


def test_verified_photo_is_snapshotted_sent_and_cited(
    mesh: Path,
    renders: list[Path],
    tmp_path: Path,
) -> None:
    photos = tmp_path / "photos"
    photos.mkdir()
    image = photos / "photo.png"
    Image.new("RGB", (16, 16)).save(image)
    write_json(
        photos / "manifest.json",
        {
            "schema_version": 1,
            "status": "succeeded",
            "mesh_sha256": sha256(mesh),
            "category": "bottle",
            "registration": "unregistered",
            "assets": [],
            "images": [{"id": "photo-1", "image": image.name, "sha256": sha256(image)}],
        },
    )
    policy = FixturePolicy(
        [reply(supporting_image_ids=["photo-1"], mesh_correspondence="Fixture part matching")]
    )
    root = tmp_path / "job"
    result = Controller(policy, GeometryTools("fixture")).run(mesh, "bottle", root, photos)
    assert "Supplement photo-1" in [label for label, _ in policy.requests[0].images]
    report = read_json(result / "canonicalization.json")
    assert report["decisions"][0]["supporting_evidence"]["image_ids"] == ["photo-1"]
    assert sha256(root / "work/inputs-1/observations/photo.png") == sha256(image)

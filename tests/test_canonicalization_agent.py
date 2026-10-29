"""Agent-selected evidence, bounded execution and user-capture continuation."""

from pathlib import Path
from typing import Any

import pytest
from PIL import Image

pytest.importorskip("numpy")
from experiments.artifacts import read_json, sha256, write_json  # noqa: E402
from experiments.canonicalization.agent_contracts import AgentTurn  # noqa: E402
from experiments.canonicalization.agent_controller import AgentController, submit_capture  # noqa: E402
from experiments.canonicalization.open_tools import OpenGeometryTools  # noqa: E402
from tests.canonical_fixtures import (  # noqa: E402
    FixturePolicy,
    identity,
    pose,
    render_fixture,
)
from tests.canonical_fixtures import (
    bundles as bundles,
)
from tests.canonical_fixtures import (
    rendering as rendering,
)


def direction_fixture(choice: dict[str, Any]) -> dict[str, Any]:
    candidate = choice["candidate_id"]
    axes = ["up", "front"] if choice["scope"] == "full_frame" else ["up"]
    return {
        "hypotheses": [
            {
                "id": axis,
                "axis": axis,
                "role": "upright" if axis == "up" else "functional",
                "description": "Synthetic direction relation; not measured semantics",
                "from_part": "fixture start",
                "to_part": "fixture end",
                "evidence_ids": [f"candidate:{candidate or 0}"],
            }
            for axis in axes
        ],
        "selected_up": "up" if candidate is not None else None,
        "selected_front": "front" if "front" in axes and candidate is not None else None,
        "issues": [],
        "symmetry": {
            "yaw": "unknown",
            "basis": "both",
            "evidence_ids": [],
            "explanation": "Not assessed",
            "counterevidence": "",
        },
        "measurements": [
            {
                "hypothesis_id": axis,
                "candidate_id": candidate,
                "image_id": f"candidate:{candidate}",
                "tail": {"x": 0.5, "y": 0.5},
                "head": {"x": 0.5 if axis == "up" else 0.8, "y": 0.2 if axis == "up" else 0.5},
                "uncertainty": 0.01,
            }
            for axis in axes
        ]
        if candidate is not None
        else [],
    }


def turn(
    kind: str = "finish",
    *,
    semantics: dict[str, Any] | None = None,
    choice: dict[str, Any] | None = None,
    directions: dict[str, Any] | None = None,
    **arguments: Any,
) -> dict[str, Any]:
    return {
        "semantics": semantics or identity(),
        "directions": directions or direction_fixture(choice or pose()),
        "action": {
            "kind": kind,
            "reason": "Fixture: disambiguate visible parts",
            "expected_evidence": "Fixture: see the dial and support feet",
            "candidate_ids": None,
            "camera": None,
            "image_id": None,
            "crop": None,
            "photo_ids": None,
            "question": None,
            **arguments,
        },
        "choice": (choice or pose()) if kind == "finish" else None,
    }


def photos(path: Path, mesh: Path) -> Path:
    path.mkdir()
    image = path / "photo.png"
    Image.new("RGB", (32, 24), "red").save(image)
    write_json(
        path / "manifest.json",
        {
            "schema_version": 1,
            "status": "succeeded",
            "category": "unclassified",
            "mesh_sha256": sha256(mesh),
            "registration": "unregistered",
            "assets": [],
            "images": [{"id": "dial", "image": image.name, "sha256": sha256(image)}],
        },
    )
    return path


def test_finish_can_expect_no_further_evidence() -> None:
    reply = AgentTurn.model_validate(turn(expected_evidence=""))
    assert reply.choice is not None and reply.action.kind == "finish"
    with pytest.raises(ValueError, match="Only finish actions require"):
        AgentTurn.model_validate({**turn(expected_evidence=""), "choice": None})


@pytest.mark.parametrize("expected", ["", "  "])
def test_evidence_action_requires_expected_observation(expected: str) -> None:
    with pytest.raises(ValueError, match="describe the expected evidence"):
        AgentTurn.model_validate(
            turn("refine_up", candidate_ids=[0, 1, 2], expected_evidence=expected)
        )


def test_agent_selects_two_distinct_crops_then_finishes(
    bundles: tuple[Path, Path],
    rendering: list[Path],
    tmp_path: Path,
) -> None:
    policy = FixturePolicy(
        [
            turn("inspect_image", image_id="candidate:0", crop=[0.1, 0.1, 0.9, 0.9]),
            turn("inspect_image", image_id="source:1", crop=[0.1, 0.1, 0.9, 0.9]),
            turn(),
        ]
    )
    controller = AgentController(policy, OpenGeometryTools("fixture"))
    root = tmp_path / "job"
    result = controller.run(*bundles, root, max_calls=4)
    report = read_json(result / "canonicalization.json")
    assert report["route"] == "oa-evidence-agent-v3" and report["scope"] == "full_frame"
    assert [a["action"]["kind"] for a in report["action_history"]] == [
        "inspect_image",
        "inspect_image",
    ]
    assert "crop:crop-1" in dict(policy.requests[1].images)
    assert "crop:crop-2" in dict(policy.requests[2].images)
    assert (result / "canonical.glb").is_file()
    assert controller.run(*bundles, root, resume=True, max_calls=4) == result
    assert len(policy.requests) == 3 and len(rendering) == 1


def test_agent_selects_new_camera_and_records_candidate_correspondence(
    bundles: tuple[Path, Path],
    rendering: list[Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cameras = []

    def render(mesh: Path, output: Path, blender: str, **options: Any) -> Path:
        cameras.append(options["pose_camera"])
        return render_fixture(mesh, output, len(options["poses"]), options["poses"])

    monkeypatch.setattr("experiments.canonicalization.agent_tools.render", render)
    policy = FixturePolicy([turn("render_view", candidate_ids=[2, 0, 1], camera="left"), turn()])
    result = AgentController(policy, OpenGeometryTools("fixture")).run(*bundles, tmp_path / "job")
    history = read_json(result / "canonicalization.json")["action_history"]
    assert [i["candidate_id"] for i in history[0]["images"]] == [2, 0, 1]
    assert cameras == [(-2, 0, 0.7)]
    assert "detail:1:2:left" in dict(policy.requests[1].images)


def test_agent_retrieves_photos_and_photo_crop_requires_correspondence(
    bundles: tuple[Path, Path],
    rendering: list[Path],
    tmp_path: Path,
) -> None:
    bundle = photos(tmp_path / "photos", bundles[0] / "source.glb")
    manifest = read_json(bundle / "manifest.json")
    dial = manifest["images"][0]
    manifest["images"] = [{**dial, "id": "context-a"}, {**dial, "id": "context-b"}, dial]
    write_json(bundle / "manifest.json", manifest)
    policy = FixturePolicy(
        [
            turn("retrieve_photos", photo_ids=["photo:dial"]),
            turn("inspect_image", image_id="photo:dial", crop=[0.1, 0.1, 0.9, 0.9]),
            turn(choice=pose(evidence_ids=["candidate:0", "candidate:1", "photo:crop-2"])),
        ]
    )
    root = tmp_path / "job"
    with pytest.raises(ValueError, match="mesh correspondence"):
        AgentController(policy, OpenGeometryTools("fixture")).run(
            *bundles, root, observations=bundle
        )
    assert "photo:dial" not in dict(policy.requests[0].images)
    assert "photo:dial" in dict(policy.requests[1].images)
    assert "photo:crop-2" in dict(policy.requests[2].images)
    assert not list(root.rglob("canonical.glb"))


def test_capture_pause_and_same_task_continuation_reuse_calls(
    bundles: tuple[Path, Path],
    rendering: list[Path],
    tmp_path: Path,
) -> None:
    policy = FixturePolicy(
        [
            turn(
                "request_capture",
                question="Show the dial markings straight on",
                semantics=identity(status="uncertain", missing_evidence="Show markings"),
            ),
            turn(
                choice=pose(
                    evidence_ids=["candidate:0", "candidate:1", "photo:capture-1-dial"],
                    mesh_correspondence="Same dial and seam as the mesh",
                )
            ),
        ]
    )
    controller = AgentController(policy, OpenGeometryTools("fixture"))
    root = tmp_path / "job"
    paused = controller.run(*bundles, root)
    before = sha256(paused / "canonicalization.json")
    report = read_json(paused / "canonicalization.json")
    assert report["status"] == "needs_evidence" and report["pending_capture"]["turn"] == 1
    assert controller.run(*bundles, root, resume=True) == paused
    assert len(policy.requests) == 1
    bundle = photos(tmp_path / "new-photos", bundles[0] / "source.glb")
    submit_capture(root, bundle)
    submit_capture(root, bundle)  # Same submission is idempotent before continuation.
    result = controller.run(*bundles, root, resume=True)
    assert result != paused and (result / "canonical.glb").is_file()
    assert sha256(paused / "canonicalization.json") == before
    assert len(policy.requests) == 2 and len(rendering) == 1
    assert read_json(root / "state.json")["pending_capture"] is None
    with pytest.raises(ValueError, match="not waiting"):
        submit_capture(root, bundle)


def test_wrong_mesh_capture_cannot_resume(
    bundles: tuple[Path, Path],
    rendering: list[Path],
    tmp_path: Path,
) -> None:
    controller = AgentController(
        FixturePolicy([turn("request_capture", question="Show dial")]), OpenGeometryTools("fixture")
    )
    root = tmp_path / "job"
    controller.run(*bundles, root)
    bundle = photos(tmp_path / "photos", bundles[0] / "source.glb")
    manifest = read_json(bundle / "manifest.json")
    manifest["mesh_sha256"] = "0" * 64
    write_json(bundle / "manifest.json", manifest)
    with pytest.raises(ValueError, match="do not match"):
        submit_capture(root, bundle)
    assert "submissions" not in read_json(root / "state.json")


def test_last_call_does_not_execute_unusable_tool(
    bundles: tuple[Path, Path],
    rendering: list[Path],
    tmp_path: Path,
) -> None:
    policy = FixturePolicy(
        [
            turn("inspect_image", image_id="source:1", crop=[0.1, 0.1, 0.9, 0.9]),
            turn("inspect_image", image_id="source:0", crop=[0.1, 0.1, 0.8, 0.8]),
        ]
    )
    root = tmp_path / "job"
    result = AgentController(policy, OpenGeometryTools("fixture")).run(*bundles, root, max_calls=2)
    report = read_json(result / "canonicalization.json")
    assert report["stop_reason"] == "budget_exhausted" and report["artifact"] is None
    assert "evidence-2" not in read_json(root / "state.json")["tools"]
    assert len(policy.requests) == 2


@pytest.mark.parametrize(
    "bad",
    [
        turn("inspect_image", image_id="../../secret", crop=[0.1, 0.1, 0.8, 0.8]),
        turn("inspect_image", image_id="source:0", crop=[-0.1, 0.1, 0.8, 0.8]),
        turn("render_view", candidate_ids=[0, 1, 99], camera="top"),
        turn("lookup_convention", convention="invented-timer-rule"),
        turn("retrieve_photos", photo_ids=["photo:missing"]),
        turn("finish", question="Unexpected execution argument"),
    ],
)
def test_invalid_action_fails_before_execution(
    bundles: tuple[Path, Path],
    rendering: list[Path],
    tmp_path: Path,
    bad: dict[str, Any],
) -> None:
    root = tmp_path / "job"
    with pytest.raises(ValueError):
        AgentController(FixturePolicy([bad] * 3), OpenGeometryTools("fixture")).run(*bundles, root)
    assert not any(k.startswith("evidence-") for k in read_json(root / "state.json")["tools"])
    assert len(read_json(root / "state.json")["calls"]) == 3
    assert not list(root.rglob("canonical.glb"))


def test_repeated_actions_do_not_consume_tool_attempts(
    bundles: tuple[Path, Path],
    rendering: list[Path],
    tmp_path: Path,
) -> None:
    action = turn("inspect_image", image_id="source:1", crop=[0.1, 0.1, 0.9, 0.9])
    policy = FixturePolicy([action, action, action])
    root = tmp_path / "job"
    with pytest.raises(ValueError, match="Repeated"):
        AgentController(policy, OpenGeometryTools("fixture")).run(*bundles, root)
    assert "evidence-2" not in read_json(root / "state.json")["tools"]


def test_tool_failure_recovers_without_repeating_decision(
    bundles: tuple[Path, Path],
    rendering: list[Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from experiments.canonicalization import agent_controller
    from experiments.canonicalization.agent_tools import execute_evidence

    attempts = []

    def flaky(*args: Any) -> Path:
        attempts.append(True)
        if len(attempts) == 1:
            raise OSError("Render interrupted")
        return execute_evidence(*args)

    monkeypatch.setattr(agent_controller, "execute_evidence", flaky)
    policy = FixturePolicy(
        [turn("inspect_image", image_id="source:1", crop=[0.1, 0.1, 0.9, 0.9]), turn()]
    )
    controller = AgentController(policy, OpenGeometryTools("fixture"))
    root = tmp_path / "job"
    with pytest.raises(OSError, match="interrupted"):
        controller.run(*bundles, root)
    result = controller.run(*bundles, root, resume=True)
    assert (result / "canonical.glb").exists()
    assert len(policy.requests) == 2 and len(attempts) == 2


def test_transport_failure_keeps_count_and_does_not_retry(
    bundles: tuple[Path, Path],
    rendering: list[Path],
    tmp_path: Path,
) -> None:
    policy = FixturePolicy([RuntimeError("Transport failed")])
    controller = AgentController(policy, OpenGeometryTools("fixture"))
    root = tmp_path / "job"
    with pytest.raises(RuntimeError, match="Transport"):
        controller.run(*bundles, root)
    with pytest.raises(RuntimeError, match="automatic retry"):
        controller.run(*bundles, root, resume=True)
    assert len(policy.requests) == 1


def test_capture_copy_failure_can_be_retried_without_new_model_call(
    bundles: tuple[Path, Path],
    rendering: list[Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from experiments.canonicalization import agent_controller
    from experiments.canonicalization.compute import copy_observations

    attempts = []

    def flaky(source: Path, target: Path, case: dict[str, Any]) -> None:
        attempts.append(True)
        if len(attempts) == 1:
            raise OSError("Interrupted capture copy")
        copy_observations(source, target, case)

    policy = FixturePolicy([turn("request_capture", question="Show dial markings"), turn()])
    controller = AgentController(policy, OpenGeometryTools("fixture"))
    root = tmp_path / "job"
    controller.run(*bundles, root)
    bundle = photos(tmp_path / "photos", bundles[0] / "source.glb")
    monkeypatch.setattr(agent_controller, "copy_observations", flaky)
    with pytest.raises(OSError, match="Interrupted"):
        submit_capture(root, bundle)
    submit_capture(root, bundle)
    result = controller.run(*bundles, root, resume=True)
    assert (result / "canonical.glb").is_file()
    assert len(policy.requests) == len(attempts) == 2


def test_submitted_photo_tampering_stops_before_next_model_call(
    bundles: tuple[Path, Path],
    rendering: list[Path],
    tmp_path: Path,
) -> None:
    policy = FixturePolicy([turn("request_capture", question="Show dial markings"), turn()])
    controller = AgentController(policy, OpenGeometryTools("fixture"))
    root = tmp_path / "job"
    controller.run(*bundles, root)
    submit_capture(root, photos(tmp_path / "photos", bundles[0] / "source.glb"))
    state = read_json(root / "state.json")
    copied = root / state["submissions"]["1"]["path"] / "photo.png"
    copied.write_bytes(b"changed")
    with pytest.raises(ValueError, match="modified artifact"):
        controller.run(*bundles, root, resume=True)
    assert len(policy.requests) == 1


def test_evidence_view_failure_does_not_reset_budget_when_captured_twice(
    bundles: tuple[Path, Path],
    rendering: list[Path],
    tmp_path: Path,
) -> None:
    policy = FixturePolicy(
        [
            turn("request_capture", question="Show front dial"),
            turn("request_capture", question="Show back markings"),
        ]
    )
    controller = AgentController(policy, OpenGeometryTools("fixture"))
    root = tmp_path / "job"
    controller.run(*bundles, root, max_calls=2)
    bundle = photos(tmp_path / "photos", bundles[0] / "source.glb")
    submit_capture(root, bundle)
    result = controller.run(*bundles, root, resume=True, max_calls=2)
    report = read_json(result / "canonicalization.json")
    assert report["stop_reason"] == "budget_exhausted" and report["pending_capture"] is None
    with pytest.raises(ValueError):
        submit_capture(root, bundle)
    assert len(policy.requests) == 2


def test_relative_cli_output_accepts_absolute_renderer_result(
    bundles: tuple[Path, Path],
    rendering: list[Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def render(mesh: Path, output: Path, blender: str, **options: Any) -> Path:
        return render_fixture(mesh, output.resolve(), len(options["poses"]), options["poses"])

    monkeypatch.setattr("experiments.canonicalization.agent_tools.render", render)
    monkeypatch.chdir(tmp_path)
    policy = FixturePolicy([turn("render_view", candidate_ids=[0, 1, 2], camera="back"), turn()])
    result = AgentController(policy, OpenGeometryTools("fixture")).run(*bundles, Path("job"))
    assert (result / "canonical.glb").exists()
    record = read_json(result / "canonicalization.json")["action_history"][0]
    assert all(not Path(i["image"]).is_absolute() for i in record["images"])


def test_captured_photo_can_be_retrieved_after_context_eviction(
    bundles: tuple[Path, Path],
    rendering: list[Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def render(mesh: Path, output: Path, blender: str, **options: Any) -> Path:
        return render_fixture(mesh, output, len(options["poses"]), options["poses"])

    monkeypatch.setattr("experiments.canonicalization.agent_tools.render", render)
    policy = FixturePolicy(
        [
            turn("request_capture", question="Show markings around the body"),
            turn("render_view", candidate_ids=[0, 1, 2], camera="back"),
            turn("retrieve_photos", photo_ids=["photo:capture-1-dial-0"]),
            turn(),
        ]
    )
    controller = AgentController(policy, OpenGeometryTools("fixture"))
    root = tmp_path / "job"
    controller.run(*bundles, root, max_calls=4)
    bundle = photos(tmp_path / "photos", bundles[0] / "source.glb")
    manifest = read_json(bundle / "manifest.json")
    first = manifest["images"][0]
    manifest["images"] = [{**first, "id": f"dial-{i}"} for i in range(8)]
    write_json(bundle / "manifest.json", manifest)
    submit_capture(root, bundle)
    result = controller.run(*bundles, root, resume=True, max_calls=4)
    assert "photo:capture-1-dial-0" not in dict(policy.requests[2].images)
    assert "photo:capture-1-dial-0" in dict(policy.requests[3].images)
    assert (result / "canonical.glb").exists()
    assert len(policy.requests) == 4 and all(len(r.images) <= 32 for r in policy.requests)


def test_trace_distinguishes_execution_replay_and_rejection(
    bundles: tuple[Path, Path], rendering: list[Path], tmp_path: Path
) -> None:
    from experiments.canonicalization.agent_report import summarize, write_report
    from experiments.canonicalization.trace import read_events

    policy = FixturePolicy(
        [
            turn("inspect_image", image_id="candidate:0", crop=[0.1, 0.1, 0.9, 0.9]),
            turn(),
        ]
    )
    controller = AgentController(policy, OpenGeometryTools("fixture"))
    root = tmp_path / "logged"
    controller.run(*bundles, root)
    controller.run(*bundles, root, resume=True)
    summary = summarize(root)
    assert summary["model_calls"] == 2
    assert len(summary["executed_evidence_actions"]) == 1
    assert summary["scope"] == "full_frame"
    records, malformed = read_events(root)
    assert malformed == 0
    assert sum(e["event"] == "model_dispatched" for e in records) == 2
    assert sum(e["event"] == "model_response_reused" for e in records) == 2
    assert "inspect_image" in write_report(root).read_text()

    rejected = tmp_path / "rejected"
    policy = FixturePolicy([turn(choice=pose(evidence_ids=["candidate:0"]))] * 3)
    with pytest.raises(ValueError):
        AgentController(policy, OpenGeometryTools("fixture")).run(*bundles, rejected)
    summary = summarize(rejected)
    assert summary["status"] == "failed" and summary["result"] is None
    assert not summary["executed_evidence_actions"]
    assert any(e["event"] == "agent_decision_rejected" for e in summary["failures"])


def test_transport_trace_keeps_budget_without_logging_exception_secrets(
    bundles: tuple[Path, Path],
    rendering: list[Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from experiments.canonicalization.agent_report import summarize
    from experiments.canonicalization.model_policy import PolicyRequest

    policy = FixturePolicy([])

    def fail(request: PolicyRequest) -> dict[str, Any]:
        raise TimeoutError("private-token-not-for-logs")

    monkeypatch.setattr(policy, "select", fail)
    root = tmp_path / "transport"
    controller = AgentController(policy, OpenGeometryTools("fixture"))
    with pytest.raises(TimeoutError):
        controller.run(*bundles, root)
    summary = summarize(root)
    assert summary["model_calls"] == 1
    assert summary["failures"][-1]["failure_kind"] == "transport"
    assert "private-token-not-for-logs" not in (root / "events.jsonl").read_text()
    with pytest.raises(RuntimeError, match="automatic retry is disabled"):
        controller.run(*bundles, root, resume=True)
    assert summarize(root)["model_calls"] == 1


def test_original_photos_are_visible_on_first_call_and_remain_visible(
    bundles: tuple[Path, Path], rendering: list[Path], tmp_path: Path
) -> None:
    bundle = photos(tmp_path / "photos", bundles[0] / "source.glb")
    policy = FixturePolicy(
        [
            turn("inspect_image", image_id="photo:dial", crop=[0.1, 0.1, 0.9, 0.9]),
            turn(
                choice=pose(
                    evidence_ids=["candidate:0", "candidate:1", "photo:dial", "photo:crop-1"],
                    mesh_correspondence="Fixture dial matches the model's visible dial",
                )
            ),
        ]
    )
    result = AgentController(policy, OpenGeometryTools("fixture")).run(
        *bundles,
        tmp_path / "job",
        observations=bundle,
    )
    assert all("photo:dial" in dict(r.images) for r in policy.requests)
    assert all(len(r.images) <= 32 for r in policy.requests)
    assert "mesh_appearance" in policy.requests[0].prompt
    report = read_json(result / "canonicalization.json")
    assert report["appearance"]["has_color_evidence"]
    assert report["artifact"]["appearance_preserved"]


def test_gray_input_requires_explicit_baseline_before_any_model_call(
    bundles: tuple[Path, Path], rendering: list[Path], tmp_path: Path
) -> None:
    import trimesh

    views, predictions = bundles
    path = views / "source.glb"
    mesh = trimesh.load_mesh(path)
    mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=[175, 175, 175, 255])
    mesh.export(path)
    manifest = read_json(views / "manifest.json")
    manifest["mesh_sha256"] = sha256(path)
    write_json(views / "manifest.json", manifest)
    inference = read_json(predictions / "predictions.json")
    inference["views_manifest_sha256"] = sha256(views / "manifest.json")
    write_json(predictions / "predictions.json", inference)
    policy = FixturePolicy([turn()])
    with pytest.raises(ValueError, match="no color/texture"):
        AgentController(policy, OpenGeometryTools("fixture")).run(*bundles, tmp_path / "rejected")
    assert not policy.requests and not rendering
    result = AgentController(policy, OpenGeometryTools("fixture"), allow_shape_only=True).run(
        *bundles,
        tmp_path / "baseline",
    )
    assert not read_json(result / "canonicalization.json")["appearance"]["has_color_evidence"]


def test_initial_photos_survive_supplementary_eviction_with_32_image_cap(
    bundles: tuple[Path, Path],
    rendering: list[Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import numpy as np

    original_views, predictions = bundles
    views = render_fixture(original_views / "source.glb", tmp_path / "expanded", 12)
    inference = read_json(predictions / "predictions.json")
    inference["views_manifest_sha256"] = sha256(views / "manifest.json")
    inference["predictions"] = [
        {"id": i, "angles_degrees": [i * 30, 0, 0], "symmetry_fit": 1} for i in range(12)
    ]
    logits: dict[str, Any] = {f"view_{i:02d}": np.zeros(900) for i in range(12)}
    np.savez(predictions / "logits.npz", **logits)
    inference["logits_sha256"] = sha256(predictions / "logits.npz")
    write_json(predictions / "predictions.json", inference)
    bundle = photos(tmp_path / "photos", views / "source.glb")
    manifest = read_json(bundle / "manifest.json")
    manifest["images"] = [{**manifest["images"][0], "id": f"p{i}"} for i in range(8)]
    write_json(bundle / "manifest.json", manifest)

    def render(mesh: Path, output: Path, blender: str, **options: Any) -> Path:
        return render_fixture(mesh, output, len(options["poses"]), options["poses"])

    monkeypatch.setattr("experiments.canonicalization.agent_tools.render", render)
    policy = FixturePolicy(
        [
            turn("retrieve_photos", photo_ids=[f"photo:p{i}" for i in range(2, 6)]),
            turn("render_view", candidate_ids=[0, 1, 2], camera="top"),
            turn("inspect_image", image_id="photo:p5", crop=[0.1, 0.1, 0.9, 0.9]),
            turn(),
        ]
    )
    AgentController(policy, OpenGeometryTools("fixture")).run(
        views,
        predictions,
        tmp_path / "job",
        observations=bundle,
        max_calls=4,
    )
    assert max(len(r.images) for r in policy.requests) == 32
    for request in policy.requests:
        assert {"photo:p0", "photo:p1"} <= dict(request.images).keys()
        assert len(request.images) == len(dict(request.images))
    assert "photo:p2" not in dict(policy.requests[-1].images)


def test_growing_evidence_uses_context_labels_without_large_nested_enums(
    bundles: tuple[Path, Path], rendering: list[Path], tmp_path: Path
) -> None:
    import json

    from experiments.canonicalization.agent_controller import agent_request

    images = [(f"detail:{i}:1:top", bundles[0] / "view-0.png") for i in range(32)]
    request = agent_request(
        images,
        {"candidates": [{"id": i} for i in range(3)], "appearance": {}, "omitted_hypotheses": 0},
        [],
        [],
        2,
    )
    assert request.response_schema is not None
    schema = json.dumps(request.response_schema)
    assert "detail:31:1:top" not in schema
    assert "detail:31:1:top" in request.prompt
    # Small action/candidate enums remain constrained in the wire schema.
    assert '"finish"' in schema and '"enum": [0, 1, 2]' in schema
    policy = FixturePolicy([turn(choice=pose(evidence_ids=["candidate:0", "not-visible"]))] * 3)
    with pytest.raises(ValueError, match="supplied image labels"):
        AgentController(policy, OpenGeometryTools("fixture")).run(*bundles, tmp_path / "invalid")
    assert not list((tmp_path / "invalid").rglob("canonical.glb"))


def test_quota_switch_and_evidence_repair_share_budget_and_replay(
    bundles: tuple[Path, Path], rendering: list[Path], tmp_path: Path
) -> None:
    from experiments.canonicalization.model_policy import PolicyRateLimited
    from experiments.canonicalization.trace import read_events

    first = FixturePolicy([PolicyRateLimited("HTTP 429")])
    first.identity = {"backend": "test_fixture", "model": "first"}
    second = FixturePolicy([turn(choice=pose(evidence_ids=["candidate:0"])), turn()])
    second.identity = {"backend": "test_fixture", "model": "second"}
    controller = AgentController(first, OpenGeometryTools("fixture"), fallback_policies=(second,))
    root = tmp_path / "job"
    result = controller.run(*bundles, root, max_calls=3)
    report = read_json(result / "canonicalization.json")
    assert report["model_calls"] == 3 and report["artifact"] is not None
    assert report["decision_policy"]["model"] == "second"
    assert "compare the selected candidate" in second.requests[1].prompt
    assert "validation_error" in second.requests[1].prompt
    assert "do not invent" in second.requests[1].prompt
    assert controller.run(*bundles, root, resume=True, max_calls=3) == result
    assert len(first.requests) == 1 and len(second.requests) == 2
    assert len(rendering) == 1
    events, _ = read_events(root)
    assert [e["model"] for e in events if e["event"] == "model_dispatched"] == [
        "first",
        "second",
        "second",
    ]
    assert read_json(root / "state.json")["calls"]["agent-1"]["status"] == "rate_limited"
    # A different model pool must not reuse this job's requests under another identity.
    changed = AgentController(first, OpenGeometryTools("fixture"))
    with pytest.raises(ValueError, match="Resume inputs"):
        changed.run(*bundles, root, resume=True, max_calls=3)


def test_switch_keeps_completed_tool_evidence(
    bundles: tuple[Path, Path], rendering: list[Path], tmp_path: Path
) -> None:
    from experiments.canonicalization.model_policy import PolicyRateLimited

    first = FixturePolicy(
        [
            turn("inspect_image", image_id="candidate:0", crop=[0.1, 0.1, 0.9, 0.9]),
            PolicyRateLimited("HTTP 429"),
        ]
    )
    second = FixturePolicy([turn()])
    controller = AgentController(first, OpenGeometryTools("fixture"), fallback_policies=(second,))
    root = tmp_path / "job"
    result = controller.run(*bundles, root)
    assert "crop:crop-1" in dict(second.requests[0].images)
    assert "inspect_image" in second.requests[0].prompt
    assert read_json(result / "canonicalization.json")["model_calls"] == 3
    assert controller.run(*bundles, root, resume=True) == result
    assert len(first.requests) == 2 and len(second.requests) == 1


@pytest.mark.parametrize("pool_size", [2, 3])
def test_quota_exhaustion_never_cycles_or_resets_budget(
    bundles: tuple[Path, Path], rendering: list[Path], tmp_path: Path, pool_size: int
) -> None:
    from experiments.canonicalization.model_policy import PolicyRateLimited
    from experiments.canonicalization.trace import read_events

    policies = [FixturePolicy([PolicyRateLimited("HTTP 429")]) for _ in range(pool_size)]
    controller = AgentController(
        policies[0], OpenGeometryTools("fixture"), fallback_policies=tuple(policies[1:])
    )
    root = tmp_path / "job"
    budget = 3 if pool_size == 2 else 2
    for resume in (False, True):
        with pytest.raises(PolicyRateLimited):
            controller.run(*bundles, root, resume=resume, max_calls=budget)
    assert sum(len(p.requests) for p in policies) == 2
    assert not list(root.rglob("canonical.glb"))
    events, _ = read_events(root)
    stops = [e for e in events if e["event"] == "model_fallback_stopped"]
    assert stops[-1]["reason"] == (
        "model_list_exhausted" if pool_size == 2 else "call_budget_exhausted"
    )


@pytest.mark.parametrize(
    "error", [RuntimeError("HTTP 403"), RuntimeError("HTTP 400"), TimeoutError()]
)
def test_nonquota_errors_do_not_switch_models(
    bundles: tuple[Path, Path], rendering: list[Path], tmp_path: Path, error: Exception
) -> None:
    first, second = FixturePolicy([error]), FixturePolicy([turn()])
    controller = AgentController(first, OpenGeometryTools("fixture"), fallback_policies=(second,))
    with pytest.raises(type(error)):
        controller.run(*bundles, tmp_path / "job")
    assert len(first.requests) == 1 and not second.requests


def test_schema_repair_is_bounded_and_preserves_invalid_reply(
    bundles: tuple[Path, Path], rendering: list[Path], tmp_path: Path
) -> None:
    policy = FixturePolicy([{"unrecognized": "private-invalid-input"}, turn()])
    controller = AgentController(policy, OpenGeometryTools("fixture"))
    root = tmp_path / "job"
    result = controller.run(*bundles, root, max_calls=2)
    assert read_json(result / "canonicalization.json")["model_calls"] == 2
    assert "required JSON schema" in policy.requests[1].prompt
    assert "private-invalid-input" not in policy.requests[1].prompt
    assert "private-invalid-input" in (root / "calls/agent-1.json").read_text()
    assert controller.run(*bundles, root, resume=True, max_calls=2) == result
    assert len(policy.requests) == 2


def test_agent_has_no_category_menu_or_lookup_and_cannot_resume_old_route(
    bundles: tuple[Path, Path], rendering: list[Path], tmp_path: Path
) -> None:
    import json

    policy = FixturePolicy([turn()])
    controller = AgentController(policy, OpenGeometryTools("fixture"))
    root = tmp_path / "job"
    controller.run(*bundles, root)
    request = policy.requests[0]
    assert request.response_schema is not None
    action = request.response_schema["$defs"]["EvidenceAction"]["properties"]
    assert "lookup_convention" not in action["kind"]["enum"]
    assert "convention" not in action
    context = json.loads(request.prompt.split("\n")[-1])
    assert "available_conventions" not in context
    assert "category" not in context
    saved = read_json(root / "request.json")
    saved["route"] = "oa-evidence-agent-v1"
    write_json(root / "request.json", saved)
    with pytest.raises(ValueError, match="Resume inputs"):
        controller.run(*bundles, root, resume=True)
    assert len(policy.requests) == 1

"""Open route behavior with explicit model/render fixtures and real rigid exports."""

import io
import json
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

np = pytest.importorskip("numpy")
trimesh = pytest.importorskip("trimesh")

from experiments.artifacts import read_json, sha256, write_json  # noqa: E402
from experiments.canonicalization.gemini_policy import GeminiPolicy  # noqa: E402
from experiments.canonicalization.model_policy import PolicyRequest  # noqa: E402
from experiments.canonicalization.oa_proposals import proposals  # noqa: E402
from experiments.canonicalization.open_contracts import ObjectSemantics  # noqa: E402
from experiments.canonicalization.open_controller import OpenController  # noqa: E402
from experiments.canonicalization.open_tools import OpenGeometryTools  # noqa: E402
from experiments.canonicalization.responses_policy import ResponsesPolicy  # noqa: E402
from experiments.orientation.geometry import OA_TO_CANONICAL, mesh_to_oa  # noqa: E402
from tests.canonical_fixtures import (  # noqa: E402
    FixturePolicy,
    identity,
    pose,
)
from tests.canonical_fixtures import (  # noqa: E402
    bundles as bundles,
)
from tests.canonical_fixtures import (  # noqa: E402
    rendering as rendering,
)


def test_unknown_category_export_preserves_color_geometry_and_resume(
    bundles: tuple[Path, Path],
    rendering: list[Path],
    tmp_path: Path,
) -> None:
    policy = FixturePolicy([identity(), pose()])
    controller = OpenController(policy, OpenGeometryTools("fixture"))
    root = tmp_path / "job"
    result = controller.run(*bundles, root)
    report = read_json(result / "canonicalization.json")
    assert report["semantics"]["hypotheses"][0]["functional_identity"] == "kitchen timer"
    assert report["scope"] == "full_frame" and not report["dataset_approved"]
    assert report["artifact"]["inverse_vertex_error"] < report["artifact"]["tolerance"]
    original = trimesh.load_scene(bundles[0] / "source.glb").to_mesh()
    exported = trimesh.load_scene(result / "canonical.glb").to_mesh()
    np.testing.assert_array_equal(original.visual.vertex_colors, exported.visual.vertex_colors)
    assert controller.run(*bundles, root, resume=True) == result
    assert len(policy.requests) == 2 and len(rendering) == 1
    assert [r.task for r in policy.requests] == ["open_identity", "open_pose"]
    assert "hypotheses" in (policy.requests[0].response_schema or {})["properties"]
    pose_request = policy.requests[1]
    assert "source:0" not in pose_request.prompt
    pose_schema = pose_request.response_schema
    assert pose_schema is not None
    assert set(pose_schema["properties"]["evidence_ids"]["items"]["enum"]) == {
        label for label, _ in pose_request.images
    }
    assert sha256(bundles[0] / "source.glb") == sha256(root / "work/inputs-1/views/source.glb")


def test_oa_coordinate_conversion_and_symmetry_alternatives(bundles: tuple[Path, Path]) -> None:
    views, predictions = bundles
    report = read_json(predictions / "predictions.json")
    report["predictions"][0]["symmetry_fit"] = 2
    report["predictions"][1]["symmetry_fit"] = 0
    write_json(predictions / "predictions.json", report)
    pool = proposals(views, predictions)["candidates"]
    pair = [c for c in pool if c["source_view"] == 0]
    assert len(pair) == 2
    base = OA_TO_CANONICAL @ mesh_to_oa([0, 0, 0], np.eye(3))
    np.testing.assert_allclose(pair[0]["rotation"], base)
    np.testing.assert_allclose(np.array(pair[0]["rotation"])[1], np.array(pair[1]["rotation"])[1])
    np.testing.assert_allclose(
        np.array(pair[0]["rotation"])[2], -np.array(pair[1]["rotation"])[2], atol=1e-15
    )
    assert any(c["symmetry_fit"] == 0 and not c["front_available"] for c in pool)


def test_uncertain_identity_stops_without_export(
    bundles: tuple[Path, Path],
    rendering: list[Path],
    tmp_path: Path,
) -> None:
    policy = FixturePolicy(
        [identity(status="uncertain", missing_evidence="Show the dial markings")]
    )
    result = OpenController(policy, OpenGeometryTools("fixture")).run(*bundles, tmp_path / "job")
    report = read_json(result / "canonicalization.json")
    assert report["status"] == "needs_evidence" and report["artifact"] is None
    assert not (result / "canonical.glb").exists()
    assert len(policy.requests) == 1 and not rendering


@pytest.mark.parametrize("budget", [2, 3])
def test_ambiguity_gets_one_bounded_opposite_round(
    bundles: tuple[Path, Path],
    rendering: list[Path],
    tmp_path: Path,
    budget: int,
) -> None:
    policy = FixturePolicy(
        [
            identity(),
            pose(scope="unresolved", candidate_id=None, missing_evidence="Show back"),
            pose(evidence_ids=["candidate:0", "opposite:0"]),
        ]
    )
    controller = OpenController(policy, OpenGeometryTools("fixture"))
    root = tmp_path / "job"
    result = controller.run(*bundles, root, max_calls=budget)
    report = read_json(result / "canonicalization.json")
    assert report["scope"] == ("full_frame" if budget == 3 else "unresolved")
    assert len(policy.requests) == budget and len(rendering) == budget - 1
    assert controller.run(*bundles, root, resume=True, max_calls=budget) == result
    assert len(policy.requests) == budget
    assert all(len(r.images) <= 32 for r in policy.requests)


@pytest.mark.parametrize(
    "change",
    [
        {"candidate_id": 99},
        {"evidence_ids": ["candidate:0", "missing:1"]},
        {"evidence_ids": ["candidate:0"]},
        {"candidate_id": True},
        {"rotation": [[1, 0, 0]]},
    ],
)
def test_invalid_policy_cannot_export(
    bundles: tuple[Path, Path],
    rendering: list[Path],
    tmp_path: Path,
    change: dict[str, Any],
) -> None:
    policy = FixturePolicy([identity(), pose(**change)])
    root = tmp_path / "job"
    with pytest.raises(ValueError):
        OpenController(policy, OpenGeometryTools("fixture")).run(*bundles, root)
    assert not list(root.rglob("canonical.glb"))
    assert (root / "calls/pose-1.json").is_file()


@pytest.mark.parametrize("scope", ["full_frame", "up_only"])
def test_unspecified_front_never_becomes_full_frame(
    bundles: tuple[Path, Path],
    rendering: list[Path],
    tmp_path: Path,
    scope: str,
) -> None:
    policy = FixturePolicy([identity(front_mode="unspecified"), pose(scope=scope)])
    controller = OpenController(policy, OpenGeometryTools("fixture"))
    if scope == "full_frame":
        with pytest.raises(ValueError, match="unique semantic front"):
            controller.run(*bundles, tmp_path / "job")
    else:
        result = controller.run(*bundles, tmp_path / "job")
        assert read_json(result / "canonicalization.json")["yaw_status"] == "unspecified"


def test_failed_transport_is_counted_without_automatic_retry(
    bundles: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    policy = FixturePolicy([RuntimeError("transport failure")])
    controller = OpenController(policy, OpenGeometryTools("fixture"))
    root = tmp_path / "job"
    with pytest.raises(RuntimeError, match="transport failure"):
        controller.run(*bundles, root)
    with pytest.raises(RuntimeError, match="automatic retry is disabled"):
        controller.run(*bundles, root, resume=True)
    assert len(policy.requests) == 1


def test_export_failure_recovers_without_repeating_model_calls(
    bundles: tuple[Path, Path],
    rendering: list[Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from experiments.canonicalization import open_controller
    from experiments.canonicalization.workflow import export

    original = export
    attempts = []

    def flaky_export(*args: Any) -> dict[str, Any]:
        attempts.append(True)
        if len(attempts) == 1:
            raise OSError("Interrupted export")
        return original(*args)

    monkeypatch.setattr(open_controller, "export", flaky_export)
    policy = FixturePolicy([identity(), pose()])
    controller = OpenController(policy, OpenGeometryTools("fixture"))
    root = tmp_path / "job"
    with pytest.raises(OSError, match="Interrupted"):
        controller.run(*bundles, root)
    result = controller.run(*bundles, root, resume=True)
    assert (result / "canonical.glb").is_file()
    assert result.name == "export-2" and len(policy.requests) == 2


@pytest.mark.parametrize("target", ["image", "logits", "saved_response"])
def test_modified_evidence_or_response_is_rejected(
    bundles: tuple[Path, Path],
    rendering: list[Path],
    tmp_path: Path,
    target: str,
) -> None:
    policy = FixturePolicy([identity(), pose()])
    controller = OpenController(policy, OpenGeometryTools("fixture"))
    root = tmp_path / "job"
    if target == "saved_response":
        controller.run(*bundles, root)
        (root / "calls/pose-1.json").write_text("{}")
    elif target == "image":
        (bundles[0] / "view-0.png").write_bytes(b"changed")
    else:
        (bundles[1] / "logits.npz").write_bytes(b"changed")
    before = len(policy.requests)
    with pytest.raises(ValueError):
        controller.run(*bundles, root, resume=True)
    assert len(policy.requests) == before


@pytest.mark.parametrize("backend", ["gemini", "responses"])
def test_cloud_transports_receive_open_schema_without_mutating_it(
    bundles: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
) -> None:
    schema = ObjectSemantics.model_json_schema()
    captured = []

    def send(request: Any, timeout: float) -> io.BytesIO:
        payload = json.loads(request.data)
        captured.append(payload)
        result: dict[str, Any]
        if backend == "gemini":
            received = payload["generationConfig"]["responseFormat"]["text"]["schema"]
            result = {
                "candidates": [
                    {
                        "finishReason": "STOP",
                        "content": {"parts": [{"text": json.dumps(identity())}]},
                    }
                ]
            }
        else:
            received = payload["text"]["format"]["schema"]
            result = {
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": json.dumps(identity())}],
                    }
                ],
            }
        assert "hypotheses" in received["properties"]
        return io.BytesIO(json.dumps(result).encode())

    monkeypatch.setattr("urllib.request.urlopen", send)
    policy = (
        GeminiPolicy("gemini-test", "fixture")
        if backend == "gemini"
        else ResponsesPolicy("test", "fixture")
    )
    reply = policy.select(
        PolicyRequest(
            "open_identity",
            "fixture",
            tuple((f"source:{i}", bundles[0] / f"view-{i}.png") for i in range(3)),
            schema,
        )
    )
    assert ObjectSemantics.model_validate_json(reply["raw_text"]).status == "resolved"
    assert schema == ObjectSemantics.model_json_schema() and len(captured) == 1


@pytest.mark.parametrize("scope", ["full_frame", "up_only"])
def test_oa_zero_fit_cannot_claim_yaw(
    bundles: tuple[Path, Path],
    rendering: list[Path],
    tmp_path: Path,
    scope: str,
) -> None:
    source = bundles[1] / "predictions.json"
    raw = read_json(source)
    for prediction in raw["predictions"]:
        prediction["symmetry_fit"] = 0
    write_json(source, raw)
    controller = OpenController(
        FixturePolicy([identity(), pose(scope=scope)]), OpenGeometryTools("fixture")
    )
    if scope == "full_frame":
        with pytest.raises(ValueError, match="OA yaw candidate"):
            controller.run(*bundles, tmp_path / "job")
    else:
        result = controller.run(*bundles, tmp_path / "job")
        assert read_json(result / "canonicalization.json")["yaw_status"] == "unspecified"


@pytest.mark.parametrize("correspondence", ["", "The same dial shape appears in photo and mesh"])
def test_original_photos_are_bound_evidence_not_category_labels(
    bundles: tuple[Path, Path],
    rendering: list[Path],
    tmp_path: Path,
    correspondence: str,
) -> None:
    observations = tmp_path / "photos"
    observations.mkdir()
    image = observations / "observation.png"
    Image.new("RGB", (16, 16), "red").save(image)
    write_json(
        observations / "manifest.json",
        {
            "schema_version": 1,
            "status": "succeeded",
            "registration": "unregistered",
            "mesh_sha256": sha256(bundles[0] / "source.glb"),
            "category": "untrusted-metadata-label",
            "assets": [],
            "images": [
                {
                    "id": "user-1",
                    "image": image.name,
                    "sha256": sha256(image),
                    "kind": "original_photo",
                }
            ],
        },
    )
    policy = FixturePolicy(
        [
            identity(),
            pose(
                evidence_ids=["candidate:0", "candidate:1", "photo:user-1"],
                mesh_correspondence=correspondence,
            ),
        ]
    )
    controller = OpenController(policy, OpenGeometryTools("fixture"))
    if not correspondence:
        with pytest.raises(ValueError, match="mesh correspondence"):
            controller.run(*bundles, tmp_path / "job", observations=observations)
    else:
        result = controller.run(*bundles, tmp_path / "job", observations=observations)
        assert (result / "canonical.glb").exists()
    assert all("untrusted-metadata-label" not in r.prompt for r in policy.requests)
    assert all("photo:user-1" in dict(r.images) for r in policy.requests)


def test_textured_mesh_retains_uv_and_embedded_image(
    bundles: tuple[Path, Path],
    rendering: list[Path],
    tmp_path: Path,
) -> None:
    mesh_path = bundles[0] / "source.glb"
    mesh = trimesh.creation.box()
    texture = Image.new("RGB", (8, 8), "green")
    mesh.visual = trimesh.visual.TextureVisuals(
        uv=np.full((len(mesh.vertices), 2), 0.5),
        image=texture,
    )
    mesh.export(mesh_path)
    manifest = read_json(bundles[0] / "manifest.json")
    manifest["mesh_sha256"] = sha256(mesh_path)
    write_json(bundles[0] / "manifest.json", manifest)
    predictions = read_json(bundles[1] / "predictions.json")
    predictions["views_manifest_sha256"] = sha256(bundles[0] / "manifest.json")
    write_json(bundles[1] / "predictions.json", predictions)
    result = OpenController(FixturePolicy([identity(), pose()]), OpenGeometryTools("fixture")).run(
        *bundles, tmp_path / "job"
    )
    original = next(iter(trimesh.load_scene(mesh_path).geometry.values())).visual
    exported = next(iter(trimesh.load_scene(result / "canonical.glb").geometry.values())).visual
    np.testing.assert_array_equal(original.uv, exported.uv)
    np.testing.assert_array_equal(
        original.material.baseColorTexture, exported.material.baseColorTexture
    )


def test_proposal_method_cannot_change_on_resume(
    bundles: tuple[Path, Path],
    rendering: list[Path],
    tmp_path: Path,
) -> None:
    policy = FixturePolicy([identity(status="uncertain", missing_evidence="Show the controls")])
    root = tmp_path / "job"
    OpenController(policy, OpenGeometryTools("fixture")).run(*bundles, root)
    with pytest.raises(ValueError, match="Resume inputs"):
        OpenController(policy, OpenGeometryTools("fixture", "distribution")).run(
            *bundles, root, resume=True
        )
    assert len(policy.requests) == 1


def test_flat_distributions_stop_before_render_or_model_call(
    bundles: tuple[Path, Path],
    rendering: list[Path],
    tmp_path: Path,
) -> None:
    tools = OpenGeometryTools("fixture", "distribution")
    snapshot = tools.snapshot(*bundles, None, tmp_path / "snapshot")
    with pytest.raises(ValueError, match="Insufficient distinct"):
        tools.prepare(snapshot, tmp_path / "candidates")
    report = read_json(tmp_path / "candidates/candidates.json")
    assert report["reason"] == "all_marginals_flat"
    assert report["candidates"] == []
    assert not rendering

"""Import real worker artifact shapes using explicit synthetic model/render fixtures."""

from pathlib import Path

import pytest

pytest.importorskip("numpy")

from experiments.artifacts import read_json, sha256, write_json  # noqa: E402
from experiments.canonicalization.agent_controller import AgentController  # noqa: E402
from experiments.canonicalization.open_tools import OpenGeometryTools  # noqa: E402
from experiments.library_import import import_job  # noqa: E402
from scan2canon.runtime import build_runtime  # noqa: E402
from tests.canonical_fixtures import FixturePolicy  # noqa: E402
from tests.canonical_fixtures import bundles as bundles  # noqa: E402
from tests.canonical_fixtures import rendering as rendering  # noqa: E402
from tests.test_canonicalization_agent import turn  # noqa: E402


def test_completed_job_import_is_idempotent_and_retains_provenance(
    bundles: tuple[Path, Path], rendering: list[Path], tmp_path: Path
) -> None:
    reply = turn()
    reply["directions"]["hypotheses"][1]["role"] = "model_reference"
    policy = FixturePolicy([reply])
    policy.identity = {**policy.identity, "implementation_sha256": "a" * 64}
    job = tmp_path / "job"
    AgentController(policy, OpenGeometryTools("fixture")).run(*bundles, job, max_calls=2)
    library = build_runtime(tmp_path / "db").library
    assessment = import_job(library, job, "Fixture")
    assert import_job(library, job, "Fixture") == assessment
    assert len(library.objects()) == 1
    assert len(library.assessments(assessment.object_id)) == 1
    assert library.matches(assessment.object_id) == []
    assert assessment.model == "scripted-not-inference"
    assert {e.kind for e in assessment.evidence} == {
        "mesh",
        "image",
        "report",
        "request",
        "model_reply",
    }
    assert all(Path(e.location).is_file() for e in assessment.evidence)
    assert assessment.state == "proposed"
    assert assessment.directions[1].role == "model_reference"
    report_ref = next(e for e in assessment.evidence if e.kind == "report")
    assert (
        read_json(Path(report_ref.location))["orientation_reference"][
            "category_annotation_verified"
        ]
        is False
    )
    image = next(e for e in assessment.evidence if e.kind == "image")
    original = Path(image.location).read_bytes()
    Path(image.location).write_bytes(b"tampered")
    other = build_runtime(tmp_path / "empty-db").library
    with pytest.raises(ValueError, match="modified"):
        import_job(other, job, "Fixture")
    assert other.objects() == []
    Path(image.location).write_bytes(original)
    assert import_job(other, job, "Fixture") == assessment
    assert len(policy.requests) == 1  # imports and retries never dispatch a model


def test_pending_job_cannot_become_library_knowledge(tmp_path: Path) -> None:
    job = tmp_path / "pending"
    job.mkdir()
    write_json(job / "state.json", {"status": "needs_evidence"})
    library = build_runtime(tmp_path / "db").library
    with pytest.raises(ValueError, match="completed"):
        import_job(library, job, "Not classified")
    assert library.objects() == []
    assert read_json(job / "state.json") == {"status": "needs_evidence"}


def test_import_rejects_changed_oa_authority_despite_valid_artifact_hash(
    bundles: tuple[Path, Path], rendering: list[Path], tmp_path: Path
) -> None:
    policy = FixturePolicy([turn()])
    policy.identity = {**policy.identity, "implementation_sha256": "a" * 64}
    job = tmp_path / "job"
    result = AgentController(policy, OpenGeometryTools("fixture")).run(*bundles, job, max_calls=2)
    path = result / "canonicalization.json"
    report = read_json(path)
    report["orientation_reference"]["category_annotation_verified"] = True
    write_json(path, report)
    state = read_json(job / "state.json")
    for stage, attempts in state["tools"].items():
        if stage.startswith("agent-export-"):
            attempts[-1]["files"]["canonicalization.json"] = sha256(path)
    write_json(job / "state.json", state)
    library = build_runtime(tmp_path / "db").library
    with pytest.raises(ValueError, match="OA reference"):
        import_job(library, job, "Fixture")
    assert library.objects() == []

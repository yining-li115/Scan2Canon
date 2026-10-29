"""Transactional proposal storage and explicit cross-instance relationships."""

import sqlite3
from pathlib import Path
from uuid import uuid4

import pytest

from scan2canon.library import (
    Assessment,
    AxisCorrespondence,
    DirectionClaim,
    EvidenceRef,
    IdentityClaim,
    MatchProposal,
    ObjectRecord,
)
from scan2canon.runtime import build_runtime
from scan2canon.storage.library import SQLiteLibrary


def fixture_entry() -> tuple[ObjectRecord, Assessment]:
    item = ObjectRecord(id=uuid4(), name="Synthetic fixture", source_mesh_sha256="a" * 64)
    refs = (
        EvidenceRef(id="mesh", location="fixture.glb", sha256="a" * 64, kind="mesh"),
        EvidenceRef(id="report", location="fixture.json", sha256="b" * 64, kind="report"),
        EvidenceRef(id="image", location="fixture.png", sha256="c" * 64, kind="image"),
    )
    assessment = Assessment(
        id=uuid4(),
        object_id=item.id,
        producer="test-fixture",
        model="not-inference",
        implementation_sha256="d" * 64,
        report_sha256="b" * 64,
        evidence=refs,
        identities=(
            IdentityClaim(
                functional_identity="fixture",
                appearance="synthetic",
                parts=("start", "end"),
                evidence_ids=("image",),
            ),
        ),
        directions=(
            DirectionClaim(
                id="up",
                axis="up",
                role="upright",
                description="fixture",
                from_part="start",
                to_part="end",
                evidence_ids=("image",),
            ),
        ),
        semantic_status="resolved",
        pose_scope="up_only",
        selected_up="up",
        selected_front=None,
    )
    return item, assessment


def test_persist_reopen_idempotency_and_no_automatic_classification(tmp_path: Path) -> None:
    library = build_runtime(tmp_path).library
    item, assessment = fixture_entry()
    library.ingest(item, assessment)
    library.ingest(item, assessment)
    reopened = build_runtime(tmp_path).library
    assert reopened.objects() == [item]
    assert reopened.assessments(item.id) == [assessment]
    assert reopened.matches(item.id) == []
    assert reopened.get_assessment(assessment.id).state == "proposed"
    with pytest.raises(ValueError, match="immutable"):
        reopened.ingest(item.model_copy(update={"name": "overwrite"}), assessment)
    assert reopened.get_object(item.id) == item


def test_conflict_rolls_back_object_insert_and_allows_clean_retry(tmp_path: Path) -> None:
    library = SQLiteLibrary(tmp_path / "library.db")
    item, assessment = fixture_entry()
    library.ingest(item, assessment)
    second = item.model_copy(update={"id": uuid4()})
    with pytest.raises(ValueError, match="conflict"):
        library.ingest(second, assessment.model_copy(update={"object_id": second.id}))
    assert library.objects() == [item]
    repaired = assessment.model_copy(update={"id": uuid4(), "object_id": second.id})
    library.ingest(second, repaired)
    assert len(library.objects()) == 2
    assert library.get_assessment(assessment.id) == assessment


def test_explicit_axis_match_references_versions_and_remains_a_proposal(tmp_path: Path) -> None:
    library = SQLiteLibrary(tmp_path / "library.db")
    left, a = fixture_entry()
    right, b = fixture_entry()
    library.ingest(left, a)
    library.ingest(right, b)
    match = MatchProposal(
        id=uuid4(),
        left_assessment_id=a.id,
        right_assessment_id=b.id,
        identity_relation="uncertain",
        correspondences=(
            AxisCorrespondence(
                axis="up",
                left_direction_id="up",
                right_direction_id="up",
                rationale="Synthetic correspondence",
            ),
        ),
        evidence=a.evidence[1],
        producer="test-fixture",
        rationale="Test only, no learned claim",
    )
    library.propose_match(match)
    library.propose_match(match)
    assert library.matches(left.id) == library.matches(right.id) == [match]
    bad = match.model_copy(
        update={
            "id": uuid4(),
            "correspondences": (match.correspondences[0].model_copy(update={"axis": "front"}),),
        }
    )
    with pytest.raises(ValueError, match="wrong-axis"):
        library.propose_match(bad)
    with pytest.raises(KeyError):
        library.propose_match(
            match.model_copy(update={"id": uuid4(), "right_assessment_id": uuid4()})
        )
    assert library.matches(left.id) == [match]
    assert match.state == "proposed"


def test_revalidate_unchecked_models_and_missing_evidence(tmp_path: Path) -> None:
    library = SQLiteLibrary(tmp_path / "library.db")
    item, assessment = fixture_entry()
    for changed in (
        {"state": "verified"},
        {"selected_up": "invented"},
        {"selected_front": "up"},
        {"evidence": assessment.evidence[:2]},
    ):
        with pytest.raises(ValueError):
            library.ingest(item, assessment.model_copy(update=changed))
    assert library.objects() == []
    with pytest.raises(ValueError, match="source mesh"):
        library.ingest(item.model_copy(update={"source_mesh_sha256": "e" * 64}), assessment)
    assert library.objects() == []


def test_future_schema_is_not_overwritten(tmp_path: Path) -> None:
    path = tmp_path / "future.db"
    with sqlite3.connect(path) as db:
        db.execute("PRAGMA user_version=99")
    with pytest.raises(RuntimeError, match="schema version"):
        SQLiteLibrary(path)
    with sqlite3.connect(path) as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == 99

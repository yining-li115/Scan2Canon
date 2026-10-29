"""Pair proposals use verified images, reject unsupported references, and replay safely."""

from pathlib import Path
from typing import Any
from uuid import UUID

import pytest

pytest.importorskip("numpy")
from experiments.artifacts import sha256  # noqa: E402
from experiments.instance_matching.runner import run  # noqa: E402
from scan2canon.library import Assessment, EvidenceRef, ObjectRecord  # noqa: E402
from scan2canon.runtime import build_runtime  # noqa: E402
from tests.canonical_fixtures import FixturePolicy  # noqa: E402
from tests.test_object_library import fixture_entry  # noqa: E402


def entry(root: Path) -> tuple[ObjectRecord, Assessment]:
    root.mkdir()
    item, assessment = fixture_entry()
    refs = []
    for i, ref in enumerate(assessment.evidence):
        path = root / f"evidence-{i}"
        path.write_bytes(f"synthetic evidence {i}".encode())
        refs.append(
            EvidenceRef(
                id="source:0" if ref.kind == "image" else ref.id,
                location=str(path),
                sha256=sha256(path),
                kind=ref.kind,
            )
        )
    item = item.model_copy(update={"source_mesh_sha256": refs[0].sha256})
    assessment = assessment.model_copy(
        update={
            "report_sha256": refs[1].sha256,
            "evidence": tuple(refs),
            "identities": tuple(
                h.model_copy(update={"evidence_ids": ("source:0",)}) for h in assessment.identities
            ),
            "directions": tuple(
                d.model_copy(update={"evidence_ids": ("source:0",)}) for d in assessment.directions
            ),
        }
    )
    return item, assessment


def reply() -> dict[str, Any]:
    return {
        "left_identity": "synthetic fixture",
        "right_identity": "synthetic fixture",
        "identity_relation": "same_function",
        "left_evidence_ids": ["left:source:0"],
        "right_evidence_ids": ["right:source:0"],
        "correspondences": [
            {
                "axis": "up",
                "left_direction_id": "up",
                "right_direction_id": "up",
                "rationale": "fixture mapping",
            }
        ],
        "shared_frame": {
            "version": 1,
            "label": "synthetic shared up",
            "applicability": ["fixture only"],
            "axes": [{"axis": "up", "from_part": "start", "to_part": "end"}],
            "alternatives": "No front established",
        },
        "rationale": "Synthetic contract test, not learned semantics",
    }


def test_pair_correction_and_resume_never_duplicate_proposals(tmp_path: Path) -> None:
    library = build_runtime(tmp_path / "store").library
    left, a = entry(tmp_path / "left")
    right, b = entry(tmp_path / "right")
    library.ingest(left, a)
    library.ingest(right, b)
    bad = reply()
    bad["left_evidence_ids"] = ["right:source:0"]
    policy = FixturePolicy([bad, reply()])
    root = tmp_path / "pair"
    result = run(library, policy, a.id, b.id, root)
    assert len(policy.requests) == 2
    assert "validation_error" in policy.requests[1].prompt
    matches = library.matches(left.id)
    assert len(matches) == 1 and matches[0].shared_frame is not None
    assert matches[0].state == "proposed"
    assert run(library, policy, a.id, b.id, root, resume=True) == result
    assert len(policy.requests) == 2 and library.matches(left.id) == matches
    assert a.model == library.get_assessment(a.id).model
    assert "Synthetic fixture" not in policy.requests[0].prompt  # display name not supplied


def test_modified_evidence_and_uncertain_dispatch_fail_closed(tmp_path: Path) -> None:
    library = build_runtime(tmp_path / "store").library
    left, a = entry(tmp_path / "left")
    right, b = entry(tmp_path / "right")
    library.ingest(left, a)
    library.ingest(right, b)
    path = Path(a.evidence[-1].location)
    original = path.read_bytes()
    path.write_bytes(b"changed")
    policy = FixturePolicy([RuntimeError("transport fixture")])
    root = tmp_path / "pair"
    with pytest.raises(ValueError, match="modified"):
        run(library, policy, a.id, b.id, root)
    assert not policy.requests and not root.exists()
    path.write_bytes(original)
    with pytest.raises(RuntimeError, match="transport"):
        run(library, policy, a.id, b.id, root)
    with pytest.raises(RuntimeError, match="no saved response"):
        run(library, policy, a.id, b.id, root, resume=True)
    assert len(policy.requests) == 1 and library.matches(left.id) == []


def test_same_function_does_not_force_a_shared_front(tmp_path: Path) -> None:
    library = build_runtime(tmp_path / "store").library
    left, a = entry(tmp_path / "left")
    right, b = entry(tmp_path / "right")
    library.ingest(left, a)
    library.ingest(right, b)
    uncertain = reply()
    uncertain.update(identity_relation="uncertain", shared_frame=None, correspondences=[])
    run(library, FixturePolicy([uncertain]), a.id, b.id, tmp_path / "pair")
    match = library.matches(left.id)[0]
    assert match.shared_frame is None and match.identity_relation == "uncertain"
    assert match.left_assessment_id == UUID(str(a.id))

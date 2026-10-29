"""Local object catalog with transactional, immutable proposal records."""

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from uuid import UUID

from scan2canon.library import Assessment, MatchProposal, ObjectRecord


class SQLiteLibrary:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as db:
            version = db.execute("PRAGMA user_version").fetchone()[0]
            if version not in {0, 1}:
                raise RuntimeError(f"Unsupported library schema version: {version}")
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS objects (
                    id TEXT PRIMARY KEY, mesh_sha256 TEXT NOT NULL, payload TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS objects_mesh ON objects(mesh_sha256);
                CREATE TABLE IF NOT EXISTS assessments (
                    id TEXT PRIMARY KEY, object_id TEXT NOT NULL REFERENCES objects(id),
                    report_sha256 TEXT NOT NULL, payload TEXT NOT NULL,
                    UNIQUE(object_id, report_sha256)
                );
                CREATE INDEX IF NOT EXISTS assessments_object ON assessments(object_id);
                CREATE TABLE IF NOT EXISTS matches (
                    id TEXT PRIMARY KEY,
                    left_id TEXT NOT NULL REFERENCES assessments(id),
                    right_id TEXT NOT NULL REFERENCES assessments(id), payload TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS matches_left ON matches(left_id);
                CREATE INDEX IF NOT EXISTS matches_right ON matches(right_id);
                PRAGMA user_version=1;
            """)

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path, timeout=10)
        db.execute("PRAGMA foreign_keys=ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    def ingest(self, item: ObjectRecord, assessment: Assessment) -> None:
        # Revalidate even if a caller used an unchecked model_copy/model_construct.
        item = ObjectRecord.model_validate_json(item.model_dump_json())
        assessment = Assessment.model_validate_json(assessment.model_dump_json())
        if assessment.object_id != item.id:
            raise ValueError("Assessment belongs to a different object")
        if not any(
            e.kind == "mesh" and e.sha256 == item.source_mesh_sha256 for e in assessment.evidence
        ):
            raise ValueError("Assessment must retain the object's source mesh")
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            old = db.execute("SELECT payload FROM objects WHERE id=?", (str(item.id),)).fetchone()
            if old and ObjectRecord.model_validate_json(old[0]) != item:
                raise ValueError("Object ID conflict; existing metadata is immutable")
            db.execute(
                "INSERT OR IGNORE INTO objects VALUES (?, ?, ?)",
                (str(item.id), item.source_mesh_sha256, item.model_dump_json()),
            )
            old = db.execute(
                "SELECT payload FROM assessments WHERE id=? OR (object_id=? AND report_sha256=?)",
                (str(assessment.id), str(item.id), assessment.report_sha256),
            ).fetchone()
            if old:
                if Assessment.model_validate_json(old[0]) != assessment:
                    raise ValueError("Assessment conflict; existing evidence is immutable")
                return
            db.execute(
                "INSERT INTO assessments VALUES (?, ?, ?, ?)",
                (
                    str(assessment.id),
                    str(item.id),
                    assessment.report_sha256,
                    assessment.model_dump_json(),
                ),
            )

    def objects(self, limit: int = 100, offset: int = 0) -> list[ObjectRecord]:
        if not 1 <= limit <= 1000 or offset < 0:
            raise ValueError("Use limit 1–1000 and a nonnegative offset")
        with self._connection() as db:
            rows = db.execute(
                "SELECT payload FROM objects ORDER BY id LIMIT ? OFFSET ?", (limit, offset)
            )
            return [ObjectRecord.model_validate_json(r[0]) for r in rows]

    def get_object(self, object_id: UUID) -> ObjectRecord:
        with self._connection() as db:
            row = db.execute("SELECT payload FROM objects WHERE id=?", (str(object_id),)).fetchone()
            if row is None:
                raise KeyError(str(object_id))
            return ObjectRecord.model_validate_json(row[0])

    def assessments(self, object_id: UUID) -> list[Assessment]:
        self.get_object(object_id)
        with self._connection() as db:
            rows = db.execute(
                "SELECT payload FROM assessments WHERE object_id=? ORDER BY id", (str(object_id),)
            )
            return [Assessment.model_validate_json(r[0]) for r in rows]

    @staticmethod
    def _assessment(db: sqlite3.Connection, assessment_id: UUID) -> Assessment:
        row = db.execute(
            "SELECT payload FROM assessments WHERE id=?", (str(assessment_id),)
        ).fetchone()
        if row is None:
            raise KeyError(str(assessment_id))
        return Assessment.model_validate_json(row[0])

    def get_assessment(self, assessment_id: UUID) -> Assessment:
        with self._connection() as db:
            return self._assessment(db, assessment_id)

    def propose_match(self, proposal: MatchProposal) -> None:
        proposal = MatchProposal.model_validate_json(proposal.model_dump_json())
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            left = self._assessment(db, proposal.left_assessment_id)
            right = self._assessment(db, proposal.right_assessment_id)
            if left.object_id == right.object_id:
                raise ValueError("Cross-instance matches require different objects")
            for pair in proposal.correspondences:
                for assessment, direction_id in (
                    (left, pair.left_direction_id),
                    (right, pair.right_direction_id),
                ):
                    if not any(
                        d.id == direction_id and d.axis == pair.axis for d in assessment.directions
                    ):
                        raise ValueError(
                            "Correspondence references a missing or wrong-axis direction"
                        )
            old = db.execute(
                "SELECT payload FROM matches WHERE id=?", (str(proposal.id),)
            ).fetchone()
            if old:
                if MatchProposal.model_validate_json(old[0]) != proposal:
                    raise ValueError("Match ID conflict; proposals are immutable")
                return
            db.execute(
                "INSERT INTO matches VALUES (?, ?, ?, ?)",
                (str(proposal.id), str(left.id), str(right.id), proposal.model_dump_json()),
            )

    def matches(self, object_id: UUID) -> list[MatchProposal]:
        self.get_object(object_id)
        with self._connection() as db:
            rows = db.execute(
                """SELECT m.payload FROM matches m
                JOIN assessments l ON l.id=m.left_id JOIN assessments r ON r.id=m.right_id
                WHERE l.object_id=? OR r.object_id=? ORDER BY m.id""",
                (str(object_id), str(object_id)),
            )
            return [MatchProposal.model_validate_json(r[0]) for r in rows]

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from uuid import UUID

from scan2canon.domain import Event, Job, Status, utc_now


class SQLiteJobs:
    """One local worker, concurrent API readers, atomic state/event checkpoints."""

    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            if version not in {0, 1}:
                raise RuntimeError(f"Unsupported database schema version: {version}")
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    payload TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS jobs_queue ON jobs(status, created_at);
                CREATE TABLE IF NOT EXISTS events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL REFERENCES jobs(id),
                    payload TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS events_job ON events(job_id, sequence);
                PRAGMA user_version=1;
            """)

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.execute("PRAGMA foreign_keys=ON")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    @staticmethod
    def _get(connection: sqlite3.Connection, job_id: UUID) -> Job:
        row = connection.execute("SELECT payload FROM jobs WHERE id=?", (str(job_id),)).fetchone()
        if row is None:
            raise KeyError(str(job_id))
        return Job.model_validate_json(row[0])

    @staticmethod
    def _save(connection: sqlite3.Connection, job: Job, event: Event) -> None:
        changed = connection.execute(
            "UPDATE jobs SET status=?, payload=? WHERE id=?",
            (job.status.value, job.model_dump_json(), str(job.id)),
        ).rowcount
        if changed != 1:
            raise KeyError(str(job.id))
        connection.execute(
            "INSERT INTO events(job_id, payload) VALUES (?, ?)",
            (str(job.id), event.model_dump_json()),
        )

    def create(self, job: Job) -> None:
        with self._connection() as connection:
            connection.execute(
                "INSERT INTO jobs(id, status, created_at, payload) VALUES (?, ?, ?, ?)",
                (str(job.id), job.status.value, job.created_at.isoformat(), job.model_dump_json()),
            )
            connection.execute(
                "INSERT INTO events(job_id, payload) VALUES (?, ?)",
                (str(job.id), Event(kind="created", message="Job queued").model_dump_json()),
            )

    def get(self, job_id: UUID) -> Job:
        with self._connection() as connection:
            return self._get(connection, job_id)

    def save(self, job: Job, event: Event) -> None:
        with self._connection() as connection:
            self._save(connection, job, event)

    def claim(self) -> Job | None:
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT payload FROM jobs WHERE status=? ORDER BY created_at, id LIMIT 1",
                (Status.QUEUED.value,),
            ).fetchone()
            if row is None:
                return None
            job = Job.model_validate_json(row[0]).model_copy(
                update={
                    "status": Status.RUNNING,
                    "updated_at": utc_now(),
                    "message": None,
                }
            )
            self._save(connection, job, Event(kind="claimed", message="Worker claimed job"))
            return job

    def events(self, job_id: UUID) -> list[Event]:
        with self._connection() as connection:
            self._get(connection, job_id)
            rows = connection.execute(
                "SELECT payload FROM events WHERE job_id=? ORDER BY sequence", (str(job_id),)
            ).fetchall()
            return [Event.model_validate_json(row[0]) for row in rows]

    def retry(self, job_id: UUID) -> Job:
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            job = self._get(connection, job_id)
            if job.status not in {Status.FAILED, Status.BLOCKED, Status.NEEDS_REVIEW}:
                raise ValueError("Only failed, blocked or reviewed jobs can be retried")
            if job.attempts >= job.max_attempts:
                raise ValueError("Attempt budget exhausted; create a new job")
            job = job.model_copy(
                update={
                    "status": Status.QUEUED,
                    "updated_at": utc_now(),
                    "message": None,
                }
            )
            self._save(connection, job, Event(kind="retried", message="Incomplete stage requeued"))
            return job

    def recover_interrupted(self) -> int:
        # The caller must hold the exclusive local worker lock.
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                "SELECT payload FROM jobs WHERE status=?", (Status.RUNNING.value,)
            ).fetchall()
            for row in rows:
                job = Job.model_validate_json(row[0]).model_copy(
                    update={
                        "status": Status.QUEUED,
                        "updated_at": utc_now(),
                        "message": "Previous worker interrupted; resuming incomplete stage",
                    }
                )
                self._save(
                    connection,
                    job,
                    Event(
                        kind="recovered",
                        message="Resuming from the last completed stage",
                    ),
                )
            return len(rows)

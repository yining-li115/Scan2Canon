"""Local development HTTP API; compute is executed by the separate worker."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import UUID

from fastapi import FastAPI, HTTPException
from pydantic import Field

from scan2canon.domain import Event, Job, Model
from scan2canon.runtime import build_runtime


class SubmitJob(Model):
    capture_id: UUID
    max_attempts: int = Field(default=12, ge=1, le=100)


def create_app(data_dir: Path | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.runtime = build_runtime(data_dir)
        yield

    app = FastAPI(title="Scan2Canon", version="0.1.0", lifespan=lifespan)

    @app.get("/healthz")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/jobs", response_model=Job, status_code=202)
    def submit(request: SubmitJob) -> Job:
        try:
            job: Job = app.state.runtime.service.submit(request.capture_id, request.max_attempts)
            return job
        except FileNotFoundError as error:
            raise HTTPException(404, "Capture not found; import images first") from error

    @app.get("/v1/jobs/{job_id}", response_model=Job)
    def status(job_id: UUID) -> Job:
        try:
            job: Job = app.state.runtime.jobs.get(job_id)
            return job
        except KeyError as error:
            raise HTTPException(404, "Job not found") from error

    @app.get("/v1/jobs/{job_id}/events", response_model=list[Event])
    def events(job_id: UUID) -> list[Event]:
        try:
            result: list[Event] = app.state.runtime.jobs.events(job_id)
            return result
        except KeyError as error:
            raise HTTPException(404, "Job not found") from error

    @app.post("/v1/jobs/{job_id}/retry", response_model=Job, status_code=202)
    def retry(job_id: UUID) -> Job:
        try:
            job: Job = app.state.runtime.jobs.retry(job_id)
            return job
        except KeyError as error:
            raise HTTPException(404, "Job not found") from error
        except ValueError as error:
            raise HTTPException(409, str(error)) from error

    return app

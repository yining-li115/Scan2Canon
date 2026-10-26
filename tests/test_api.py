from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient

from scan2canon.api import create_app
from scan2canon.runtime import Runtime
from scan2canon.worker import run_once


def test_api_submission_and_worker_are_separate(runtime: Runtime, images: Path) -> None:
    capture = runtime.service.import_capture(images)
    with TestClient(create_app(runtime.root)) as client:
        response = client.post("/v1/jobs", json={"capture_id": str(capture.id)})
        assert response.status_code == 202
        job_id = response.json()["id"]
        assert client.get(f"/v1/jobs/{job_id}").json()["status"] == "queued"
        assert client.post(f"/v1/jobs/{job_id}/retry").status_code == 409
        run_once(runtime)
        assert client.get(f"/v1/jobs/{job_id}").json()["status"] == "blocked"
        assert client.get(f"/v1/jobs/{job_id}/events").json()[-1]["kind"] == "blocked"
        assert client.post(f"/v1/jobs/{job_id}/retry").status_code == 202


def test_api_validation_and_missing_resources(runtime: Runtime) -> None:
    with TestClient(create_app(runtime.root)) as client:
        assert client.get("/healthz").json() == {"status": "ok"}
        assert client.get(f"/v1/jobs/{uuid4()}").status_code == 404
        assert client.get("/v1/jobs/not-a-uuid").status_code == 422
        assert client.post("/v1/jobs", json={"capture_id": str(uuid4())}).status_code == 404
        assert (
            client.post(
                "/v1/jobs",
                json={
                    "capture_id": str(uuid4()),
                    "source_path": "/private",
                },
            ).status_code
            == 422
        )
        assert (
            client.post(
                "/v1/jobs",
                json={
                    "capture_id": str(uuid4()),
                    "max_attempts": 0,
                },
            ).status_code
            == 422
        )

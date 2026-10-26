import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def test_cli_import_submit_and_resume_across_processes(tmp_path: Path, images: Path) -> None:
    def invoke(*arguments: str) -> dict[str, Any]:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "scan2canon.cli",
                "--data-dir",
                str(tmp_path / "state"),
                *arguments,
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
        )
        payload: dict[str, Any] = json.loads(result.stdout)
        return payload

    capture = invoke("import", str(images))
    job = invoke("submit", capture["id"])
    result = invoke("worker", "--once")
    assert result["id"] == job["id"]
    assert result["status"] == "blocked"
    assert invoke("status", job["id"])["completed"] == ["inspect"]
    invoke("retry", job["id"])
    resumed = invoke("worker", "--once")
    assert resumed["attempts"] == 1
    assert resumed["outputs"] == result["outputs"]

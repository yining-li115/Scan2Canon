from pathlib import Path

import pytest

from scan2canon.domain import REQUIRED_OUTPUT, CanonicalPose, Event, Job, Stage, StageResult, Status
from scan2canon.ports import Decision, Tool, ToolContext
from scan2canon.runtime import Runtime, build_runtime
from scan2canon.worker import run_once, worker_lock


class FixtureTool:
    """Test-only contract fixtures, never production reconstruction outputs."""

    version = "test/1"

    def __init__(self, stage: Stage, quality_passed: bool = True):
        self.stage = stage
        self.quality_passed = quality_passed
        self.calls = 0

    def execute(self, context: ToolContext) -> StageResult:
        self.calls += 1
        payload = b"{}"
        if self.stage == Stage.CANONICALIZE:
            payload = (
                CanonicalPose(
                    rotation=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
                    convention_id="test-v1",
                    category="fixture",
                    symmetry="none",
                )
                .model_dump_json()
                .encode()
            )
        key = context.artifacts.write(context.job.id, f"{self.stage}.json", payload)
        return StageResult(
            outputs={REQUIRED_OUTPUT[self.stage]: key},
            quality_passed=self.quality_passed,
        )


def submit(runtime: Runtime, images: Path, max_attempts: int = 12) -> Job:
    capture = runtime.service.import_capture(images)
    return runtime.service.submit(capture.id, max_attempts)


def test_default_pipeline_stops_truthfully_and_survives_restart(
    runtime: Runtime,
    images: Path,
) -> None:
    job = submit(runtime, images)
    result = run_once(runtime)
    assert result is not None
    assert result.status == Status.BLOCKED
    assert result.completed == (Stage.INSPECT,)
    assert result.message == "No reconstruct adapter configured"
    assert result.attempts == 1
    assert "asset" not in result.outputs
    reopened = build_runtime(runtime.root)
    assert reopened.jobs.get(job.id) == result
    assert [event.kind for event in reopened.jobs.events(job.id)] == [
        "created",
        "claimed",
        "stage_started",
        "stage_completed",
        "blocked",
    ]


def test_retry_resumes_without_repeating_completed_work(runtime: Runtime, images: Path) -> None:
    job = submit(runtime, images)
    first = run_once(runtime)
    assert first is not None
    runtime.harness.tools.update(
        {stage: FixtureTool(stage) for stage in Stage if stage != Stage.INSPECT}
    )
    runtime.jobs.retry(job.id)
    result = run_once(runtime)
    assert result is not None
    assert result.status == Status.SUCCEEDED
    assert result.completed == tuple(Stage)
    assert result.attempts == 6
    assert result.outputs["capture_report"] == first.outputs["capture_report"]
    assert (
        len(
            [
                event
                for event in runtime.jobs.events(job.id)
                if event.kind == "stage_started" and event.stage == Stage.INSPECT
            ]
        )
        == 1
    )


def test_failed_quality_cannot_export(runtime: Runtime, images: Path) -> None:
    job = submit(runtime, images)
    exporter = FixtureTool(Stage.EXPORT)
    runtime.harness.tools.update({stage: FixtureTool(stage) for stage in Stage})
    runtime.harness.tools[Stage.VALIDATE] = FixtureTool(Stage.VALIDATE, quality_passed=False)
    runtime.harness.tools[Stage.EXPORT] = exporter
    result = run_once(runtime)
    assert result is not None
    assert result.status == Status.NEEDS_REVIEW
    assert result.next_stage == Stage.VALIDATE
    assert exporter.calls == 0
    assert "asset" not in runtime.jobs.get(job.id).outputs


def test_attempt_budget_survives_retry(runtime: Runtime, images: Path) -> None:
    job = submit(runtime, images, max_attempts=1)
    result = run_once(runtime)
    assert result is not None
    assert result.status == Status.FAILED
    assert result.attempts == 1
    with pytest.raises(ValueError, match="budget exhausted"):
        runtime.jobs.retry(job.id)


class SkipPolicy:
    def decide(self, job: Job) -> Decision:
        return Decision(Stage.EXPORT, "Skip straight to export")


def test_policy_cannot_skip_validation(runtime: Runtime, images: Path) -> None:
    submit(runtime, images)
    exporter = FixtureTool(Stage.EXPORT)
    runtime.harness.policy = SkipPolicy()
    runtime.harness.tools[Stage.EXPORT] = exporter
    result = run_once(runtime)
    assert result is not None and result.status == Status.FAILED
    assert result.completed == ()
    assert exporter.calls == 0


def test_crash_recovery_retains_attempt_count(runtime: Runtime, images: Path) -> None:
    job = submit(runtime, images)
    claimed = runtime.jobs.claim()
    assert claimed is not None and claimed.status == Status.RUNNING
    runtime.jobs.save(
        claimed.model_copy(update={"attempts": 1}),
        Event(kind="stage_started", stage=Stage.INSPECT, message="Interrupted tool call"),
    )
    result = run_once(build_runtime(runtime.root))
    assert result is not None and result.status == Status.BLOCKED
    assert result.attempts == 2
    assert any(event.kind == "recovered" for event in runtime.jobs.events(job.id))
    assert run_once(runtime) is None


def test_second_worker_is_rejected(runtime: Runtime, images: Path) -> None:
    job = submit(runtime, images)
    with worker_lock(runtime.root), pytest.raises(RuntimeError, match="Another worker"):
        run_once(runtime)
    assert runtime.jobs.get(job.id).status == Status.QUEUED


class MissingArtifactTool:
    version = "test/1"

    def execute(self, context: ToolContext) -> StageResult:
        return StageResult(outputs={"reconstruction": f"jobs/{context.job.id}/missing.json"})


def test_missing_artifact_does_not_advance_checkpoint(runtime: Runtime, images: Path) -> None:
    submit(runtime, images)
    runtime.harness.tools[Stage.RECONSTRUCT] = MissingArtifactTool()
    result = run_once(runtime)
    assert result is not None
    assert result.status == Status.FAILED
    assert result.completed == (Stage.INSPECT,)
    assert "reconstruction" not in result.outputs


class InvalidPoseTool:
    version = "test/1"

    def execute(self, context: ToolContext) -> StageResult:
        key = context.artifacts.write(context.job.id, "pose.json", b'{"rotation": []}')
        return StageResult(outputs={"pose": key})


def test_invalid_pose_cannot_reach_export(runtime: Runtime, images: Path) -> None:
    submit(runtime, images)
    tools: dict[Stage, Tool] = {stage: FixtureTool(stage) for stage in Stage}
    tools[Stage.CANONICALIZE] = InvalidPoseTool()
    runtime.harness.tools.update(tools)
    result = run_once(runtime)
    assert result is not None
    assert result.status == Status.FAILED
    assert result.next_stage == Stage.CANONICALIZE


def test_queued_job_cannot_execute_without_claim(runtime: Runtime, images: Path) -> None:
    job = submit(runtime, images)
    with pytest.raises(ValueError, match="Claim"):
        runtime.harness.run(job)

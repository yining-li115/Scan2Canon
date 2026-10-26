"""Execute policy decisions under persistence, stage-order and validation constraints."""

import logging
from time import perf_counter

from scan2canon.domain import (
    REQUIRED_OUTPUT,
    CanonicalPose,
    Event,
    Job,
    Stage,
    StageResult,
    Status,
    utc_now,
)
from scan2canon.ports import (
    ArtifactStore,
    JobRepository,
    Policy,
    ReviewRequired,
    Tool,
    ToolContext,
    ToolUnavailable,
)

logger = logging.getLogger(__name__)


class Harness:
    def __init__(
        self,
        jobs: JobRepository,
        artifacts: ArtifactStore,
        policy: Policy,
        tools: dict[Stage, Tool],
    ):
        self.jobs = jobs
        self.artifacts = artifacts
        self.policy = policy
        self.tools = tools

    def run(self, job: Job) -> Job:
        if job.status != Status.RUNNING:
            raise ValueError("Claim a queued job before executing it")
        while job.next_stage is not None:
            if job.attempts >= job.max_attempts:
                return self._stop(job, Status.FAILED, "Attempt budget exhausted")
            stage = job.next_stage
            started = perf_counter()
            try:
                decision = self.policy.decide(job.model_copy(deep=True))
                if decision.stage != stage:
                    raise ValueError("Policy attempted to skip a required stage")
                tool = self.tools.get(stage)
                if tool is None:
                    raise ToolUnavailable(f"No {stage.value} adapter configured")
                job = job.model_copy(
                    update={
                        "attempts": job.attempts + 1,
                        "updated_at": utc_now(),
                    }
                )
                self.jobs.save(
                    job,
                    Event(
                        kind="stage_started",
                        stage=stage,
                        message=decision.reason,
                        tool_version=tool.version,
                    ),
                )
                result = tool.execute(
                    ToolContext(
                        job=job.model_copy(deep=True),
                        capture=self.artifacts.capture(job.capture_id),
                        artifacts=self.artifacts,
                    )
                )
                self._validate_result(job, stage, result)
                job = job.model_copy(
                    update={
                        "completed": (*job.completed, stage),
                        "outputs": {**job.outputs, **result.outputs},
                        "updated_at": utc_now(),
                    }
                )
                self.jobs.save(
                    job,
                    Event(
                        kind="stage_completed",
                        stage=stage,
                        message="Stage checkpoint saved",
                        tool_version=tool.version,
                        duration_seconds=perf_counter() - started,
                        metrics=result.metrics,
                    ),
                )
            except ToolUnavailable as error:
                return self._stop(job, Status.BLOCKED, str(error))
            except ReviewRequired as error:
                return self._stop(job, Status.NEEDS_REVIEW, str(error))
            except Exception as error:
                logger.exception("Job %s failed at %s", job.id, stage)
                return self._stop(
                    job,
                    Status.FAILED,
                    f"{stage.value} failed ({type(error).__name__}); see worker logs",
                )
        return self._stop(job, Status.SUCCEEDED, "All stages completed and validation passed")

    def _validate_result(self, job: Job, stage: Stage, result: StageResult) -> None:
        if REQUIRED_OUTPUT[stage] not in result.outputs:
            raise ValueError(f"Missing required output: {REQUIRED_OUTPUT[stage]}")
        if job.outputs.keys() & result.outputs.keys():
            raise ValueError("Tools may not replace completed stages' outputs")
        for key in result.outputs.values():
            if not key.startswith(f"jobs/{job.id}/"):
                raise ValueError("Output artifacts must belong to the current job")
            if not self.artifacts.exists(key):
                raise ValueError(f"Tool returned an absent artifact: {key}")
        if stage == Stage.CANONICALIZE:
            CanonicalPose.model_validate_json(self.artifacts.read(result.outputs["pose"]))
        if stage == Stage.VALIDATE and result.quality_passed is not True:
            raise ReviewRequired("Quality validation did not approve export")

    def _stop(self, job: Job, status: Status, message: str) -> Job:
        job = job.model_copy(
            update={
                "status": status,
                "message": message,
                "updated_at": utc_now(),
            }
        )
        self.jobs.save(job, Event(kind=status.value, stage=job.next_stage, message=message))
        return job

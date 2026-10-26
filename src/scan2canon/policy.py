"""Deterministic baseline policy. No LLM calls are made here."""

from scan2canon.domain import Job
from scan2canon.ports import Decision


class BaselinePolicy:
    def decide(self, job: Job) -> Decision:
        stage = job.next_stage
        if stage is None:
            raise ValueError("The job has no remaining stages")
        return Decision(stage=stage, reason="Run the next incomplete pipeline stage")

"""Explicit composition root for the standalone canonicalization worker."""

import os
from pathlib import Path

from experiments.canonicalization.compute import GeometryTools
from experiments.canonicalization.controller import Controller
from experiments.canonicalization.local_policy import LocalPolicy
from experiments.canonicalization.model_policy import ModelPolicy
from experiments.canonicalization.responses_policy import ResponsesPolicy


def build_controller(
    backend: str, blender: str, model: str | None = None, weights: Path | None = None
) -> Controller:
    policy: ModelPolicy
    if backend == "responses":
        policy = ResponsesPolicy(model or "", os.environ.get("OPENAI_API_KEY", ""))
    elif backend == "local" and weights is not None:
        policy = LocalPolicy(weights)
    else:
        raise ValueError("Choose responses with --model or local with --weights")
    return Controller(policy, GeometryTools(blender))

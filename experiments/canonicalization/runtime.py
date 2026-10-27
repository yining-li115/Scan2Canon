"""Explicit composition root for the standalone canonicalization worker."""

import os
from pathlib import Path

from experiments.canonicalization.compute import GeometryTools
from experiments.canonicalization.controller import Controller
from experiments.canonicalization.gemini_policy import GeminiPolicy
from experiments.canonicalization.local_policy import LocalPolicy
from experiments.canonicalization.model_policy import ModelPolicy
from experiments.canonicalization.responses_policy import ResponsesPolicy


def build_controller(
    backend: str, blender: str, model: str | None = None, weights: Path | None = None
) -> Controller:
    policy: ModelPolicy
    if backend == "gemini":
        policy = GeminiPolicy(
            model if model is not None else os.environ.get("GEMINI_MODEL", ""),
            os.environ.get("GEMINI_API_KEY", ""),
        )
    elif backend == "responses":
        policy = ResponsesPolicy(model or "", os.environ.get("OPENAI_API_KEY", ""))
    elif backend == "local" and weights is not None:
        policy = LocalPolicy(weights)
    else:
        raise ValueError("Choose gemini/responses with a model, or local with --weights")
    return Controller(policy, GeometryTools(blender))

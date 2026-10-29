"""Explicit composition root for the standalone canonicalization worker."""

import os
from pathlib import Path

from experiments.artifacts import read_json
from experiments.canonicalization.agent_controller import AgentController
from experiments.canonicalization.claude_policy import ClaudePolicy
from experiments.canonicalization.compute import GeometryTools
from experiments.canonicalization.controller import Controller
from experiments.canonicalization.gemini_policy import GeminiPolicy
from experiments.canonicalization.local_policy import LocalPolicy
from experiments.canonicalization.model_policy import ModelPolicy
from experiments.canonicalization.open_controller import OpenController
from experiments.canonicalization.open_tools import OpenGeometryTools
from experiments.canonicalization.responses_policy import ResponsesPolicy
from experiments.canonicalization.shared_frame import SharedFrameController
from scan2canon.library import MatchProposal


def build_policy(
    backend: str, model: str | None = None, weights: Path | None = None
) -> ModelPolicy:
    policy: ModelPolicy
    if backend == "gemini":
        policy = GeminiPolicy(
            model if model is not None else os.environ.get("GEMINI_MODEL", ""),
            os.environ.get("GEMINI_API_KEY", ""),
        )
    elif backend == "claude":
        policy = ClaudePolicy(
            model if model is not None else os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5-5"),
            os.environ.get("ANTHROPIC_API_KEY", ""),
            stop_at=os.environ.get("ANTHROPIC_STOP_AT", ""),
        )
    elif backend == "responses":
        policy = ResponsesPolicy(model or "", os.environ.get("OPENAI_API_KEY", ""))
    elif backend == "local" and weights is not None:
        policy = LocalPolicy(weights)
    else:
        raise ValueError("Choose gemini/claude/responses with a model, or local with --weights")
    return policy


def build_controller(
    backend: str, blender: str, model: str | None = None, weights: Path | None = None
) -> Controller:
    return Controller(build_policy(backend, model, weights), GeometryTools(blender))


def build_open_controller(backend: str, blender: str, model: str | None = None) -> OpenController:
    if backend not in ("gemini", "claude", "responses"):
        raise ValueError(
            "Open semantics requires a cloud policy supporting custom response schemas"
        )
    return OpenController(build_policy(backend, model), OpenGeometryTools(blender))


def build_agent_controller(
    backend: str, blender: str, model: str | None = None, *, allow_shape_only: bool = False
) -> AgentController:
    if backend not in ("gemini", "claude", "responses"):
        raise ValueError("Evidence agent requires a cloud policy with structured visual outputs")
    fallbacks: tuple[ModelPolicy, ...] = ()
    if backend == "gemini" and model is None:
        pool = read_json(
            Path(
                os.environ.get(
                    "GEMINI_MODELS_FILE", str(Path(__file__).with_name("gemini-models.json"))
                )
            )
        )
        models = pool.get("models")
        if (
            not isinstance(models, list)
            or not 1 <= len(models) <= 4
            or not all(isinstance(name, str) for name in models)
            or len(set(models)) != len(models)
        ):
            raise ValueError("Gemini model list requires one to four distinct model IDs")
        model = os.environ.get("GEMINI_MODEL") or models[0]
        if model not in models:
            raise ValueError("GEMINI_MODEL must be in the model list; --model pins a single model")
        fallbacks = tuple(build_policy(backend, name) for name in models if name != model)
    return AgentController(
        build_policy(backend, model),
        OpenGeometryTools(blender),
        allow_shape_only=allow_shape_only,
        fallback_policies=fallbacks,
    )


def build_shared_controller(
    backend: str,
    blender: str,
    proposal: "MatchProposal",
    source_mesh_sha256: str,
    model: str | None = None,
) -> "SharedFrameController":
    return SharedFrameController(
        build_policy(backend, model), OpenGeometryTools(blender), proposal, source_mesh_sha256
    )

"""Lazy local-model boundary; GPU packages are needed only when dispatching a call."""

from pathlib import Path
from typing import Any

from experiments.artifacts import sha256
from experiments.canonicalization.model_policy import PolicyRequest
from experiments.semantic_orientation.assets import verify_weights
from experiments.semantic_orientation.selector import LocalSelector


class LocalPolicy:
    def __init__(self, weights: Path) -> None:
        verify_weights(weights, "vlm")
        self.weights = weights
        self.selector: LocalSelector | None = None

    @property
    def identity(self) -> dict[str, str]:
        return {
            "backend": "local_qwen",
            "weights_manifest_sha256": sha256(self.weights / "manifest.json"),
            "implementation_sha256": sha256(Path(__file__)),
            "selector_sha256": sha256(
                Path(__file__).parents[1] / "semantic_orientation/selector.py"
            ),
        }

    def select(self, request: PolicyRequest) -> dict[str, Any]:
        if self.selector is None:
            self.selector = LocalSelector(self.weights)
        return self.selector.select_labeled(request.prompt, list(request.images))

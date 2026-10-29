"""Model boundary for bounded candidate selection; no geometry or model SDKs here."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from experiments.semantic_orientation.contracts import Selection


class PolicyRateLimited(RuntimeError):
    """Provider explicitly rejected a call for quota or rate limits."""


class InvalidPolicyReply(ValueError):
    """A saved model reply failed the response contract, not an input integrity check."""


@dataclass(frozen=True)
class PolicyRequest:
    task: str
    prompt: str
    images: tuple[tuple[str, Path], ...]
    response_schema: dict[str, Any] | None = None


class PolicyReply(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    selection: Selection
    supporting_image_ids: list[str] = Field(default_factory=list)
    mesh_correspondence: str = ""


class ModelPolicy(Protocol):
    @property
    def identity(self) -> dict[str, str]: ...

    def select(self, request: PolicyRequest) -> dict[str, Any]:
        """Return raw_text plus optional provider usage; perform exactly one model call."""
        ...

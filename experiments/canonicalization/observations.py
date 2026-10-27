"""Bind supplemental photographs to a mesh without treating them as camera poses."""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from experiments.artifacts import read_json
from experiments.orientation.common import checked_file


class SupportingEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    manifest: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    image_ids: list[str] = Field(min_length=1)
    mesh_correspondence: str = Field(min_length=1)


def verify_support(directory: Path, case: dict[str, Any], evidence: SupportingEvidence) -> None:
    path = checked_file(directory, evidence.manifest, evidence.sha256)
    bundle = read_json(path)
    if (
        bundle.get("schema_version") != 1
        or bundle.get("status") != "succeeded"
        or bundle.get("mesh_sha256") != case["mesh_sha256"]
        or bundle.get("category") != case["category"]
        or bundle.get("registration") != "unregistered"
    ):
        raise ValueError("Supplemental observations do not match the mesh and category")
    images = bundle["images"]
    ids = {image["id"] for image in images}
    if (
        not images
        or len(ids) != len(images)
        or len(set(evidence.image_ids)) != len(evidence.image_ids)
        or not set(evidence.image_ids).issubset(ids)
    ):
        raise ValueError("Supplemental image IDs must be unique and present in the bundle")
    for asset in bundle["assets"]:
        checked_file(path.parent, asset["path"], asset["sha256"])
    for image in images:
        checked_file(path.parent, image["image"], image["sha256"])
    # Identity and visual correspondence remain evidence claims, not a pose estimate.
    # The only executable rotation still comes from the checked geometric candidates.

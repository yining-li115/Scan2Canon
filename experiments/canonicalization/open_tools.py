"""Appearance-preserving evidence preparation for open semantic canonicalization."""

import shutil
from pathlib import Path
from typing import Any

from experiments.appearance import describe
from experiments.artifacts import read_json, sha256, write_json
from experiments.canonicalization.compute import copy_observations
from experiments.canonicalization.oa_proposals import load_predictions, proposals
from experiments.canonicalization.observations import SupportingEvidence, verify_support
from experiments.orientation.common import checked_file, verify_views
from experiments.orientation.render import render


def photographs(directory: Path | None, mesh_sha256: str) -> list[tuple[str, Path]]:
    if directory is None:
        return []
    manifest = read_json(directory / "manifest.json")
    # Category metadata verifies an existing bundle only; it is never a VLM input
    # or an identity label in this route. Identity is inferred independently.
    verify_support(
        directory,
        {"mesh_sha256": mesh_sha256, "category": manifest.get("category")},
        SupportingEvidence(
            manifest="manifest.json",
            sha256=sha256(directory / "manifest.json"),
            image_ids=[i["id"] for i in manifest["images"]],
            mesh_correspondence="Input association, not an estimated camera pose",
        ),
    )
    if len(manifest["images"]) > 8:
        raise ValueError("At most eight supplemental photographs are supported")
    if any(i.get("kind") == "opposite_candidate_view" for i in manifest["images"]):
        raise ValueError("Use original observations, not another policy's candidate evidence")
    return [(f"photo:{i['id']}", directory / i["image"]) for i in manifest["images"]]


class OpenGeometryTools:
    def __init__(self, blender: str, proposal_method: str = "decoded") -> None:
        if proposal_method not in ("decoded", "distribution"):
            raise ValueError("Choose decoded or distribution proposals")
        self.blender = blender
        self.proposal_method = proposal_method

    def snapshot(
        self, views: Path, predictions: Path, observations: Path | None, output: Path
    ) -> Path:
        manifest, _ = load_predictions(views, predictions)
        photographs(observations, manifest["mesh_sha256"])
        target = output / "views"
        target.mkdir(parents=True)
        for relative, digest in [
            (manifest["mesh"], manifest["mesh_sha256"]),
            *((v["image"], v["sha256"]) for v in manifest["views"]),
        ]:
            original = checked_file(views, relative, digest)
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original, destination)
        shutil.copyfile(views / "manifest.json", target / "manifest.json")
        (output / "oa").mkdir()
        for name in ("predictions.json", "logits.npz", "checkpoint-manifest.json"):
            shutil.copyfile(predictions / name, output / "oa" / name)
        if observations:
            copy_observations(
                observations,
                output / "observations",
                {
                    "mesh_sha256": manifest["mesh_sha256"],
                    "category": read_json(observations / "manifest.json").get("category"),
                },
            )
        write_json(output / "appearance.json", describe(target / manifest["mesh"]))
        return output

    def prepare(self, snapshot: Path, output: Path) -> Path:
        if self.proposal_method == "distribution":
            from experiments.canonicalization.distribution_proposals import proposals as fused

            report = fused(snapshot / "views", snapshot / "oa")
        else:
            report = proposals(snapshot / "views", snapshot / "oa")
        report["appearance"] = read_json(snapshot / "appearance.json")
        output.mkdir()
        write_json(output / "candidates.json", report)
        if len(report["candidates"]) < 3:
            raise ValueError("Insufficient distinct proposals for the bounded comparison tools")
        return self.render_candidates(snapshot, output, report)

    def render_candidates(self, snapshot: Path, output: Path, report: dict[str, Any]) -> Path:
        manifest = verify_views(snapshot / "views")
        render(
            snapshot / "views" / manifest["mesh"],
            output / "primary",
            self.blender,
            poses=[c["rotation"] for c in report["candidates"]],
            pose_camera=(0, -2, 0.7),
        )
        return output

    def opposite(self, snapshot: Path, candidates: Path, output: Path) -> Path:
        manifest = verify_views(snapshot / "views")
        output.mkdir()
        report = read_json(candidates / "candidates.json")
        render(
            snapshot / "views" / manifest["mesh"],
            output / "opposite",
            self.blender,
            poses=[c["rotation"] for c in report["candidates"]],
            pose_camera=(0, 2, 0.7),
        )
        return output


def image_list(directory: Path, prefix: str) -> list[tuple[str, Path]]:
    manifest = verify_views(directory)
    return [(f"{prefix}:{v['id']}", directory / v["image"]) for v in manifest["views"]]

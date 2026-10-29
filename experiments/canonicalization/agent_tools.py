"""Local, bounded evidence actions; models choose IDs, never file paths or commands."""

import shutil
from pathlib import Path
from typing import Any

from PIL import Image

from experiments.artifacts import read_json, sha256, write_json
from experiments.canonicalization.agent_contracts import EvidenceAction
from experiments.orientation.common import checked_file, verify_views
from experiments.orientation.render import render

CAMERAS = {"back": (0, 2, 0.7), "left": (-2, 0, 0.7), "right": (2, 0, 0.7), "top": (0, -0.01, 2)}


def action_key(action: EvidenceAction) -> dict[str, Any]:
    key = action.model_dump(exclude={"reason", "expected_evidence"}, exclude_none=True)
    for field in ("candidate_ids", "photo_ids"):
        if field in key:
            key[field] = sorted(key[field])
    return key


def validate_action(
    action: EvidenceAction,
    candidates: dict[str, Any],
    images: dict[str, Path],
    photos: dict[str, Path],
    completed: list[dict[str, Any]],
) -> None:
    if action.kind in ("finish", "request_capture"):
        return
    if action_key(action) in completed:
        raise ValueError("Repeated evidence action; obtain different evidence or finish")
    if action.kind == "refine_up" and any(a["kind"] == "refine_up" for a in completed):
        raise ValueError("Only one bounded up refinement is allowed per job")
    if action.candidate_ids and not set(action.candidate_ids) <= {
        c["id"] for c in candidates["candidates"]
    }:
        raise ValueError("Unknown render candidate")
    if action.kind == "refine_up" and any(
        "refinement" in c
        for c in candidates["candidates"]
        if c["id"] in (action.candidate_ids or [])
    ):
        raise ValueError("Choose original candidates, not chained refinements")
    if action.image_id is not None and action.image_id not in images:
        raise ValueError("Crop source is not a currently visible image")
    if action.photo_ids and not set(action.photo_ids) <= photos.keys():
        raise ValueError("Requested photo is not in the verified observation bundle")
    if action.photo_ids and set(action.photo_ids) <= images.keys():
        raise ValueError("Requested photos are already visible")


def execute_evidence(
    action: EvidenceAction,
    snapshot: Path,
    candidates: dict[str, Any],
    images: dict[str, Path],
    photos: dict[str, Path],
    output: Path,
    blender: str,
    turn: int,
) -> Path:
    output = output.resolve()
    output.mkdir()
    report: dict[str, Any] = {"action": action.model_dump(), "images": [], "result": {}}

    def append(label: str, path: Path, **metadata: Any) -> None:
        report["images"].append(
            {
                "id": label,
                "image": str(path.relative_to(output)),
                "sha256": sha256(path),
                **metadata,
            }
        )

    if action.kind == "refine_up":
        import trimesh

        from experiments.canonicalization.up_refinement import refine_up

        assert action.candidate_ids is not None
        manifest = verify_views(snapshot / "views")
        mesh_path = snapshot / "views" / manifest["mesh"]
        result = refine_up(
            trimesh.load_scene(mesh_path, process=False).to_mesh(), candidates, action.candidate_ids
        )
        additions = result.pop("new_candidates")
        report["result"] = result
        report["new_candidates"] = additions
        if additions:
            poses = [c["rotation"] for c in additions]
            # The shared renderer takes batches of at least three. Padding views are
            # recorded but never supplied as additional hypotheses or model evidence.
            padded = poses + [poses[-1]] * max(0, 3 - len(poses))
            bundle = render(
                mesh_path, output / "renders", blender, poses=padded, pose_camera=(0, -2, 0.2)
            )
            rendered = verify_views(bundle)
            report["result"]["padding_views_not_exposed"] = len(padded) - len(poses)
            for view, candidate in zip(rendered["views"][: len(additions)], additions, strict=True):
                append(
                    f"candidate:{candidate['id']}",
                    bundle / view["image"],
                    candidate_id=candidate["id"],
                    parent_candidate_id=candidate["refinement"]["parent_candidate_id"],
                    camera="near_level_front",
                )
    elif action.kind == "render_view":
        assert action.candidate_ids is not None and action.camera is not None
        rows = {c["id"]: c for c in candidates["candidates"]}
        manifest = verify_views(snapshot / "views")
        bundle = render(
            snapshot / "views" / manifest["mesh"],
            output / "renders",
            blender,
            poses=[rows[i]["rotation"] for i in action.candidate_ids],
            pose_camera=CAMERAS[action.camera],
        )
        rendered = verify_views(bundle)
        for view, candidate in zip(rendered["views"], action.candidate_ids, strict=True):
            append(
                f"detail:{turn}:{candidate}:{action.camera}",
                bundle / view["image"],
                candidate_id=candidate,
                camera=action.camera,
            )
        report["result"] = {"note": "New cameras on existing geometry; no new observed surfaces"}
    elif action.kind == "inspect_image":
        assert action.image_id is not None and action.crop is not None
        source = images[action.image_id]
        with Image.open(source) as image:
            width, height = image.size
            left, top, right, bottom = action.crop
            bounds = (
                int(left * width),
                int(top * height),
                int(right * width),
                int(bottom * height),
            )
            if bounds[0] >= bounds[2] or bounds[1] >= bounds[3]:
                raise ValueError("Crop is smaller than one source pixel")
            image.crop(bounds).save(output / "crop.png")
        prefix = "photo" if action.image_id.startswith("photo:") else "crop"
        append(
            f"{prefix}:crop-{turn}",
            output / "crop.png",
            source_id=action.image_id,
            source_sha256=sha256(source),
            pixel_bounds=list(bounds),
        )
    elif action.kind == "retrieve_photos":
        assert action.photo_ids is not None
        for index, label in enumerate(action.photo_ids):
            original = photos[label]
            path = output / f"photo-{index}{original.suffix}"
            shutil.copyfile(original, path)
            append(label, path)
        report["result"] = {"registration": "unregistered", "requires_mesh_correspondence": True}
    else:
        raise ValueError("Only evidence tools can execute here")
    write_json(output / "evidence.json", report)
    return output


def read_evidence(output: Path) -> tuple[dict[str, Any], list[tuple[str, Path]]]:
    report = read_json(output / "evidence.json")
    return report, [
        (i["id"], checked_file(output, i["image"], i["sha256"])) for i in report["images"]
    ]

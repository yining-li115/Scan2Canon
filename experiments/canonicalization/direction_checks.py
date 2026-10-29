"""Check projected axis/landmark relations; never certify landmark semantics or symmetry."""

import math
import textwrap
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from experiments.artifacts import read_json, sha256
from experiments.canonicalization.direction_contracts import DirectionAssessment
from experiments.orientation.geometry import rotation


def projected_relation(
    tail: list[float], head: list[float], uncertainty: float, camera_axis: list[float], fov: float
) -> dict[str, Any]:
    """Pinhole tangent at the observed tail; positive depth cancels from its direction."""
    start, end, axis = np.asarray(tail), np.asarray(head), np.asarray(camera_axis)
    if not 0 < fov < math.pi or not np.isfinite(axis).all():
        raise ValueError("Invalid projection calibration")
    tangent = math.tan(fov / 2)
    ray = np.array([(2 * start[0] - 1) * tangent, (1 - 2 * start[1]) * tangent])
    projected = np.array([axis[0] + ray[0] * axis[2], -axis[1] - ray[1] * axis[2]])
    delta = end - start
    length = float(np.linalg.norm(delta))
    if np.linalg.norm(projected) < 0.15:
        return {"status": "indeterminate", "reason": "axis_near_viewing_ray"}
    if length <= max(0.04, 4 * uncertainty):
        return {"status": "indeterminate", "reason": "landmarks_too_close_or_uncertain"}
    angle = math.degrees(
        math.acos(float(np.clip(delta @ projected / (length * np.linalg.norm(projected)), -1, 1)))
    )
    # Endpoint uncertainty and the tail-dependent perspective direction both matter.
    measured_bound = math.asin(min(1.0, 2 * uncertainty / length))
    projection_bound = math.asin(
        min(1.0, 2 * tangent * uncertainty * abs(axis[2]) / float(np.linalg.norm(projected)))
    )
    bound = math.degrees(measured_bound + projection_bound)
    low, high = max(0.0, angle - bound), min(180.0, angle + bound)
    status = "compatible" if high <= 20 else "conflict" if low > 20 else "indeterminate"
    return {
        "status": status,
        "angle_degrees": angle,
        "angle_interval_degrees": [low, high],
        "tolerance_degrees": 20,
        "reason": "projected_part_direction",
        "expected_image_direction": (projected / np.linalg.norm(projected)).tolist(),
    }


def evaluate(
    assessment: DirectionAssessment,
    candidates: dict[str, Any],
    images: dict[str, Path],
    previous_checks: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    rows = {c["id"]: c for c in candidates["candidates"]}
    hypotheses = {h.id: h for h in assessment.hypotheses}
    citations = [h.evidence_ids for h in assessment.hypotheses]
    citations.extend(i.evidence_ids for i in assessment.issues)
    citations.append(assessment.symmetry.evidence_ids)
    for labels in citations:
        if not set(labels) <= images.keys():
            raise ValueError("Direction evidence must cite currently visible images")
    checks = []
    for measurement in assessment.measurements:
        m = measurement
        if m.image_id not in images or m.candidate_id not in rows:
            raise ValueError("Direction measurement cites an unavailable image or candidate")
        h = hypotheses[m.hypothesis_id]
        record: dict[str, Any] = {
            **m.model_dump(),
            "axis": h.axis,
            "from_part": h.from_part,
            "to_part": h.to_part,
            "image_sha256": sha256(images[m.image_id]),
        }
        # Rephrased IDs and shifted points do not make unchanged pixels new evidence.
        key = ("image_sha256", "candidate_id", "axis", "from_part", "to_part")
        previous = [
            c["uncertainty"]
            for c in previous_checks or []
            if all(c.get(k) == record[k] for k in key)
        ]
        record["reported_uncertainty"] = m.uncertainty
        record["uncertainty"] = max([m.uncertainty, *previous])
        record["uncertainty_floor_applied"] = record["uncertainty"] > m.uncertainty
        pieces = m.image_id.split(":")
        image_candidate = (
            int(pieces[1])
            if pieces[0] == "candidate" and len(pieces) == 2
            else int(pieces[2])
            if pieces[0] == "detail" and len(pieces) == 4
            else None
        )
        if image_candidate is None:
            record.update(status="indeterminate", reason="unregistered_or_unsupported_image")
        elif image_candidate != m.candidate_id:
            raise ValueError("Measurement candidate differs from rendered image candidate")
        else:
            path = images[m.image_id]
            manifest = read_json(path.parent / "manifest.json")
            view = next(v for v in manifest["views"] if v["image"] == path.name)
            if "camera_horizontal_fov_radians" not in manifest:
                record.update(status="indeterminate", reason="missing_camera_calibration")
            else:
                canonical_axis = [0, 1, 0] if h.axis == "up" else [0, 0, 1]
                camera_axis = (
                    rotation(view["camera_to_world"]).T
                    @ rotation(manifest["mesh_to_blender"])
                    @ rotation(rows[m.candidate_id]["rotation"]).T
                    @ canonical_axis
                )
                record.update(
                    projected_relation(
                        [m.tail.x, m.tail.y],
                        [m.head.x, m.head.y],
                        record["uncertainty"],
                        camera_axis.tolist(),
                        manifest["camera_horizontal_fov_radians"],
                    )
                )
                record["render_manifest_sha256"] = sha256(path.parent / "manifest.json")
        checks.append(record)
    return {
        "method": "projected_part_relation_v1",
        "checks": checks,
        "landmark_source": "policy_estimate_not_independent_annotation",
        "symmetry_verified": False,
        "semantic_correctness_verified": False,
        "scope": "2D compatibility only; not a 3D pose or part localization certificate",
    }


def validate_export(
    assessment: DirectionAssessment, report: dict[str, Any], scope: str, candidate: int | None
) -> None:
    if scope == "unresolved":
        return
    axes = ["up", "front"] if scope == "full_frame" else ["up"]
    for axis in axes:
        selected = getattr(assessment, f"selected_{axis}")
        if selected is None or any(i.axis == axis for i in assessment.issues):
            raise ValueError(f"Resolve {axis} meaning/issues before exporting that axis")
        matching = [
            c
            for c in report["checks"]
            if c["hypothesis_id"] == selected and c["candidate_id"] == candidate
        ]
        if not any(c["status"] == "compatible" for c in matching) or any(
            c["status"] == "conflict" for c in matching
        ):
            raise ValueError(
                f"{axis} requires a compatible projected part check without conflicting checks; "
                "inspect another view, correct the candidate or preserve unresolved axes"
            )
    if scope == "up_only" and assessment.selected_front is not None:
        raise ValueError("An up-only export must leave selected_front null")


def render_checks(report: dict[str, Any], images: dict[str, Path], output: Path) -> None:
    """Audit the policy's actual landmark estimates; these overlays are not model evidence."""
    output.mkdir(exist_ok=True)
    for index, check in enumerate(report["checks"]):
        with Image.open(images[check["image_id"]]) as original:
            rgba = original.convert("RGBA")
            background = Image.new("RGBA", rgba.size, "white")
            background.alpha_composite(rgba)
            background.thumbnail((768, 768))
        width, height = background.size
        canvas = Image.new("RGB", (max(width, 600), height + 150), "white")
        canvas.paste(background.convert("RGB"), (0, 50))
        draw = ImageDraw.Draw(canvas)
        font = ImageFont.load_default(size=17)
        color = {"compatible": "#138645", "conflict": "#c62a39", "indeterminate": "#a06c10"}[
            check["status"]
        ]
        draw.text(
            (10, 12),
            f"{check['image_id']} | {check['axis']} | {check['status']}",
            fill=color,
            font=font,
        )
        tail = np.array([check["tail"]["x"] * width, check["tail"]["y"] * height + 50])
        head = np.array([check["head"]["x"] * width, check["head"]["y"] * height + 50])
        draw.line([tuple(tail), tuple(head)], fill=color, width=4)
        for name, point in (("A", tail), ("B", head)):
            radius = max(3, check["uncertainty"] * width)
            x, y = point
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=2)
            draw.text(
                (x + 5, y + 5), name, fill=color, font=font, stroke_width=1, stroke_fill="white"
            )
        if "expected_image_direction" in check:
            expected = np.asarray(check["expected_image_direction"])
            end = tail + expected * min(width * 0.25, float(np.linalg.norm(head - tail)))
            draw.line([tuple(tail), tuple(end)], fill="#2466da", width=2)
            draw.text(tuple(end), "+axis", fill="#2466da", font=font)
        label = f"A: {check['from_part']} -> B: {check['to_part']}"
        draw.multiline_text(
            (10, height + 58), "\n".join(textwrap.wrap(label, 70)[:3]), fill="#172238", font=font
        )
        draw.text(
            (10, height + 125),
            "Policy landmarks; 2D check, not semantic ground truth",
            fill="#505b6c",
            font=font,
        )
        canvas.save(output / f"check-{index:02d}.png")

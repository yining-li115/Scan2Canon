"""Inspect stored mesh appearance independently of geometry and semantic identity."""

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import trimesh


def describe(mesh: Path) -> dict[str, Any]:
    """Inventory actual GLB assets; presence of color is not proof of faithful reconstruction."""
    scene = trimesh.load_scene(mesh, process=False)
    records = []
    for geometry in scene.geometry.values():
        visual = geometry.visual
        record: dict[str, Any] = {"kind": visual.kind, "color_evidence": False, "textures": []}
        if visual.kind in ("vertex", "face"):
            colors = np.asarray(getattr(visual, f"{visual.kind}_colors"), dtype=np.uint8)
            rgb = colors[:, :3]
            record.update(
                colors_sha256=hashlib.sha256(colors.tobytes()).hexdigest(),
                color_evidence=bool(np.any(rgb != rgb[:, :1]) or np.any(rgb != rgb[0])),
            )
        if visual.kind == "texture":
            uv = getattr(visual, "uv", None)
            material = visual.material
            if uv is not None:
                uv = np.asarray(uv, dtype=np.float32)
                if uv.shape != (len(geometry.vertices), 2) or not np.isfinite(uv).all():
                    raise ValueError("Invalid UV coordinates")
                record["uv_sha256"] = hashlib.sha256(uv.tobytes()).hexdigest()
            for slot in (
                "baseColorTexture",
                "emissiveTexture",
                "normalTexture",
                "metallicRoughnessTexture",
                "occlusionTexture",
                "image",
            ):
                image = getattr(material, slot, None)
                if image is not None:
                    if uv is None:
                        raise ValueError("Texture image has no usable UV mapping")
                    rgba = image.convert("RGBA")
                    record["textures"].append(
                        {
                            "slot": slot,
                            "size": list(rgba.size),
                            "sha256": hashlib.sha256(rgba.tobytes()).hexdigest(),
                        }
                    )
            for field in (
                "baseColorFactor",
                "emissiveFactor",
                "metallicFactor",
                "roughnessFactor",
                "alphaMode",
                "alphaCutoff",
                "doubleSided",
            ):
                value = getattr(material, field, None)
                if value is not None:
                    record[field] = np.asarray(value).tolist()
            factor = np.asarray(getattr(material, "baseColorFactor", [255, 255, 255, 255]))
            record["color_evidence"] = bool(
                any(
                    t["slot"] in ("baseColorTexture", "image", "emissiveTexture")
                    for t in record["textures"]
                )
                or (factor.ndim == 1 and len(factor) >= 3 and np.any(factor[:3] != factor[0]))
            )
        records.append(record)
    # Geometry names and scene transforms may legitimately change on export.
    records.sort(key=lambda r: json.dumps(r, sort_keys=True))
    return {
        "schema_version": 1,
        "has_color_evidence": any(r["color_evidence"] for r in records),
        "has_texture": any(r["textures"] for r in records),
        "geometries": records,
        "scope": (
            "Stored appearance only; generated colors/textures are not verified real-world evidence"
        ),
    }


def require_color(report: dict[str, Any]) -> None:
    if not report["has_color_evidence"]:
        raise ValueError(
            "Input has no color/texture evidence. Supply an appearance-bearing GLB; "
            "use --allow-shape-only explicitly for a geometry-only baseline. "
            "Separate photographs do not texture a mesh."
        )

"""Verified OA V2 image predictions converted into GLB-space pose proposals."""

from collections import defaultdict
from itertools import zip_longest
from pathlib import Path
from typing import Any

import numpy as np

from experiments.artifacts import read_json, sha256
from experiments.orientation.common import config, verify_views
from experiments.orientation.geometry import (
    OA_TO_CANONICAL,
    angular_error,
    axis_rotation,
    mesh_to_oa,
    rotation,
)


def load_predictions(views: Path, predictions: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = verify_views(views)
    result = read_json(predictions / "predictions.json")
    checkpoint = read_json(predictions / "checkpoint-manifest.json")
    if (
        result.get("schema_version") != 1
        or result.get("status") != "succeeded"
        or result.get("settings") != config()
        or checkpoint.get("settings") != config()
        or result.get("views_manifest_sha256") != sha256(views / "manifest.json")
        or result.get("logits_sha256") != sha256(predictions / "logits.npz")
    ):
        raise ValueError("OA predictions must match the complete render bundle and pinned model")
    digest = checkpoint.get("sha256", "")
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("Missing checkpoint provenance")
    for view in manifest["views"]:
        if type(view["id"]) is not int:
            raise ValueError("OA source view IDs must be integers")
        rotation(view["camera_to_world"])
    expected = {v["id"] for v in manifest["views"]}
    rows = result["predictions"]
    if len(rows) != len(expected) or {p["id"] for p in rows} != expected:
        raise ValueError("OA predictions must cover each source view exactly once")
    with np.load(predictions / "logits.npz", allow_pickle=False) as logits:
        for p in rows:
            if type(p["id"]) is not int or type(p["symmetry_fit"]) is not int:
                raise ValueError("OA view IDs and symmetry fits must be integers")
            if p["symmetry_fit"] not in (0, 1, 2, 4):
                raise ValueError("Unsupported OA symmetry fit")
            if len(p["angles_degrees"]) != 3 or not np.isfinite(p["angles_degrees"]).all():
                raise ValueError("Invalid OA angles")
            value = logits[f"view_{p['id']:02d}"]
            if value.shape != (900,) or not np.isfinite(value).all():
                raise ValueError("Invalid saved OA logits")
    return manifest, result


def proposals(views: Path, predictions: Path, maximum: int = 12) -> dict[str, Any]:
    if not 3 <= maximum <= 12:
        raise ValueError("Use 3–12 candidates to preserve the visual-call image budget")
    manifest, inference = load_predictions(views, predictions)
    cameras = {v["id"]: v["camera_to_world"] for v in manifest["views"]}
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for prediction in sorted(inference["predictions"], key=lambda p: p["id"]):
        base = OA_TO_CANONICAL @ mesh_to_oa(prediction["angles_degrees"], cameras[prediction["id"]])
        order = prediction["symmetry_fit"]
        for offset in range(order or 1):
            matrix = rotation(axis_rotation(1, 360 * offset / (order or 1)) @ base)
            groups[order].append(
                {
                    "source_view": prediction["id"],
                    "symmetry_fit": order,
                    "yaw_offset_degrees": 360 * offset / (order or 1),
                    "rotation": matrix.tolist(),
                    "front_available": order != 0,
                }
            )
    # Retain different orientations first, then source-view duplicates if fewer than
    # three exist. Every entry still comes from an actual saved OA prediction.
    for order, group in groups.items():
        diverse: list[dict[str, Any]] = []
        duplicates: list[dict[str, Any]] = []
        for candidate in group:
            destination = (
                duplicates
                if any(
                    angular_error(rotation(candidate["rotation"]), rotation(c["rotation"])) < 5
                    for c in diverse
                )
                else diverse
            )
            destination.append(candidate)
        groups[order] = diverse + duplicates
    # Preserve minority symmetry modes instead of silently turning majority votes
    # into a symmetry label. Truncation is recorded and never implies certainty.
    pool = [
        candidate
        for group in zip_longest(*(groups[k] for k in sorted(groups)))
        for candidate in group
        if candidate is not None
    ]
    selected = [{"id": i, **c} for i, c in enumerate(pool[:maximum])]
    return {
        "schema_version": 1,
        "status": "succeeded",
        "mesh_sha256": manifest["mesh_sha256"],
        "predictions_sha256": sha256(predictions / "predictions.json"),
        "checkpoint_manifest_sha256": sha256(predictions / "checkpoint-manifest.json"),
        "views_manifest_sha256": sha256(views / "manifest.json"),
        "candidates": selected,
        "available_hypotheses": len(pool),
        "omitted_hypotheses": max(0, len(pool) - len(selected)),
        "symmetry_fits": sorted(groups),
        "coordinate_convention": "+Y up, +Z front; proper GLB-space rotations",
        "note": "OA learned axes are proposals; fit 0 is unresolved yaw, not physical symmetry",
    }

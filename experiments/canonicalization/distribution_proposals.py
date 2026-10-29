"""Adapt verified multiview OA distributions to bounded agent pose proposals."""

from pathlib import Path
from typing import Any

import numpy as np

from experiments.artifacts import sha256
from experiments.canonicalization.oa_proposals import load_predictions
from experiments.orientation import distribution
from experiments.orientation.geometry import OA_TO_CANONICAL


def proposals(views: Path, predictions: Path) -> dict[str, Any]:
    manifest, _ = load_predictions(views, predictions)
    rows = sorted(manifest["views"], key=lambda r: r["id"])
    with np.load(predictions / "logits.npz", allow_pickle=False) as archive:
        logits = np.stack([archive[f"view_{r['id']:02d}"] for r in rows])
    report = distribution.fuse(np.array([r["camera_to_world"] for r in rows]), logits)
    return {
        **report,
        "schema_version": 1,
        "status": "succeeded",
        "mesh_sha256": manifest["mesh_sha256"],
        "predictions_sha256": sha256(predictions / "predictions.json"),
        "checkpoint_manifest_sha256": sha256(predictions / "checkpoint-manifest.json"),
        "views_manifest_sha256": sha256(views / "manifest.json"),
        "fusion_code_sha256": sha256(Path(distribution.__file__)),
        "omitted_hypotheses": "Unknown: finite pose search, not exhaustive",
        "candidates": [
            {
                **{k: v for k, v in c.items() if k != "mesh_to_oa"},
                "rotation": (OA_TO_CANONICAL @ np.array(c["mesh_to_oa"])).tolist(),
                "source_views": [r["id"] for r in rows],
                "front_available": True,
                "front_status": "Hypothesis to inspect; not verified or a symmetry claim",
                "proposal_method": distribution.SETTINGS["version"],
            }
            for c in report["candidates"]
        ],
        "coordinate_convention": "+Y up, +Z front; proper GLB-space rotations",
    }

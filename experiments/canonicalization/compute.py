"""CPU/render tools used by the canonicalization controller."""

import shutil
from pathlib import Path
from typing import Any

from experiments.artifacts import read_json, sha256, write_json
from experiments.canonicalization.observations import SupportingEvidence, verify_support
from experiments.canonicalization.semantic import descriptor, verify
from experiments.canonicalization.upright import prepare as prepare_upright
from experiments.canonicalization.workflow import finish, next_front
from experiments.orientation.common import checked_file
from experiments.orientation.render import render


def copy_observations(source: Path, target: Path, case: dict[str, Any]) -> None:
    bundle = read_json(source / "manifest.json")
    support = SupportingEvidence(
        manifest="manifest.json",
        sha256=sha256(source / "manifest.json"),
        image_ids=[image["id"] for image in bundle["images"]],
        mesh_correspondence="Input evidence association; not a pose estimate",
    )
    verify_support(source, case, support)
    files = {asset["path"]: asset["sha256"] for asset in bundle["assets"]}
    files.update({image["image"]: image["sha256"] for image in bundle["images"]})
    target.mkdir(parents=True, exist_ok=False)
    for relative, digest in files.items():
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError("Observation paths must be relative and remain inside their bundle")
        original = checked_file(source, relative, digest)
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, destination)
    shutil.copyfile(source / "manifest.json", target / "manifest.json")


class GeometryTools:
    def __init__(self, blender: str) -> None:
        self.blender = blender

    def up(self, mesh: Path, category: str, observations: Path | None, output: Path) -> Path:
        prepare_upright(mesh, category, output, self.blender)
        self._attach(observations, output)
        return output

    def front(self, up: Path, decision: Path, observations: Path | None, output: Path) -> Path:
        next_front(up, decision, output, self.blender)
        self._attach(observations, output)
        return output

    def augment(self, base: Path, output: Path) -> Path:
        case, _ = verify(base)
        shutil.copytree(base, output)
        observations = output / "observations"
        observations.mkdir(exist_ok=True)
        manifest = observations / "manifest.json"
        if manifest.exists():
            bundle = read_json(manifest)
        else:
            bundle = {
                "schema_version": 1,
                "status": "succeeded",
                "mesh_sha256": case["mesh_sha256"],
                "category": case["category"],
                "registration": "unregistered",
                "scope": "Extra views of existing candidates; no externally supplied pose",
                "assets": [],
                "images": [],
            }
        alternate = observations / "controller-opposite"
        render(
            base / "poses/source.glb",
            alternate,
            self.blender,
            poses=[c["rotation"] for c in case["candidates"]],
            pose_camera=(-1, 2, 0.8),
        )
        for path in sorted(alternate.iterdir()):
            if path.is_file():
                bundle["assets"].append(
                    {"path": "controller-opposite/" + path.name, "sha256": sha256(path)}
                )
        for view in read_json(alternate / "manifest.json")["views"]:
            bundle["images"].append(
                {
                    "id": f"opposite-candidate-{view['id']:02d}",
                    "image": "controller-opposite/" + view["image"],
                    "sha256": view["sha256"],
                    "kind": "opposite_candidate_view",
                }
            )
        write_json(manifest, bundle)
        return output

    def export(
        self,
        up: Path,
        up_decision: Path,
        front: Path | None,
        front_decision: Path | None,
        output: Path,
    ) -> Path:
        return finish(up, up_decision, output, front, front_decision)

    @staticmethod
    def _attach(observations: Path | None, output: Path) -> None:
        if observations is not None:
            copy_observations(observations, output / "observations", read_json(descriptor(output)))

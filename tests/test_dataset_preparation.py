import json
import shutil
from pathlib import Path

import pytest
from PIL import Image

np = pytest.importorskip("numpy")
trimesh = pytest.importorskip("trimesh")

from experiments.artifacts import read_json, sha256, write_json  # noqa: E402
from experiments.datasets.nocs import prepare, render_inputs  # noqa: E402
from experiments.datasets.references import build  # noqa: E402


@pytest.fixture
def catalog(tmp_path: Path) -> Path:
    entries = []
    for index, category in enumerate(("camera", "mug")):
        path = tmp_path / f"{category}.obj"
        mesh = trimesh.creation.box(extents=[1 + index, 2, 3])
        mesh.apply_translation([4, -2, 9])
        mesh.export(path)
        entries.append(
            {
                "id": f"fixture-{category}",
                "source": "nocs",
                "category": category,
                "partition": "development" if index == 0 else "holdout",
                "mesh": {"path": path.name, "sha256": sha256(path)},
            }
        )
    result = tmp_path / "catalog.json"
    write_json(result, {"samples": entries})
    return result


def test_rotation_inputs_preserve_geometry_and_do_not_expose_answers(
    catalog: Path, tmp_path: Path
) -> None:
    output = prepare(catalog, tmp_path, tmp_path / "prepared")
    refs = read_json(output / "references.json")
    inputs = read_json(output / "inputs/index.json")
    assert inputs["status"] == refs["status"] == "succeeded"
    assert len(refs["cases"]) == 10
    assert {c["partition"] for c in inputs["cases"]} == {"development", "holdout"}
    assert not any(word in json.dumps(inputs) for word in ("rotation", "seed", "center"))
    assert {p.suffix for p in (output / "inputs").iterdir()} == {".json", ".glb"}
    for case in refs["cases"]:
        scene = trimesh.load_scene(output / "inputs" / case["mesh"], process=False)
        for node in scene.graph.nodes_geometry:
            np.testing.assert_allclose(scene.graph[node][0], np.eye(4), atol=1e-7)
        moved = scene.to_mesh()
        source = trimesh.load_mesh(tmp_path / f"{case['category']}.obj", process=False)
        undo = np.asarray(case["input_to_source_rotation"])
        np.testing.assert_allclose(undo @ undo.T, np.eye(3), atol=1e-7)
        assert np.linalg.det(undo) == pytest.approx(1)
        np.testing.assert_allclose(
            moved.vertices @ undo.T + case["source_center"], source.vertices, atol=1e-6
        )
        np.testing.assert_array_equal(moved.faces, source.faces)
        assert not np.allclose(moved.vertices, source.vertices - case["source_center"])
    before = sha256(output / "references.json")
    with pytest.raises(FileExistsError):
        prepare(catalog, tmp_path, output)
    assert sha256(output / "references.json") == before


def test_catalog_order_does_not_change_instance_rotations(catalog: Path, tmp_path: Path) -> None:
    first = prepare(catalog, tmp_path, tmp_path / "first", count=2)
    settings = read_json(catalog)
    settings["samples"].reverse()
    write_json(catalog, settings)
    second = prepare(catalog, tmp_path, tmp_path / "second", count=2)
    for path in (first / "inputs").glob("*.glb"):
        assert sha256(path) == sha256(second / "inputs" / path.name)


def test_modified_source_marks_failed_and_preserves_completed_cases(
    catalog: Path, tmp_path: Path
) -> None:
    (tmp_path / "mug.obj").write_text("modified")
    output = tmp_path / "partial"
    with pytest.raises(ValueError, match="modified artifact"):
        prepare(catalog, tmp_path, output, count=2)
    assert read_json(output / "references.json")["status"] == "failed"
    assert len(read_json(output / "inputs/index.json")["cases"]) == 2
    assert len(list((output / "inputs").glob("*.glb"))) == 2
    with pytest.raises(ValueError, match="incomplete"):
        render_inputs(output, tmp_path / "renders", "unused")


def fake_render(mesh: Path, output: Path, blender: str, views: int = 12) -> Path:
    output.mkdir()
    shutil.copyfile(mesh, output / "source.glb")
    images = []
    for i in range(views):
        path = output / f"view-{i:02d}.png"
        Image.new("RGB", (16, 16), "gray").save(path)
        images.append({"id": i, "image": path.name, "sha256": sha256(path)})
    write_json(
        output / "manifest.json",
        {
            "schema_version": 1,
            "status": "succeeded",
            "mesh": "source.glb",
            "mesh_sha256": sha256(mesh),
            "preset": "sphere",
            "renderer_sha256": sha256(
                Path(__file__).parents[1] / "experiments/orientation/blender_render.py"
            ),
            "views": images,
        },
    )
    return output


def test_render_retry_reuses_verified_completed_cases(
    catalog: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared = prepare(catalog, tmp_path, tmp_path / "prepared", count=1)
    output = tmp_path / "renders"
    calls = []

    def interrupted(mesh: Path, target: Path, blender: str, views: int = 12) -> Path:
        calls.append(target.name)
        if len(calls) == 2:
            raise RuntimeError("interrupted before second render")
        return fake_render(mesh, target, blender, views)

    monkeypatch.setattr("experiments.datasets.nocs.render", interrupted)
    with pytest.raises(RuntimeError, match="interrupted"):
        render_inputs(prepared, output, "unused", views=3)
    progress = read_json(output / "index.json")
    assert progress["status"] == "failed" and len(progress["cases"]) == 1
    first = output / progress["cases"][0]["case_id"] / "manifest.json"
    original_mtime = first.stat().st_mtime_ns
    monkeypatch.setattr("experiments.datasets.nocs.render", fake_render)
    render_inputs(prepared, output, "unused", views=3)
    assert read_json(output / "index.json")["status"] == "succeeded"
    assert first.stat().st_mtime_ns == original_mtime
    (first.parent / "view-00.png").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="modified artifact"):
        render_inputs(prepared, output, "unused", views=3)
    assert read_json(output / "index.json")["status"] == "failed"


@pytest.mark.parametrize("count,seed", [(0, 17), (21, 17), (5, -1)])
def test_invalid_rotation_request_writes_nothing(
    catalog: Path, tmp_path: Path, count: int, seed: int
) -> None:
    output = tmp_path / "invalid"
    with pytest.raises(ValueError):
        prepare(catalog, tmp_path, output, count, seed)
    assert not output.exists()


def reference_rules(catalog: Path, tmp_path: Path) -> Path:
    objects = read_json(catalog)["samples"]
    path = tmp_path / "conventions.json"
    write_json(
        path,
        {
            "status": "source_convention_checked",
            "catalog_sha256": sha256(catalog),
            "objects": [
                {
                    "object_id": obj["id"],
                    "category": obj["category"],
                    "source_mesh_sha256": obj["mesh"]["sha256"],
                    "status": "source_convention_checked",
                }
                for obj in objects
            ],
            "categories": {
                obj["category"]: {
                    "source_to_project_rotation": [[0, 0, -1], [0, 1, 0], [1, 0, 0]],
                    "source_up": [0, 1, 0],
                    "source_reference_front": [1, 0, 0],
                    "metric": "full_frame" if i == 0 else "up_only",
                }
                for i, obj in enumerate(objects)
            },
        },
    )
    return path


def test_reference_composition_and_up_only_cases(catalog: Path, tmp_path: Path) -> None:
    prepared = prepare(catalog, tmp_path, tmp_path / "prepared", count=2)
    rules = reference_rules(catalog, tmp_path)
    result = build(prepared, rules, tmp_path / "scoring.json")
    known = {c["case_id"]: c for c in read_json(prepared / "references.json")["cases"]}
    for case in read_json(result)["cases"]:
        applied = np.asarray(known[case["case_id"]]["source_to_input_rotation"])
        np.testing.assert_allclose(case["up_in_input_frame"], applied @ [0, 1, 0])
        if case["metric"] == "full_frame":
            expected = np.asarray(case["input_to_project_rotation"])
            np.testing.assert_allclose(expected @ applied @ [1, 0, 0], [0, 0, 1], atol=1e-7)
            np.testing.assert_allclose(expected @ applied @ [0, 1, 0], [0, 1, 0], atol=1e-7)
        else:
            assert case["input_to_project_rotation"] is None
            assert case["reference_front_in_input_frame"] is None
    with pytest.raises(ValueError, match="outside model inputs"):
        build(prepared, rules, prepared / "inputs/leaked-reference.json")


def test_reference_rejects_wrong_source_and_inverse(catalog: Path, tmp_path: Path) -> None:
    prepared = prepare(catalog, tmp_path, tmp_path / "prepared", count=1)
    rules = reference_rules(catalog, tmp_path)
    original = read_json(rules)
    broken = read_json(rules)
    broken["objects"][0]["source_mesh_sha256"] = "wrong"
    write_json(rules, broken)
    with pytest.raises(ValueError, match="does not match"):
        build(prepared, rules, tmp_path / "bad-source.json")
    assert not (tmp_path / "bad-source.json").exists()
    write_json(rules, original)
    known = read_json(prepared / "references.json")
    known["cases"][0]["input_to_source_rotation"] = np.eye(3).tolist()
    write_json(prepared / "references.json", known)
    with pytest.raises(ValueError, match="inverse"):
        build(prepared, rules, tmp_path / "bad-inverse.json")


def test_render_index_rejects_escaping_case_ids(catalog: Path, tmp_path: Path) -> None:
    prepared = prepare(catalog, tmp_path, tmp_path / "prepared", count=1)
    index = read_json(prepared / "inputs/index.json")
    index["cases"][0]["case_id"] = "../escaped"
    write_json(prepared / "inputs/index.json", index)
    with pytest.raises(ValueError, match="safe case identifiers"):
        render_inputs(prepared, tmp_path / "renders", "unused", views=3)
    assert not (tmp_path / "renders").exists()

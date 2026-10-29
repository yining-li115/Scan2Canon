"""Standalone Blender renderer. Invoke through experiments.orientation.render."""

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Vector


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mesh", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--views", type=int, default=12)
    parser.add_argument("--resolution", type=int, default=518)
    parser.add_argument("--poses", type=Path)
    parser.add_argument(
        "--preset", choices=("sphere", "canonical", "ring", "axes"), default="sphere"
    )
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(args.mesh))
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not meshes:
        raise ValueError("Input has no mesh objects")
    corners = [obj.matrix_world @ Vector(v) for obj in meshes for v in obj.bound_box]
    lower = Vector([min(v[i] for v in corners) for i in range(3)])
    upper = Vector([max(v[i] for v in corners) for i in range(3)])
    center = (lower + upper) / 2
    radius = max((v - center).length for v in corners)
    if not math.isfinite(radius) or radius <= 1e-8:
        raise ValueError("Input has invalid or zero extent")
    scene = bpy.context.scene
    # Imported cameras/lights are not part of the object evidence.
    for obj in list(scene.objects):
        if obj.type in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(obj, do_unlink=True)
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 16
    scene.cycles.use_denoising = True
    scene.render.resolution_x = scene.render.resolution_y = args.resolution
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = True
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.exposure = -1.0
    scene.world = bpy.data.worlds.new("Render lighting")
    scene.world.use_nodes = True
    scene.world.node_tree.nodes["Background"].inputs[0].default_value = (1, 1, 1, 1)
    scene.world.node_tree.nodes["Background"].inputs[1].default_value = 0.5
    for index, direction in enumerate([(1, -1, 2), (-2, 1, 1), (0, 2, -1)]):
        light = bpy.data.lights.new(f"Light {index}", "AREA")
        light.energy = 25 * radius * radius
        light.shape = "DISK"
        light.size = radius * 3
        obj = bpy.data.objects.new(light.name, light)
        scene.collection.objects.link(obj)
        obj.location = center + Vector(direction).normalized() * radius * 3
        obj.rotation_euler = (center - obj.location).to_track_quat("-Z", "Y").to_euler()
    camera_data = bpy.data.cameras.new("Capture")
    camera = bpy.data.objects.new("Capture", camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    camera_data.type = "PERSP"
    camera_data.lens = 50
    camera_data.sensor_width = 36
    camera_data.sensor_fit = "HORIZONTAL"
    distance = radius / math.sin(camera_data.angle_x / 2) * 1.12
    camera_data.clip_start = radius * 0.01
    camera_data.clip_end = radius * 20
    directions: list[tuple[str, tuple[float, float, float]]] = []
    if args.preset == "axes":
        directions = [
            ("Source +Z", (0, -1, 0)),
            ("Source -Z", (0, 1, 0)),
            ("Source +X", (1, 0, 0)),
            ("Source -X", (-1, 0, 0)),
            ("Source +Y", (0, 0, 1)),
            ("Source -Y", (0, 0, -1)),
            ("Source oblique", (1, -1, 1)),
        ]
    elif args.preset == "canonical":
        directions = [
            ("Front (+Z)", (0, -1, 0)),
            ("Back (-Z)", (0, 1, 0)),
            ("Side (+X)", (1, 0, 0)),
            ("Top (+Y)", (0, 0, 1)),
            ("Bottom (-Y)", (0, 0, -1)),
            ("Three-quarter", (1, -1, 1)),
        ]
    elif args.preset == "ring":
        for index in range(args.views):
            yaw = 2 * math.pi * index / args.views
            directions.append((f"Candidate {index:02d}", (math.sin(yaw), -math.cos(yaw), 0.35)))
    else:
        for index in range(args.views):
            z = 1 - 2 * (index + 0.5) / args.views
            azimuth = index * math.pi * (3 - math.sqrt(5))
            horizontal = math.sqrt(1 - z * z)
            directions.append(
                (
                    f"View {index:02d}",
                    (horizontal * math.cos(azimuth), horizontal * math.sin(azimuth), z),
                )
            )
    views = []
    pose_config = json.loads(args.poses.read_text()) if args.poses else None
    pose_values = pose_config["rotations"] if pose_config else None
    if pose_values is not None and pose_config is not None:
        directions = [
            (f"Pose {i:02d}", pose_config.get("camera_direction", (1, -2, 1)))
            for i in range(len(pose_values))
        ]
    glb_to_blender = Matrix(((1, 0, 0), (0, 0, -1), (0, 1, 0)))
    for index, (label, direction_values) in enumerate(directions):
        direction = Vector(direction_values).normalized()
        camera.location = center + direction * distance
        camera.rotation_euler = (center - camera.location).to_track_quat("-Z", "Y").to_euler()
        if pose_values is not None:
            inverse_pose = (
                glb_to_blender
                @ Matrix(pose_values[index]).transposed()
                @ glb_to_blender.transposed()
            )
            camera.location = center + inverse_pose @ (camera.location - center)
            camera.rotation_euler = (inverse_pose @ camera.rotation_euler.to_matrix()).to_euler()
        bpy.context.view_layer.update()
        filename = f"view-{index:02d}.png"
        scene.render.filepath = str(args.output / filename)
        bpy.ops.render.render(write_still=True)
        views.append(
            {
                "id": index,
                "label": label,
                "image": filename,
                "camera_to_world": [list(row) for row in camera.matrix_world.to_3x3()],
                "camera_position": list(camera.location),
            }
        )
    # Blender imported glTF using (x, y, z) -> (x, -z, y).
    center_glb = [center.x, center.z, -center.y]
    result = {
        "blender_version": bpy.app.version_string,
        "renderer": "Cycles CPU, 16 samples, Standard color transform",
        "exposure": -1.0,
        "camera_convention": "Blender/OpenGL camera-to-world: right, up, backward",
        "mesh_to_blender": [[1, 0, 0], [0, 0, -1], [0, 1, 0]],
        "appearance_rendering": {
            "mode": "imported_materials_no_override",
            "mesh_color_attributes": sum(len(o.data.color_attributes) for o in meshes),
            "material_slots": sum(len(o.data.materials) for o in meshes),
            "image_textures": sum(
                image.source == "FILE" or image.packed_file is not None for image in bpy.data.images
            ),
        },
        "resolution": args.resolution,
        "preset": args.preset,
        "center_glb": center_glb,
        "radius": radius,
        "camera_horizontal_fov_radians": camera_data.angle_x,
        "views": views,
    }
    (args.output / "cameras.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()

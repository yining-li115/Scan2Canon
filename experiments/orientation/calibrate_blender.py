"""Check coordinate equations against the actual upstream Blender asset.

blender --background --python-exit-code 1 --python THIS_FILE -- SOURCE_DIR OUTPUT_JSON
"""

import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.orientation.geometry import (  # noqa: E402
    GLB_TO_BLENDER,
    OA_CAMERA_TO_WORLD,
    mesh_to_oa,
    oa_object_rotation,
)


def main() -> None:
    source, output = (Path(p).resolve() for p in sys.argv[sys.argv.index("--") + 1 :])
    if output.exists():
        raise FileExistsError(output)
    bpy.ops.wm.open_mainfile(filepath=str(source / "assets/axis_render.blend"))
    camera = next(obj for obj in bpy.data.objects if obj.type == "CAMERA")
    np.testing.assert_allclose(camera.matrix_world.to_3x3(), OA_CAMERA_TO_WORLD, atol=1e-6)
    for name, expected in zip(("X", "Y", "Z"), np.eye(3), strict=True):
        obj = bpy.data.objects[name]
        center = sum((obj.matrix_world @ Vector(v) for v in obj.bound_box), Vector()) / 8
        np.testing.assert_allclose(center.normalized(), expected, atol=1e-6)
    axes = bpy.data.objects["三轴"]
    samples = [(0, 0, 0), (45, 20, -35), (179, -70, 120), (300, 45, -179)]
    for az, el, roll in samples:
        axes.rotation_mode = "ZYX"
        axes.rotation_euler = tuple(math.radians(v) for v in (roll, el, -az))
        bpy.context.view_layer.update()
        actual = np.asarray(axes.matrix_world.to_3x3())
        np.testing.assert_allclose(actual, oa_object_rotation(az, el, roll), atol=1e-6)
        recovered = mesh_to_oa([az, el, roll], np.asarray(camera.matrix_world.to_3x3()))
        np.testing.assert_allclose(recovered @ GLB_TO_BLENDER.T @ actual, np.eye(3), atol=1e-6)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "status": "passed",
                "blender": bpy.app.version_string,
                "checks": [
                    "upstream camera",
                    "front/side/up axes",
                    "Blender Euler order",
                    "inverse pose recovery",
                ],
                "angle_samples": samples,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Coordinate calibration passed: {output}")


if __name__ == "__main__":
    main()

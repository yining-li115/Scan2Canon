"""Column-vector rotations connecting OA's axis renderer, Blender and GLB.

OA's zero-angle camera sits at +X, with camera right=+Y and up=+Z.
Its axis renderer applies Rx(roll) Ry(elevation) Rz(-azimuth).
All cameras below use Blender/OpenGL local axes: right, up, backward.
"""

import math

import numpy as np
from numpy.typing import NDArray

Matrix = NDArray[np.float64]
GLB_TO_BLENDER = np.array([[1.0, 0, 0], [0, 0, -1], [0, 1, 0]])
OA_CAMERA_TO_WORLD = np.array([[0.0, 0, 1], [1, 0, 0], [0, 1, 0]])
# OA +X front, +Z up -> exported +Z front, +Y up. Determinant stays +1.
OA_TO_CANONICAL = np.array([[0.0, 1, 0], [0, 0, 1], [1, 0, 0]])


def rotation(value: object) -> Matrix:
    matrix = np.asarray(value, dtype=float)
    if (
        matrix.shape != (3, 3)
        or not np.isfinite(matrix).all()
        or not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-5)
        or not np.isclose(np.linalg.det(matrix), 1.0, atol=1e-5)
    ):
        raise ValueError("Expected a finite proper rotation, without reflection or scale")
    return matrix


def axis_rotation(axis: int, degrees: float) -> Matrix:
    if axis not in (0, 1, 2) or not math.isfinite(degrees):
        raise ValueError("Invalid rotation axis or angle")
    angle = math.radians(degrees)
    c, s = math.cos(angle), math.sin(angle)
    matrix = np.eye(3)
    i, j = (axis + 1) % 3, (axis + 2) % 3
    matrix[i, i] = matrix[j, j] = c
    matrix[i, j], matrix[j, i] = -s, s
    return matrix


def oa_object_rotation(azimuth: float, elevation: float, roll: float) -> Matrix:
    return axis_rotation(0, roll) @ axis_rotation(1, elevation) @ axis_rotation(2, -azimuth)


def mesh_to_oa(angles: list[float], camera_to_world: object) -> Matrix:
    if len(angles) != 3:
        raise ValueError("Expected azimuth, elevation and roll")
    camera = rotation(camera_to_world)
    predicted = oa_object_rotation(*angles)
    return rotation(predicted.T @ OA_CAMERA_TO_WORLD @ camera.T @ GLB_TO_BLENDER)


def angular_error(first: Matrix, second: Matrix) -> float:
    cosine = np.clip((np.trace(first @ second.T) - 1) / 2, -1, 1)
    return math.degrees(math.acos(float(cosine)))


def symmetry_error(first: Matrix, second: Matrix, order: int) -> float:
    if order == 0:
        # Up-only comparison. A flat fit does NOT establish physical axial symmetry.
        cosine = np.clip(np.dot(first[2], second[2]), -1, 1)
        return math.degrees(math.acos(float(cosine)))
    if order not in (1, 2, 4):
        raise ValueError("Unsupported OA symmetry fit")
    return min(
        angular_error(first, axis_rotation(2, 360 * k / order) @ second) for k in range(order)
    )


def centered_transform(mesh_to_oa_rotation: Matrix, center: object) -> Matrix:
    origin = np.asarray(center, dtype=float)
    if origin.shape != (3,) or not np.isfinite(origin).all():
        raise ValueError("Expected a finite center")
    transform = np.eye(4)
    transform[:3, :3] = OA_TO_CANONICAL @ rotation(mesh_to_oa_rotation)
    transform[:3, 3] = -transform[:3, :3] @ origin
    return transform

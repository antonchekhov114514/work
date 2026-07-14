from __future__ import annotations

import numpy as np

from .mesh import TetMesh


BONE = 1
SOFT = 2
SAT = 3
SKIN = 4


def layered_torso_mesh(n_xy: int = 4, n_z: int | None = None) -> TetMesh:
    """Create a small conforming layered block used only for verification."""
    if n_xy < 3:
        raise ValueError("n_xy must be at least 3")
    n_z = n_z if n_z is not None else max(2, n_xy // 2)
    xs = np.linspace(-1.0, 1.0, n_xy + 1)
    ys = np.linspace(-1.0, 1.0, n_xy + 1)
    zs = np.linspace(-0.6, 0.6, n_z + 1)
    points = np.asarray([(x, y, z) for z in zs for y in ys for x in xs], dtype=float)

    def node(i: int, j: int, k: int) -> int:
        return k * (n_xy + 1) * (n_xy + 1) + j * (n_xy + 1) + i

    tetra: list[list[int]] = []
    pattern = np.asarray(
        [
            [0, 1, 2, 6],
            [0, 2, 3, 6],
            [0, 3, 7, 6],
            [0, 7, 4, 6],
            [0, 4, 5, 6],
            [0, 5, 1, 6],
        ],
        dtype=np.int64,
    )
    for k in range(n_z):
        for j in range(n_xy):
            for i in range(n_xy):
                cube = np.asarray(
                    [
                        node(i, j, k),
                        node(i + 1, j, k),
                        node(i + 1, j + 1, k),
                        node(i, j + 1, k),
                        node(i, j, k + 1),
                        node(i + 1, j, k + 1),
                        node(i + 1, j + 1, k + 1),
                        node(i, j + 1, k + 1),
                    ],
                    dtype=np.int64,
                )
                tetra.extend(cube[pattern].tolist())
    tetra_array = np.asarray(tetra, dtype=np.int64)
    centroid = points[tetra_array].mean(axis=1)
    radius = np.maximum(np.abs(centroid[:, 0]), np.abs(centroid[:, 1]))
    tags = np.full(len(tetra_array), SKIN, dtype=np.int64)
    tags[radius <= 0.88] = SAT
    tags[radius <= 0.58] = SOFT
    tags[radius <= 0.28] = BONE
    names = {"BONE": BONE, "SOFT": SOFT, "SAT": SAT, "SKIN": SKIN}
    return TetMesh(points, tetra_array, tags, names)


from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np


@dataclass
class TetMesh:
    points: np.ndarray
    tetra: np.ndarray
    cell_tags: np.ndarray
    tag_names: dict[str, int] = field(default_factory=dict)
    cell_data: Mapping[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.points = np.asarray(self.points, dtype=float)
        self.tetra = np.asarray(self.tetra, dtype=np.int64)
        self.cell_tags = np.asarray(self.cell_tags, dtype=np.int64)
        self.tag_names = {str(k): int(v) for k, v in self.tag_names.items()}
        self.cell_data = {
            str(name): np.asarray(values)
            for name, values in dict(self.cell_data).items()
        }

        if self.points.ndim != 2 or self.points.shape[1] != 3:
            raise ValueError("points must have shape (N, 3)")
        if self.tetra.ndim != 2 or self.tetra.shape[1] != 4:
            raise ValueError("tetra must have shape (M, 4)")
        if self.cell_tags.shape != (len(self.tetra),):
            raise ValueError("cell_tags must have one value per tetrahedron")
        for name, values in self.cell_data.items():
            if values.shape[0] != len(self.tetra):
                raise ValueError(f"cell_data[{name!r}] must have one value per tetrahedron")
        if self.tetra.size and (self.tetra.min() < 0 or self.tetra.max() >= len(self.points)):
            raise ValueError("tetra contains an invalid point index")

        self._orient_positive()

    def _orient_positive(self) -> None:
        x = self.points[self.tetra]
        dm = np.stack((x[:, 1] - x[:, 0], x[:, 2] - x[:, 0], x[:, 3] - x[:, 0]), axis=2)
        det = np.linalg.det(dm)
        scale = max(float(np.ptp(self.points, axis=0).max()), 1.0)
        tol = np.finfo(float).eps * scale**3 * 100.0
        if np.any(np.abs(det) <= tol):
            bad = np.flatnonzero(np.abs(det) <= tol)[:10]
            raise ValueError(f"degenerate tetrahedra detected at indices {bad.tolist()}")
        negative = det < 0.0
        if np.any(negative):
            tmp = self.tetra[negative, 1].copy()
            self.tetra[negative, 1] = self.tetra[negative, 2]
            self.tetra[negative, 2] = tmp

    @property
    def n_points(self) -> int:
        return int(len(self.points))

    @property
    def n_cells(self) -> int:
        return int(len(self.tetra))

    def resolve_tag(self, value: str | int) -> int:
        if isinstance(value, (int, np.integer)):
            return int(value)
        text = str(value)
        if text in self.tag_names:
            return self.tag_names[text]
        try:
            return int(text)
        except ValueError as exc:
            known = ", ".join(sorted(self.tag_names)) or "<none>"
            raise KeyError(f"unknown tag {text!r}; known names: {known}") from exc

    def nodes_for_tags(self, tags: list[int] | tuple[int, ...] | np.ndarray) -> np.ndarray:
        selected = np.isin(self.cell_tags, np.asarray(tags, dtype=np.int64))
        if not np.any(selected):
            return np.empty(0, dtype=np.int64)
        return np.unique(self.tetra[selected].ravel())

    def reference_geometry(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return reference volumes, shape gradients, and inverse edge matrices."""
        x = self.points[self.tetra]
        dm = np.stack((x[:, 1] - x[:, 0], x[:, 2] - x[:, 0], x[:, 3] - x[:, 0]), axis=2)
        inv_dm = np.linalg.inv(dm)
        volumes = np.linalg.det(dm) / 6.0
        grads = np.empty((self.n_cells, 4, 3), dtype=float)
        grads[:, 1:, :] = inv_dm
        grads[:, 0, :] = -np.sum(inv_dm, axis=1)
        return volumes, grads, inv_dm

    def cell_volumes(self, points: np.ndarray | None = None) -> np.ndarray:
        p = self.points if points is None else np.asarray(points, dtype=float)
        x = p[self.tetra]
        ds = np.stack((x[:, 1] - x[:, 0], x[:, 2] - x[:, 0], x[:, 3] - x[:, 0]), axis=2)
        return np.linalg.det(ds) / 6.0

    def copy_with_points(self, points: np.ndarray) -> "TetMesh":
        return TetMesh(
            points,
            self.tetra.copy(),
            self.cell_tags.copy(),
            self.tag_names.copy(),
            {name: values.copy() for name, values in self.cell_data.items()},
        )

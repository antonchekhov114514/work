from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .mesh import TetMesh


@dataclass
class ContactConstraints:
    node_ids: np.ndarray
    coefficients: np.ndarray
    normals: np.ndarray
    clearance: np.ndarray
    penalty: np.ndarray

    def __post_init__(self) -> None:
        self.node_ids = np.asarray(self.node_ids, dtype=np.int64)
        self.coefficients = np.asarray(self.coefficients, dtype=float)
        self.normals = np.asarray(self.normals, dtype=float)
        count = len(self.node_ids)
        self.clearance = np.broadcast_to(np.asarray(self.clearance, dtype=float), (count,)).copy()
        self.penalty = np.broadcast_to(np.asarray(self.penalty, dtype=float), (count,)).copy()
        if self.node_ids.ndim != 2 or self.node_ids.shape[1] != 4:
            raise ValueError("contact node_ids must have shape (N, 4)")
        if self.coefficients.shape != self.node_ids.shape:
            raise ValueError("contact coefficients must match node_ids")
        if self.normals.shape != (count, 3):
            raise ValueError("contact normals must have shape (N, 3)")
        lengths = np.linalg.norm(self.normals, axis=1)
        if np.any(lengths <= 0.0):
            raise ValueError("contact normals must be non-zero")
        self.normals /= lengths[:, None]
        if np.any(self.penalty <= 0.0):
            raise ValueError("contact penalty must be positive")

    @property
    def count(self) -> int:
        return int(len(self.node_ids))

    def gaps(self, points: np.ndarray) -> np.ndarray:
        support = np.sum(
            self.coefficients[:, :, None] * np.asarray(points)[self.node_ids], axis=1
        )
        return np.einsum("ij,ij->i", support, self.normals) - self.clearance

    def summary(self, points: np.ndarray) -> dict[str, object]:
        gaps = self.gaps(points)
        active = gaps < 0.0
        return {
            "constraint_count": self.count,
            "active_constraint_count": int(np.count_nonzero(active)),
            "minimum_gap": float(gaps.min()) if len(gaps) else 0.0,
            "maximum_penetration": float(np.maximum(-gaps, 0.0).max()) if len(gaps) else 0.0,
            "contact_energy": float(
                0.5 * np.sum(self.penalty[active] * gaps[active] ** 2)
            ),
        }

    def evaluate(self, points: np.ndarray) -> tuple["ContactConstraints", np.ndarray]:
        return self, self.gaps(points)


@dataclass
class DynamicContact:
    slave_nodes: np.ndarray
    master_triangles: np.ndarray
    penalty: float
    search_distance: float
    candidates: int = 12

    def __post_init__(self) -> None:
        self.slave_nodes = np.unique(np.asarray(self.slave_nodes, dtype=np.int64))
        self.master_triangles = np.asarray(self.master_triangles, dtype=np.int64)
        if self.master_triangles.ndim != 2 or self.master_triangles.shape[1] != 3:
            raise ValueError("master_triangles must have shape (N, 3)")
        if self.penalty <= 0.0 or self.search_distance <= 0.0:
            raise ValueError("dynamic contact penalty and search distance must be positive")

    @property
    def count(self) -> int:
        return int(len(self.slave_nodes))

    def evaluate(self, points: np.ndarray) -> tuple[ContactConstraints, np.ndarray]:
        points = np.asarray(points, dtype=float)
        triangles = points[self.master_triangles]
        centers = triangles.mean(axis=1)
        from scipy.spatial import cKDTree

        k = min(max(1, int(self.candidates)), len(triangles))
        _, candidates = cKDTree(centers).query(points[self.slave_nodes], k=k)
        if k == 1:
            candidates = candidates[:, None]
        node_rows: list[np.ndarray] = []
        coefficient_rows: list[np.ndarray] = []
        normal_rows: list[np.ndarray] = []
        gap_rows: list[float] = []
        for slave, nearby in zip(self.slave_nodes, candidates):
            point = points[slave]
            best = None
            for face_index in np.atleast_1d(nearby):
                nodes = self.master_triangles[int(face_index)]
                closest, barycentric = _closest_point_triangle(point, points[nodes])
                distance = float(np.linalg.norm(point - closest))
                if best is None or distance < best[0]:
                    normal = np.cross(
                        points[nodes[1]] - points[nodes[0]],
                        points[nodes[2]] - points[nodes[0]],
                    )
                    length = float(np.linalg.norm(normal))
                    if length > 0.0:
                        best = (distance, nodes, barycentric, normal / length, closest)
            if best is None:
                continue
            signed_gap = float(np.dot(point - best[4], best[3]))
            if best[0] > self.search_distance and signed_gap >= 0.0:
                continue
            node_rows.append(np.concatenate(([slave], best[1])))
            coefficient_rows.append(np.concatenate(([1.0], -best[2])))
            normal_rows.append(best[3])
            gap_rows.append(signed_gap)
        if not node_rows:
            empty = ContactConstraints(
                np.empty((0, 4), dtype=np.int64),
                np.empty((0, 4), dtype=float),
                np.empty((0, 3), dtype=float),
                np.empty(0),
                np.empty(0),
            )
            return empty, np.empty(0)
        state = ContactConstraints(
            np.asarray(node_rows),
            np.asarray(coefficient_rows),
            np.asarray(normal_rows),
            np.zeros(len(node_rows)),
            np.full(len(node_rows), self.penalty),
        )
        return state, np.asarray(gap_rows)

    def summary(self, points: np.ndarray) -> dict[str, object]:
        state, gaps = self.evaluate(points)
        active = gaps < 0.0
        return {
            "type": "dynamic_finite_sliding",
            "slave_node_count": self.count,
            "candidate_constraint_count": state.count,
            "active_constraint_count": int(np.count_nonzero(active)),
            "minimum_gap": float(gaps.min()) if len(gaps) else 0.0,
            "maximum_penetration": float(np.maximum(-gaps, 0.0).max()) if len(gaps) else 0.0,
            "contact_energy": float(0.5 * self.penalty * np.sum(gaps[active] ** 2)),
        }


def load_contact_constraints(path: str | Path, mesh: TetMesh) -> ContactConstraints | DynamicContact:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if raw.get("format") == "satmorph-dynamic-contact-v1":
        contact = DynamicContact(
            raw["slave_nodes"],
            raw["master_triangles"],
            float(raw["penalty"]),
            float(raw["search_distance"]),
            int(raw.get("candidates", 12)),
        )
        if contact.slave_nodes.size and contact.slave_nodes.max() >= mesh.n_points:
            raise ValueError("dynamic contact file contains a node outside the mesh")
        return contact
    constraints = ContactConstraints(
        raw["node_ids"],
        raw["coefficients"],
        raw["normals"],
        raw.get("clearance", 0.0),
        raw.get("penalty", 1.0e5),
    )
    if constraints.node_ids.size and (
        constraints.node_ids.min() < 0 or constraints.node_ids.max() >= mesh.n_points
    ):
        raise ValueError("contact file contains a node outside the mesh")
    return constraints


def save_contact_constraints(
    path: str | Path,
    constraints: ContactConstraints,
    *,
    metadata: dict[str, object] | None = None,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "satmorph-node-triangle-contact-v1",
        "node_ids": constraints.node_ids.tolist(),
        "coefficients": constraints.coefficients.tolist(),
        "normals": constraints.normals.tolist(),
        "clearance": constraints.clearance.tolist(),
        "penalty": constraints.penalty.tolist(),
        "metadata": metadata or {},
    }
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def save_dynamic_contact(
    path: str | Path,
    contact: DynamicContact,
    *,
    metadata: dict[str, object] | None = None,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "satmorph-dynamic-contact-v1",
        "slave_nodes": contact.slave_nodes.tolist(),
        "master_triangles": contact.master_triangles.tolist(),
        "penalty": contact.penalty,
        "search_distance": contact.search_distance,
        "candidates": contact.candidates,
        "metadata": metadata or {},
    }
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def build_label_dynamic_contact(
    mesh: TetMesh,
    slave_labels: list[int] | tuple[int, ...],
    master_labels: list[int] | tuple[int, ...],
    *,
    search_distance: float,
    penalty: float = 1.0e5,
    candidates: int = 12,
) -> DynamicContact:
    labels = mesh.cell_data.get("source_label")
    if labels is None:
        raise ValueError("automatic contact generation requires source_label cell data")
    labels = np.asarray(labels, dtype=np.int64)
    slave_faces = _boundary_faces(mesh, np.isin(labels, slave_labels))
    master_faces = _boundary_faces(mesh, np.isin(labels, master_labels))
    if len(slave_faces) == 0 or len(master_faces) == 0:
        raise ValueError("selected contact labels have no boundary faces")
    slave_nodes = np.setdiff1d(
        np.unique(slave_faces), np.unique(master_faces), assume_unique=False
    )
    if len(slave_nodes) == 0:
        raise ValueError("contact surfaces share all candidate nodes and are already conforming/bonded")
    return DynamicContact(slave_nodes, master_faces, penalty, search_distance, candidates)


def build_label_contact_constraints(
    mesh: TetMesh,
    slave_labels: list[int] | tuple[int, ...],
    master_labels: list[int] | tuple[int, ...],
    *,
    search_distance: float,
    penalty: float = 1.0e5,
    candidates: int = 12,
    max_constraints: int = 100_000,
) -> ContactConstraints:
    """Build reference-normal node-to-triangle constraints between label surfaces."""
    if search_distance <= 0.0:
        raise ValueError("search_distance must be positive")
    labels = mesh.cell_data.get("source_label")
    if labels is None:
        raise ValueError("automatic contact generation requires source_label cell data")
    labels = np.asarray(labels, dtype=np.int64)
    slave_faces = _boundary_faces(mesh, np.isin(labels, slave_labels))
    master_faces = _boundary_faces(mesh, np.isin(labels, master_labels))
    if len(slave_faces) == 0 or len(master_faces) == 0:
        raise ValueError("selected contact labels have no boundary faces")

    slave_nodes = np.setdiff1d(
        np.unique(slave_faces.ravel()), np.unique(master_faces.ravel()), assume_unique=False
    )
    master_centers = mesh.points[master_faces].mean(axis=1)
    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:
        raise RuntimeError("automatic contact generation requires scipy") from exc
    tree = cKDTree(master_centers)
    k = min(max(1, int(candidates)), len(master_faces))
    _, candidate_indices = tree.query(mesh.points[slave_nodes], k=k)
    if k == 1:
        candidate_indices = candidate_indices[:, None]

    rows: list[np.ndarray] = []
    coefficients: list[np.ndarray] = []
    normals: list[np.ndarray] = []
    for slave, nearby in zip(slave_nodes, candidate_indices):
        point = mesh.points[slave]
        best = None
        for face_index in np.atleast_1d(nearby):
            triangle = master_faces[int(face_index)]
            if slave in triangle:
                continue
            closest, barycentric = _closest_point_triangle(point, mesh.points[triangle])
            distance = float(np.linalg.norm(point - closest))
            if best is None or distance < best[0]:
                best = (distance, triangle, barycentric, closest)
        if best is None or best[0] <= 1.0e-12 or best[0] > search_distance:
            continue
        direction = point - best[3]
        normal = direction / np.linalg.norm(direction)
        rows.append(np.concatenate(([slave], best[1])))
        coefficients.append(np.concatenate(([1.0], -best[2])))
        normals.append(normal)
        if len(rows) >= max_constraints:
            break
    if not rows:
        raise ValueError("no separated contact candidates were found within search_distance")
    return ContactConstraints(
        np.asarray(rows),
        np.asarray(coefficients),
        np.asarray(normals),
        np.zeros(len(rows)),
        np.full(len(rows), float(penalty)),
    )


def _boundary_faces(mesh: TetMesh, selected: np.ndarray) -> np.ndarray:
    local_faces = np.asarray(((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1)))
    faces = mesh.tetra[selected][:, local_faces].reshape(-1, 3)
    if len(faces) == 0:
        return np.empty((0, 3), dtype=np.int64)
    keys = np.sort(faces, axis=1)
    subset_structured = np.ascontiguousarray(keys).view(
        np.dtype((np.void, keys.dtype.itemsize * 3))
    ).ravel()
    _, subset_inverse, subset_counts = np.unique(
        subset_structured, return_inverse=True, return_counts=True
    )
    mask = subset_counts[subset_inverse] == 1
    return faces[mask]


def _closest_point_triangle(point: np.ndarray, triangle: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Ericson's closest-point regions, retaining barycentric coordinates.
    a, b, c = triangle
    ab, ac, ap = b - a, c - a, point - a
    d1, d2 = ab @ ap, ac @ ap
    if d1 <= 0.0 and d2 <= 0.0:
        return a, np.asarray([1.0, 0.0, 0.0])
    bp = point - b
    d3, d4 = ab @ bp, ac @ bp
    if d3 >= 0.0 and d4 <= d3:
        return b, np.asarray([0.0, 1.0, 0.0])
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        return a + v * ab, np.asarray([1.0 - v, v, 0.0])
    cp = point - c
    d5, d6 = ab @ cp, ac @ cp
    if d6 >= 0.0 and d5 <= d6:
        return c, np.asarray([0.0, 0.0, 1.0])
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        return a + w * ac, np.asarray([1.0 - w, 0.0, w])
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return b + w * (c - b), np.asarray([0.0, 1.0 - w, w])
    denominator = 1.0 / (va + vb + vc)
    v, w = vb * denominator, vc * denominator
    return a + ab * v + ac * w, np.asarray([1.0 - v - w, v, w])

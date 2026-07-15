from __future__ import annotations

from pathlib import Path

import numpy as np

from .surface_map import SurfaceMesh


def compute_point_normals(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    triangles = np.asarray(triangles, dtype=np.int64)
    normals = np.zeros_like(points, dtype=float)
    if len(points) == 0 or len(triangles) == 0:
        return normals

    face_points = points[triangles]
    face_normals = np.cross(
        face_points[:, 1] - face_points[:, 0],
        face_points[:, 2] - face_points[:, 0],
    )
    lengths = np.linalg.norm(face_normals, axis=1)
    valid = lengths > 0.0
    face_normals[valid] /= lengths[valid, None]
    np.add.at(normals, triangles[:, 0], face_normals)
    np.add.at(normals, triangles[:, 1], face_normals)
    np.add.at(normals, triangles[:, 2], face_normals)
    normal_lengths = np.linalg.norm(normals, axis=1)
    valid_points = normal_lengths > 0.0
    normals[valid_points] /= normal_lengths[valid_points, None]
    return normals


def smooth_surface(
    surface: SurfaceMesh,
    *,
    method: str = "taubin",
    iterations: int = 20,
    laplacian_lambda: float = 0.35,
    taubin_lambda: float = 0.5,
    taubin_mu: float = -0.53,
) -> SurfaceMesh:
    if iterations < 0:
        raise ValueError("iterations must be non-negative")
    if method not in {"none", "laplacian", "taubin"}:
        raise ValueError("smoothing method must be 'none', 'laplacian', or 'taubin'")
    if method == "none" or iterations == 0 or surface.n_triangles == 0:
        return SurfaceMesh(surface.points.copy(), surface.triangles.copy())

    points = surface.points.copy()
    adjacency = _adjacency_matrix(surface.triangles, surface.n_points)

    if method == "laplacian":
        for _ in range(iterations):
            points += laplacian_lambda * _laplacian(points, adjacency)
    else:
        for _ in range(iterations):
            points += taubin_lambda * _laplacian(points, adjacency)
            points += taubin_mu * _laplacian(points, adjacency)
    return SurfaceMesh(points, surface.triangles.copy())


def save_surface_mesh(path: str | Path, surface: SurfaceMesh, *, normals: bool = True) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    point_normals = compute_point_normals(surface.points, surface.triangles) if normals else None
    if path.suffix.lower() == ".npz":
        payload = {"points": surface.points, "triangles": surface.triangles}
        if point_normals is not None:
            payload["normals"] = point_normals
        np.savez_compressed(path, **payload)
        return

    try:
        import meshio
    except ImportError as exc:
        raise RuntimeError("meshio is required to write non-NPZ surface meshes") from exc

    cells = [("triangle", surface.triangles)] if surface.n_triangles else []
    point_data = {}
    if point_normals is not None and path.suffix.lower() not in {".stl", ".obj"}:
        point_data["Normals"] = point_normals
    meshio.write(path, meshio.Mesh(surface.points, cells, point_data=point_data))


def _adjacency_matrix(triangles: np.ndarray, n_points: int):
    pairs = np.vstack(
        (
            triangles[:, [0, 1]],
            triangles[:, [1, 2]],
            triangles[:, [2, 0]],
        )
    )
    directed = np.vstack((pairs, pairs[:, ::-1]))
    try:
        from scipy.sparse import coo_matrix
    except ImportError:
        neighbors: list[set[int]] = [set() for _ in range(n_points)]
        for left, right in directed:
            neighbors[int(left)].add(int(right))
        return neighbors

    data = np.ones(len(directed), dtype=float)
    matrix = coo_matrix((data, (directed[:, 0], directed[:, 1])), shape=(n_points, n_points))
    matrix.sum_duplicates()
    degree = np.asarray(matrix.sum(axis=1)).ravel()
    degree[degree == 0.0] = 1.0
    return matrix.tocsr(), degree


def _laplacian(points: np.ndarray, adjacency) -> np.ndarray:
    if isinstance(adjacency, tuple):
        matrix, degree = adjacency
        return matrix @ points / degree[:, None] - points
    out = np.zeros_like(points)
    for index, neighbors in enumerate(adjacency):
        if neighbors:
            out[index] = points[list(neighbors)].mean(axis=0) - points[index]
    return out

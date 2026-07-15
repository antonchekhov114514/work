from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .mesh import TetMesh


@dataclass
class SurfaceMesh:
    points: np.ndarray
    triangles: np.ndarray

    def __post_init__(self) -> None:
        self.points = np.asarray(self.points, dtype=float)
        self.triangles = np.asarray(self.triangles, dtype=np.int64)
        if self.points.ndim != 2 or self.points.shape[1] != 3:
            raise ValueError("surface points must have shape (N, 3)")
        if self.triangles.size == 0:
            self.triangles = np.empty((0, 3), dtype=np.int64)
        if self.triangles.ndim != 2 or self.triangles.shape[1] != 3:
            raise ValueError("surface triangles must have shape (M, 3)")
        if self.triangles.size and (
            self.triangles.min() < 0 or self.triangles.max() >= len(self.points)
        ):
            raise ValueError("surface triangles contain an invalid point index")

    @property
    def n_points(self) -> int:
        return int(len(self.points))

    @property
    def n_triangles(self) -> int:
        return int(len(self.triangles))

    def triangle_centers(self) -> np.ndarray:
        if self.n_triangles == 0:
            return np.empty((0, 3), dtype=float)
        return self.points[self.triangles].mean(axis=1)

    def copy_with_points(self, points: np.ndarray) -> "SurfaceMesh":
        return SurfaceMesh(points, self.triangles.copy())


@dataclass
class SurfaceMapResult:
    points: np.ndarray
    displacement: np.ndarray
    cell_index: np.ndarray
    barycentric: np.ndarray
    inside: np.ndarray
    residual: np.ndarray
    center_points: np.ndarray | None = None
    center_displacement: np.ndarray | None = None
    center_cell_index: np.ndarray | None = None
    center_barycentric: np.ndarray | None = None
    center_inside: np.ndarray | None = None
    center_residual: np.ndarray | None = None

    def summary(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "mapped_points": int(len(self.points)),
            "inside_points": int(np.count_nonzero(self.inside)),
            "outside_points": int(len(self.inside) - np.count_nonzero(self.inside)),
            "maximum_point_residual": float(np.max(self.residual)) if len(self.residual) else 0.0,
            "maximum_point_displacement": (
                float(np.linalg.norm(self.displacement, axis=1).max())
                if len(self.displacement)
                else 0.0
            ),
        }
        if self.center_points is not None:
            center_inside = self.center_inside
            center_residual = self.center_residual
            center_displacement = self.center_displacement
            payload.update(
                {
                    "mapped_triangle_centers": int(len(self.center_points)),
                    "inside_triangle_centers": int(np.count_nonzero(center_inside)),
                    "outside_triangle_centers": int(
                        len(center_inside) - np.count_nonzero(center_inside)
                    ),
                    "maximum_center_residual": (
                        float(np.max(center_residual)) if len(center_residual) else 0.0
                    ),
                    "maximum_center_displacement": (
                        float(np.linalg.norm(center_displacement, axis=1).max())
                        if len(center_displacement)
                        else 0.0
                    ),
                }
            )
        return payload


def load_surface(path: str | Path) -> SurfaceMesh:
    path = Path(path)
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as data:
            points = data["points"]
            if "triangles" in data:
                triangles = data["triangles"]
            elif "faces" in data:
                triangles = data["faces"]
            else:
                triangles = np.empty((0, 3), dtype=np.int64)
        return SurfaceMesh(points, triangles)

    try:
        import meshio
    except ImportError as exc:
        raise RuntimeError("meshio is required to read non-NPZ surface meshes") from exc

    source = meshio.read(path)
    triangle_blocks: list[np.ndarray] = []
    for block in source.cells:
        if block.type == "triangle":
            triangle_blocks.append(np.asarray(block.data, dtype=np.int64))
        elif block.type == "quad":
            quad = np.asarray(block.data, dtype=np.int64)
            triangle_blocks.append(quad[:, [0, 1, 2]])
            triangle_blocks.append(quad[:, [0, 2, 3]])
    triangles = (
        np.vstack(triangle_blocks)
        if triangle_blocks
        else np.empty((0, 3), dtype=np.int64)
    )
    return SurfaceMesh(np.asarray(source.points[:, :3], dtype=float), triangles)


def save_surface_result(
    path: str | Path,
    surface: SurfaceMesh,
    result: SurfaceMapResult,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".npz":
        deformed_normals = _compute_point_normals(result.points, surface.triangles)
        np.savez_compressed(
            path,
            points=surface.points,
            triangles=surface.triangles,
            deformed_points=result.points,
            displacement=result.displacement,
            deformed_normals=deformed_normals,
            cell_index=result.cell_index,
            barycentric=result.barycentric,
            inside=result.inside,
            residual=result.residual,
            center_points=(
                np.empty((0, 3), dtype=float)
                if result.center_points is None
                else result.center_points
            ),
            center_displacement=(
                np.empty((0, 3), dtype=float)
                if result.center_displacement is None
                else result.center_displacement
            ),
            center_cell_index=(
                np.empty(0, dtype=np.int64)
                if result.center_cell_index is None
                else result.center_cell_index
            ),
            center_barycentric=(
                np.empty((0, 4), dtype=float)
                if result.center_barycentric is None
                else result.center_barycentric
            ),
            center_inside=(
                np.empty(0, dtype=bool)
                if result.center_inside is None
                else result.center_inside
            ),
            center_residual=(
                np.empty(0, dtype=float)
                if result.center_residual is None
                else result.center_residual
            ),
        )
        return

    try:
        import meshio
    except ImportError as exc:
        raise RuntimeError("meshio is required to write non-NPZ surface meshes") from exc

    cells = [("triangle", surface.triangles)] if surface.n_triangles else []
    deformed_normals = _compute_point_normals(result.points, surface.triangles)
    point_data = {
        "displacement": result.displacement,
        "mapped_cell": result.cell_index,
        "inside_tet": result.inside.astype(np.int8),
        "map_residual": result.residual,
        "Normals": deformed_normals,
    }
    suffix = path.suffix.lower()
    if suffix in {".stl", ".obj", ".ply"}:
        point_data = {}
    meshio.write(path, meshio.Mesh(result.points, cells, point_data=point_data))


def save_center_result(path: str | Path, result: SurfaceMapResult) -> None:
    if result.center_points is None:
        raise ValueError("triangle centers were not mapped")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        center_points=result.center_points,
        center_displacement=result.center_displacement,
        deformed_center_points=result.center_points + result.center_displacement,
        center_cell_index=result.center_cell_index,
        center_barycentric=result.center_barycentric,
        center_inside=result.center_inside,
        center_residual=result.center_residual,
    )


def map_surface(
    mesh: TetMesh,
    displacement: np.ndarray,
    surface: SurfaceMesh,
    *,
    candidate_cells: int = 64,
    outside_mode: str = "clamp",
    tolerance: float = 1.0e-8,
    map_centers: bool = True,
) -> SurfaceMapResult:
    point_map = map_points_to_tet_displacement(
        mesh,
        displacement,
        surface.points,
        candidate_cells=candidate_cells,
        outside_mode=outside_mode,
        tolerance=tolerance,
    )
    centers = surface.triangle_centers()
    center_map = None
    if map_centers and len(centers):
        center_map = map_points_to_tet_displacement(
            mesh,
            displacement,
            centers,
            candidate_cells=candidate_cells,
            outside_mode=outside_mode,
            tolerance=tolerance,
        )
    return SurfaceMapResult(
        points=surface.points + point_map.displacement,
        displacement=point_map.displacement,
        cell_index=point_map.cell_index,
        barycentric=point_map.barycentric,
        inside=point_map.inside,
        residual=point_map.residual,
        center_points=centers if center_map is not None else None,
        center_displacement=None if center_map is None else center_map.displacement,
        center_cell_index=None if center_map is None else center_map.cell_index,
        center_barycentric=None if center_map is None else center_map.barycentric,
        center_inside=None if center_map is None else center_map.inside,
        center_residual=None if center_map is None else center_map.residual,
    )


@dataclass
class _PointMap:
    displacement: np.ndarray
    cell_index: np.ndarray
    barycentric: np.ndarray
    inside: np.ndarray
    residual: np.ndarray


def map_points_to_tet_displacement(
    mesh: TetMesh,
    displacement: np.ndarray,
    query_points: np.ndarray,
    *,
    candidate_cells: int = 64,
    outside_mode: str = "clamp",
    tolerance: float = 1.0e-8,
) -> _PointMap:
    displacement = np.asarray(displacement, dtype=float)
    query_points = np.asarray(query_points, dtype=float)
    if displacement.shape != mesh.points.shape:
        raise ValueError("displacement must have the same shape as mesh.points")
    if query_points.ndim != 2 or query_points.shape[1] != 3:
        raise ValueError("query_points must have shape (N, 3)")
    if candidate_cells < 1:
        raise ValueError("candidate_cells must be at least one")
    if outside_mode not in {"clamp", "linear", "fail"}:
        raise ValueError("outside_mode must be 'clamp', 'linear', or 'fail'")

    _, _, inv_dm = mesh.reference_geometry()
    tet_points = mesh.points[mesh.tetra]
    centroids = tet_points.mean(axis=1)
    candidate_count = min(int(candidate_cells), mesh.n_cells)
    candidate_indices = _candidate_indices(centroids, query_points, candidate_count)

    out_displacement = np.empty_like(query_points)
    out_cells = np.empty(len(query_points), dtype=np.int64)
    out_bary = np.empty((len(query_points), 4), dtype=float)
    out_inside = np.zeros(len(query_points), dtype=bool)
    out_residual = np.empty(len(query_points), dtype=float)

    for row, point in enumerate(query_points):
        cell, bary, inside, residual = _choose_cell(
            point, mesh, inv_dm, candidate_indices[row], tolerance
        )
        if not inside:
            if outside_mode == "fail":
                raise ValueError(
                    f"query point {row} is outside the tetrahedral mesh; "
                    "use outside_mode='clamp' or increase candidate_cells"
                )
            if outside_mode == "clamp":
                bary = _clamp_barycentric(bary)
        nodes = mesh.tetra[cell]
        out_displacement[row] = bary @ displacement[nodes]
        out_cells[row] = cell
        out_bary[row] = bary
        out_inside[row] = inside
        out_residual[row] = residual
    return _PointMap(out_displacement, out_cells, out_bary, out_inside, out_residual)


def _candidate_indices(
    centroids: np.ndarray, query_points: np.ndarray, candidate_count: int
) -> np.ndarray:
    if len(query_points) == 0:
        return np.empty((0, candidate_count), dtype=np.int64)
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        distance2 = np.sum((query_points[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        return np.argsort(distance2, axis=1)[:, :candidate_count]

    tree = cKDTree(centroids)
    _, indices = tree.query(query_points, k=candidate_count)
    return np.asarray(indices, dtype=np.int64).reshape((len(query_points), candidate_count))


def _choose_cell(
    point: np.ndarray,
    mesh: TetMesh,
    inv_dm: np.ndarray,
    candidates: np.ndarray,
    tolerance: float,
) -> tuple[int, np.ndarray, bool, float]:
    best_cell = int(candidates[0])
    best_bary = np.zeros(4, dtype=float)
    best_residual = np.inf
    best_inside_score = -np.inf

    for raw_cell in candidates:
        cell = int(raw_cell)
        bary = _barycentric(point, mesh.points[mesh.tetra[cell, 0]], inv_dm[cell])
        minimum = float(np.min(bary))
        residual = _outside_residual(bary)
        if minimum >= -tolerance:
            score = minimum - residual
            if score > best_inside_score:
                best_cell = cell
                best_bary = bary
                best_residual = residual
                best_inside_score = score
        elif best_inside_score == -np.inf and residual < best_residual:
            best_cell = cell
            best_bary = bary
            best_residual = residual

    inside = best_inside_score != -np.inf
    return best_cell, best_bary, inside, float(best_residual)


def _barycentric(point: np.ndarray, x0: np.ndarray, inv_dm: np.ndarray) -> np.ndarray:
    local = inv_dm @ (point - x0)
    return np.asarray([1.0 - local.sum(), local[0], local[1], local[2]], dtype=float)


def _outside_residual(bary: np.ndarray) -> float:
    low = np.minimum(bary, 0.0)
    high = np.maximum(bary - 1.0, 0.0)
    return float(np.linalg.norm(low) + np.linalg.norm(high) + abs(np.sum(bary) - 1.0))


def _clamp_barycentric(bary: np.ndarray) -> np.ndarray:
    clipped = np.clip(bary, 0.0, 1.0)
    total = float(np.sum(clipped))
    if total <= 0.0:
        clipped[np.argmax(bary)] = 1.0
        return clipped
    return clipped / total


def _compute_point_normals(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
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

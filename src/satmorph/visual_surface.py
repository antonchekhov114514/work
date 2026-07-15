from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .surface_map import SurfaceMesh
from .surface_ops import compute_point_normals, save_surface_mesh, smooth_surface
from .voxel_convert import _block_occupancy, _coarse_edges, load_voxel_mat, surface_from_blocks


def extract_visual_surface_from_voxel_mat(
    input_path: str | Path,
    output_path: str | Path,
    *,
    report_path: str | Path | None = None,
    variable: str = "MaterialLabelGrid",
    axis_keys: tuple[str, str, str] = ("Axis0", "Axis1", "Axis2"),
    axis_unit: str = "m",
    output_unit: str = "m",
    voxel_size_mm: float = 1.0,
    surface_stride: int = 2,
    method: str = "marching-cubes",
    pre_smooth_sigma: float = 0.0,
    smooth_method: str = "taubin",
    smooth_iterations: int = 20,
    laplacian_lambda: float = 0.35,
    taubin_lambda: float = 0.5,
    taubin_mu: float = -0.53,
) -> dict[str, object]:
    if surface_stride < 1:
        raise ValueError("surface_stride must be at least one")
    if method not in {"marching-cubes", "blocks"}:
        raise ValueError("method must be 'marching-cubes' or 'blocks'")
    if pre_smooth_sigma < 0.0:
        raise ValueError("pre_smooth_sigma must be non-negative")

    labels, source_axes = load_voxel_mat(
        input_path,
        variable=variable,
        axis_keys=axis_keys,
        axis_unit=axis_unit,
        output_unit=output_unit,
        voxel_size_mm=voxel_size_mm,
    )
    occupied = _block_occupancy(labels, surface_stride)
    surface_axes = tuple(
        _coarse_edges(axis, labels.shape[index], surface_stride)
        for index, axis in enumerate(source_axes)
    )

    if method == "marching-cubes":
        surface = _marching_cubes_surface(
            occupied,
            surface_axes,
            pre_smooth_sigma=pre_smooth_sigma,
        )
        method_used = "marching-cubes"
    else:
        surface = surface_from_blocks(occupied, surface_axes)
        method_used = "blocks"

    raw_surface = surface
    surface = smooth_surface(
        surface,
        method=smooth_method,
        iterations=smooth_iterations,
        laplacian_lambda=laplacian_lambda,
        taubin_lambda=taubin_lambda,
        taubin_mu=taubin_mu,
    )
    normals = compute_point_normals(surface.points, surface.triangles)
    save_surface_mesh(output_path, surface, normals=True)

    bounds_min = surface.points.min(axis=0) if surface.n_points else np.zeros(3)
    bounds_max = surface.points.max(axis=0) if surface.n_points else np.zeros(3)
    report: dict[str, object] = {
        "input": str(Path(input_path)),
        "label_variable": variable,
        "source_grid_shape": list(labels.shape),
        "output": str(Path(output_path)),
        "source_axis_unit": axis_unit,
        "output_coordinate_unit": output_unit,
        "surface_stride_voxels": surface_stride,
        "method": method_used,
        "pre_smooth_sigma": pre_smooth_sigma,
        "smoothing": {
            "method": smooth_method,
            "iterations": smooth_iterations,
            "laplacian_lambda": laplacian_lambda,
            "taubin_lambda": taubin_lambda,
            "taubin_mu": taubin_mu,
        },
        "raw_surface": {
            "points": raw_surface.n_points,
            "triangles": raw_surface.n_triangles,
        },
        "surface": {
            "points": surface.n_points,
            "triangles": surface.n_triangles,
            "bounds_min": [float(value) for value in bounds_min],
            "bounds_max": [float(value) for value in bounds_max],
            "normal_length_min": float(np.linalg.norm(normals, axis=1).min())
            if len(normals)
            else 0.0,
            "normal_length_max": float(np.linalg.norm(normals, axis=1).max())
            if len(normals)
            else 0.0,
        },
    }

    if report_path is not None:
        report_file = Path(report_path)
        report_file.parent.mkdir(parents=True, exist_ok=True)
        report_file.write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    return report


def _marching_cubes_surface(
    occupied: np.ndarray,
    axes: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    pre_smooth_sigma: float,
) -> SurfaceMesh:
    try:
        from skimage.measure import marching_cubes
    except ImportError as exc:
        raise RuntimeError(
            "marching-cubes visual surfaces require scikit-image; install with "
            "python -m pip install -e '.[visual]' or use --method blocks"
        ) from exc

    volume = np.pad(occupied.astype(np.float32), 1, mode="constant")
    if pre_smooth_sigma > 0.0:
        try:
            from scipy.ndimage import gaussian_filter
        except ImportError as exc:
            raise RuntimeError("pre-smoothing requires scipy.ndimage") from exc
        volume = gaussian_filter(volume, sigma=pre_smooth_sigma)

    vertices, faces, _, _ = marching_cubes(volume, level=0.5, allow_degenerate=False)
    edge_coordinates = vertices - 0.5
    points = np.column_stack(
        [
            _axis_values_from_edge_coordinates(axes[axis], edge_coordinates[:, axis])
            for axis in range(3)
        ]
    )
    return SurfaceMesh(points, np.asarray(faces, dtype=np.int64))


def _axis_values_from_edge_coordinates(edges: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
    upper = len(edges) - 1
    clipped = np.clip(coordinates, 0.0, float(upper))
    lower = np.floor(clipped).astype(np.int64)
    lower = np.minimum(lower, upper - 1)
    fraction = clipped - lower
    return edges[lower] * (1.0 - fraction) + edges[lower + 1] * fraction

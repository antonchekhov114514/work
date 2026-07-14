from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .mesh import TetMesh


POINT_KEYS = (
    "points", "nodes", "node", "vertices", "vertex", "verts",
    "coordinates", "coords", "coord", "xyz", "x", "v",
)
TETRA_KEYS = (
    "tetra", "tetras", "tet", "tets", "tet4", "elements", "element",
    "elems", "elem", "cells", "cell", "connectivity", "e", "t",
)
TAG_KEYS = (
    "cell_tags", "tags", "labels", "label", "materials", "material",
    "regions", "region", "tissues", "tissue", "element_tags",
    "elem_tags", "domains", "domain",
)
SURFACE_POINT_KEYS = (
    "surface_points", "surf_points", "skin_points", "skin_vertices",
    "surface_vertices", "points_surface", "vertices_surface",
)
SURFACE_FACE_KEYS = (
    "triangles", "triangle", "tris", "tri", "faces", "face",
    "surface_faces", "skin_faces", "faces_surface",
)


def load_mat_arrays(path: str | Path) -> dict[str, np.ndarray]:
    """Load numeric top-level arrays from MATLAB v7.2 or v7.3 files."""
    path = Path(path)
    try:
        from scipy.io import loadmat

        raw = loadmat(path, squeeze_me=True, struct_as_record=False)
        return {
            str(key): np.asarray(value)
            for key, value in raw.items()
            if not key.startswith("__") and _is_numeric_array(value)
        }
    except (NotImplementedError, ValueError, OSError) as scipy_error:
        try:
            import h5py
        except ImportError as exc:
            raise RuntimeError(
                "This appears to be a MATLAB v7.3 HDF5 file. "
                "Install optional support with: python -m pip install -e '.[mat]'"
            ) from exc

        arrays: dict[str, np.ndarray] = {}
        try:
            with h5py.File(path, "r") as source:
                source.visititems(lambda name, obj: _collect_hdf5(name, obj, arrays))
        except OSError:
            raise scipy_error
        return arrays


def _is_numeric_array(value: Any) -> bool:
    array = np.asarray(value)
    return array.dtype.kind in "biufc" and array.size > 0


def _collect_hdf5(name: str, obj: Any, arrays: dict[str, np.ndarray]) -> None:
    if not hasattr(obj, "shape") or getattr(obj, "dtype", None) is None:
        return
    if obj.dtype.kind not in "biufc" or obj.size == 0:
        return
    value = np.asarray(obj)
    if value.ndim >= 2:
        value = value.transpose(tuple(reversed(range(value.ndim))))
    arrays[name] = value
    arrays.setdefault(name.rsplit("/", 1)[-1], value)


def describe_arrays(arrays: Mapping[str, np.ndarray]) -> list[dict[str, Any]]:
    return [
        {"name": name, "shape": list(value.shape), "dtype": str(value.dtype)}
        for name, value in sorted(arrays.items())
    ]


def _find_key(
    arrays: Mapping[str, np.ndarray], requested: str | None, candidates: tuple[str, ...]
) -> str | None:
    if requested is not None:
        if requested not in arrays:
            known = ", ".join(sorted(arrays)) or "<none>"
            raise KeyError(f"MAT variable {requested!r} was not found; available: {known}")
        return requested
    lower = {name.lower(): name for name in arrays}
    return next((lower[name] for name in candidates if name in lower), None)


def _points(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float).squeeze()
    if array.ndim == 1 and array.size == 3:
        array = array.reshape(1, 3)
    if array.ndim != 2:
        raise ValueError(f"{name!r} must be a two-dimensional coordinate array")
    if array.shape[1] != 3 and array.shape[0] == 3:
        array = array.T
    if array.shape[1] != 3:
        raise ValueError(f"{name!r} must have shape (N, 3) or (3, N), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name!r} contains NaN or infinite coordinates")
    return array


def _connectivity(value: np.ndarray, name: str, width: int) -> np.ndarray:
    raw = np.asarray(value).squeeze()
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    if raw.ndim != 2:
        raise ValueError(f"{name!r} must be a two-dimensional connectivity array")
    allowed_rows = {width, 10} if width == 4 else {width}
    if raw.shape[1] < width and raw.shape[0] >= width:
        raw = raw.T
    elif raw.shape[0] in allowed_rows and raw.shape[1] not in allowed_rows:
        raw = raw.T
    if raw.shape[1] < width:
        raise ValueError(f"{name!r} must provide at least {width} node indices per cell")
    if not np.all(np.isfinite(raw)) or not np.all(raw == np.rint(raw)):
        raise ValueError(f"{name!r} contains non-integer node indices")
    return np.asarray(raw[:, :width], dtype=np.int64)


def _tags(value: np.ndarray, name: str, count: int) -> np.ndarray:
    array = np.asarray(value).squeeze().reshape(-1)
    if len(array) != count:
        raise ValueError(f"{name!r} has {len(array)} values, expected {count}")
    if not np.all(np.isfinite(array)) or not np.all(array == np.rint(array)):
        raise ValueError(f"{name!r} must contain integer tissue labels")
    return np.asarray(array, dtype=np.int64)


def _zero_based(connectivity: np.ndarray, point_count: int, mode: str) -> tuple[np.ndarray, bool]:
    minimum = int(connectivity.min())
    maximum = int(connectivity.max())
    one_based = mode == "one" or (mode == "auto" and minimum == 1 and maximum <= point_count)
    result = connectivity - 1 if one_based else connectivity.copy()
    if result.min() < 0 or result.max() >= point_count:
        raise ValueError(
            f"connectivity indices are outside the valid point range after {mode!r} base handling"
        )
    return result, one_based


def convert_mat(
    input_path: str | Path,
    output_path: str | Path,
    *,
    points_key: str | None = None,
    tetra_key: str | None = None,
    tags_key: str | None = None,
    surface_points_key: str | None = None,
    surface_faces_key: str | None = None,
    surface_output: str | Path | None = None,
    report_path: str | Path | None = None,
    index_base: str = "auto",
    tag_names: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    arrays = load_mat_arrays(input_path)
    selected = {
        "points": _find_key(arrays, points_key, POINT_KEYS),
        "tetra": _find_key(arrays, tetra_key, TETRA_KEYS),
        "cell_tags": _find_key(arrays, tags_key, TAG_KEYS),
        "surface_points": _find_key(arrays, surface_points_key, SURFACE_POINT_KEYS),
        "surface_faces": _find_key(arrays, surface_faces_key, SURFACE_FACE_KEYS),
    }
    missing = [name for name in ("points", "tetra", "cell_tags") if selected[name] is None]
    if missing:
        raise ValueError(
            "could not detect required MAT arrays: " + ", ".join(missing)
            + "; use --list-variables and the corresponding --*-key options"
        )

    points = _points(arrays[selected["points"]], selected["points"])
    tetra_raw = _connectivity(arrays[selected["tetra"]], selected["tetra"], 4)
    tetra, tetra_one_based = _zero_based(tetra_raw, len(points), index_base)
    cell_tags = _tags(arrays[selected["cell_tags"]], selected["cell_tags"], len(tetra))
    mesh = TetMesh(points, tetra, cell_tags, dict(tag_names or {}))

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        points=mesh.points,
        tetra=mesh.tetra,
        cell_tags=mesh.cell_tags,
        tag_names_json=np.asarray(json.dumps(mesh.tag_names)),
    )

    volumes = mesh.cell_volumes()
    report: dict[str, Any] = {
        "input": str(Path(input_path)),
        "output": str(output),
        "source_keys": selected,
        "mesh": {
            "points": mesh.n_points,
            "tetrahedra": mesh.n_cells,
            "matlab_one_based_indices_converted": tetra_one_based,
            "tag_counts": {
                str(tag): int(np.count_nonzero(mesh.cell_tags == tag))
                for tag in np.unique(mesh.cell_tags)
            },
            "minimum_tetra_volume": float(volumes.min()),
            "maximum_tetra_volume": float(volumes.max()),
        },
    }

    has_surface = selected["surface_points"] is not None or selected["surface_faces"] is not None
    if has_surface:
        if selected["surface_points"] is None or selected["surface_faces"] is None:
            raise ValueError("surface conversion requires both surface points and surface faces")
        if surface_output is None:
            raise ValueError("surface arrays were selected; provide --surface-output")
        surface_points = _points(
            arrays[selected["surface_points"]], selected["surface_points"]
        )
        faces_raw = _connectivity(
            arrays[selected["surface_faces"]], selected["surface_faces"], 3
        )
        triangles, faces_one_based = _zero_based(faces_raw, len(surface_points), index_base)
        surface_path = Path(surface_output)
        surface_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(surface_path, points=surface_points, triangles=triangles)
        report["surface"] = {
            "output": str(surface_path),
            "points": int(len(surface_points)),
            "triangles": int(len(triangles)),
            "matlab_one_based_indices_converted": faces_one_based,
        }

    if report_path is not None:
        report_file = Path(report_path)
        report_file.parent.mkdir(parents=True, exist_ok=True)
        report_file.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.io import loadmat

from .demo import BONE, SAT, SKIN, SOFT
from .io import save_npz
from .mesh import TetMesh
from .surface_map import SurfaceMesh


DEFAULT_SAT_LABELS = (1,)
DEFAULT_SKIN_LABELS = (2,)
DEFAULT_BONE_LABELS = (19, 68, 69, 70)
TAG_NAMES = {"BONE": BONE, "SOFT": SOFT, "SAT": SAT, "SKIN": SKIN}

_TET_PATTERN = np.asarray(
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
_CUBE_OFFSETS = np.asarray(
    [
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1],
    ],
    dtype=np.int64,
)


def _as_labels(value: np.ndarray, name: str) -> np.ndarray:
    labels = np.asarray(value)
    if labels.ndim != 3:
        raise ValueError(f"{name!r} must be a three-dimensional label grid")
    if labels.dtype.kind not in "biu" and not np.all(labels == np.rint(labels)):
        raise ValueError(f"{name!r} must contain integer labels")
    labels = np.asarray(labels, dtype=np.int32)
    if labels.min() < 0:
        raise ValueError(f"{name!r} contains negative labels")
    return labels


def _axis_edges(
    value: np.ndarray | None,
    size: int,
    *,
    axis_unit: str,
    output_unit: str,
    voxel_size_mm: float,
) -> np.ndarray:
    if value is None:
        edges_m = np.arange(size + 1, dtype=float) * voxel_size_mm * 1.0e-3
    else:
        axis = np.asarray(value, dtype=float).reshape(-1)
        if axis.size == size + 1:
            edges = axis
        elif axis.size == size:
            if size < 2:
                step = voxel_size_mm * (1.0e-3 if axis_unit == "m" else 1.0)
                edges = np.asarray([axis[0] - step / 2.0, axis[0] + step / 2.0])
            else:
                middle = 0.5 * (axis[:-1] + axis[1:])
                edges = np.concatenate(
                    ([axis[0] - (middle[0] - axis[0])], middle, [axis[-1] + (axis[-1] - middle[-1])])
                )
        else:
            raise ValueError(
                f"axis length {axis.size} is incompatible with grid size {size}"
            )
        edges_m = edges if axis_unit == "m" else edges * 1.0e-3
    return edges_m if output_unit == "m" else edges_m * 1000.0


def load_voxel_mat(
    path: str | Path,
    *,
    variable: str = "MaterialLabelGrid",
    axis_keys: tuple[str, str, str] = ("Axis0", "Axis1", "Axis2"),
    axis_unit: str = "m",
    output_unit: str = "m",
    voxel_size_mm: float = 1.0,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    requested = [variable, *axis_keys]
    raw = loadmat(path, variable_names=requested)
    if variable not in raw:
        raise KeyError(f"MAT variable {variable!r} was not found in {path}")
    labels = _as_labels(raw[variable], variable)
    axes = tuple(
        _axis_edges(
            raw.get(key),
            labels.shape[index],
            axis_unit=axis_unit,
            output_unit=output_unit,
            voxel_size_mm=voxel_size_mm,
        )
        for index, key in enumerate(axis_keys)
    )
    return labels, axes


def map_anatomical_regions(
    labels: np.ndarray,
    *,
    sat_labels: Iterable[int] = DEFAULT_SAT_LABELS,
    skin_labels: Iterable[int] = DEFAULT_SKIN_LABELS,
    bone_labels: Iterable[int] = DEFAULT_BONE_LABELS,
) -> np.ndarray:
    groups = {
        SAT: {int(value) for value in sat_labels},
        SKIN: {int(value) for value in skin_labels},
        BONE: {int(value) for value in bone_labels},
    }
    all_special = [value for values in groups.values() for value in values]
    if len(all_special) != len(set(all_special)):
        raise ValueError("SAT, skin, and bone source-label groups must not overlap")
    maximum = int(labels.max())
    lookup = np.full(maximum + 1, SOFT, dtype=np.uint8)
    lookup[0] = 0
    for target, source_values in groups.items():
        valid = [value for value in source_values if 0 <= value <= maximum]
        lookup[valid] = target
    return lookup[labels]


def _block_region_majority(
    regions: np.ndarray,
    stride: int,
    *,
    envelope_fraction: float = 0.25,
    skin_fraction: float = 0.25,
) -> np.ndarray:
    if stride < 1:
        raise ValueError("stride must be at least 1")
    if not 0.0 <= envelope_fraction <= 1.0 or not 0.0 <= skin_fraction <= 1.0:
        raise ValueError("region fractions must be between 0 and 1")
    nx, ny, nz = regions.shape
    shape = tuple((value + stride - 1) // stride for value in regions.shape)
    counts = np.zeros((4, *shape), dtype=np.uint32)
    valid_counts = np.zeros(shape, dtype=np.uint32)
    pad_y = shape[1] * stride - ny
    pad_z = shape[2] * stride - nz
    for i in range(shape[0]):
        slab = regions[i * stride : min((i + 1) * stride, nx)]
        pad_x = stride - slab.shape[0]
        padded = np.pad(slab, ((0, pad_x), (0, pad_y), (0, pad_z)))
        blocks = padded.reshape(stride, shape[1], stride, shape[2], stride)
        valid = np.ones(slab.shape, dtype=np.uint8)
        valid = np.pad(valid, ((0, pad_x), (0, pad_y), (0, pad_z)))
        valid_blocks = valid.reshape(stride, shape[1], stride, shape[2], stride)
        valid_counts[i] = np.count_nonzero(valid_blocks, axis=(0, 2, 4))
        for tag in (BONE, SOFT, SAT, SKIN):
            counts[tag - 1, i] = np.count_nonzero(blocks == tag, axis=(0, 2, 4))
    dominant = np.argmax(counts, axis=0).astype(np.uint8) + 1
    body_counts = np.sum(counts, axis=0)
    dominant[body_counts == 0] = 0
    low_occupancy = (body_counts > 0) & (
        body_counts < envelope_fraction * valid_counts
    )
    thin_skin = (dominant == SKIN) & (
        counts[SKIN - 1] < skin_fraction * valid_counts
    )
    dominant[low_occupancy | thin_skin] = SOFT
    return dominant


def _block_occupancy(labels: np.ndarray, stride: int) -> np.ndarray:
    nx, ny, nz = labels.shape
    shape = tuple((value + stride - 1) // stride for value in labels.shape)
    occupied = np.zeros(shape, dtype=bool)
    pad_y = shape[1] * stride - ny
    pad_z = shape[2] * stride - nz
    for i in range(shape[0]):
        slab = labels[i * stride : min((i + 1) * stride, nx)] != 0
        pad_x = stride - slab.shape[0]
        padded = np.pad(slab, ((0, pad_x), (0, pad_y), (0, pad_z)))
        blocks = padded.reshape(stride, shape[1], stride, shape[2], stride)
        occupied[i] = np.any(blocks, axis=(0, 2, 4))
    return occupied


def _coarse_edges(axis: np.ndarray, size: int, stride: int) -> np.ndarray:
    indices = np.append(np.arange(0, size, stride, dtype=np.int64), size)
    return axis[indices]


def tetrahedralize_blocks(
    region_blocks: np.ndarray,
    axes: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    max_tetrahedra: int = 500_000,
) -> TetMesh:
    cells = np.argwhere(region_blocks != 0)
    tetrahedron_count = int(len(cells) * len(_TET_PATTERN))
    if tetrahedron_count == 0:
        raise ValueError("the downsampled label grid contains no anatomical cells")
    if tetrahedron_count > max_tetrahedra:
        raise ValueError(
            f"conversion would create {tetrahedron_count:,} tetrahedra, above the "
            f"limit {max_tetrahedra:,}; increase --stride or --max-tetrahedra"
        )

    node_shape = np.asarray(region_blocks.shape, dtype=np.int64) + 1
    corners = cells[:, None, :] + _CUBE_OFFSETS[None, :, :]
    corner_ids = np.ravel_multi_index(
        (corners[..., 0], corners[..., 1], corners[..., 2]), tuple(node_shape)
    )
    tetra_global = corner_ids[:, _TET_PATTERN].reshape(-1, 4)
    used, inverse = np.unique(tetra_global, return_inverse=True)
    index = np.asarray(np.unravel_index(used, tuple(node_shape))).T
    points = np.column_stack(
        (axes[0][index[:, 0]], axes[1][index[:, 1]], axes[2][index[:, 2]])
    )
    tags = np.repeat(region_blocks[tuple(cells.T)], len(_TET_PATTERN))
    return TetMesh(points, inverse.reshape(-1, 4), tags, TAG_NAMES)


def surface_from_blocks(
    occupied: np.ndarray,
    axes: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> SurfaceMesh:
    node_shape = np.asarray(occupied.shape, dtype=np.int64) + 1
    face_specs = (
        (0, -1, [0, 4, 7, 3]),
        (0, 1, [1, 2, 6, 5]),
        (1, -1, [0, 1, 5, 4]),
        (1, 1, [3, 7, 6, 2]),
        (2, -1, [0, 3, 2, 1]),
        (2, 1, [4, 5, 6, 7]),
    )
    quads: list[np.ndarray] = []
    for axis, direction, corner_order in face_specs:
        boundary = occupied.copy()
        current = [slice(None)] * 3
        neighbor = [slice(None)] * 3
        if direction < 0:
            current[axis] = slice(1, None)
            neighbor[axis] = slice(None, -1)
        else:
            current[axis] = slice(None, -1)
            neighbor[axis] = slice(1, None)
        boundary[tuple(current)] &= ~occupied[tuple(neighbor)]
        cells = np.argwhere(boundary)
        if len(cells) == 0:
            continue
        corners = cells[:, None, :] + _CUBE_OFFSETS[np.asarray(corner_order)][None, :, :]
        quads.append(
            np.ravel_multi_index(
                (corners[..., 0], corners[..., 1], corners[..., 2]), tuple(node_shape)
            )
        )
    if not quads:
        raise ValueError("could not extract a surface from the downsampled body")
    quad_ids = np.vstack(quads)
    triangle_global = np.vstack((quad_ids[:, [0, 1, 2]], quad_ids[:, [0, 2, 3]]))
    used, inverse = np.unique(triangle_global, return_inverse=True)
    index = np.asarray(np.unravel_index(used, tuple(node_shape))).T
    points = np.column_stack(
        (axes[0][index[:, 0]], axes[1][index[:, 1]], axes[2][index[:, 2]])
    )
    return SurfaceMesh(points, inverse.reshape(-1, 3))


def convert_voxel_mat(
    input_path: str | Path,
    output_path: str | Path,
    *,
    surface_output: str | Path | None = None,
    report_path: str | Path | None = None,
    variable: str = "MaterialLabelGrid",
    axis_keys: tuple[str, str, str] = ("Axis0", "Axis1", "Axis2"),
    axis_unit: str = "m",
    output_unit: str = "m",
    voxel_size_mm: float = 1.0,
    stride: int = 20,
    surface_stride: int = 4,
    sat_labels: Iterable[int] = DEFAULT_SAT_LABELS,
    skin_labels: Iterable[int] = DEFAULT_SKIN_LABELS,
    bone_labels: Iterable[int] = DEFAULT_BONE_LABELS,
    max_tetrahedra: int = 500_000,
    envelope_fraction: float = 0.25,
    skin_fraction: float = 0.25,
) -> dict[str, object]:
    labels, source_axes = load_voxel_mat(
        input_path,
        variable=variable,
        axis_keys=axis_keys,
        axis_unit=axis_unit,
        output_unit=output_unit,
        voxel_size_mm=voxel_size_mm,
    )
    regions = map_anatomical_regions(
        labels, sat_labels=sat_labels, skin_labels=skin_labels, bone_labels=bone_labels
    )
    region_blocks = _block_region_majority(
        regions,
        stride,
        envelope_fraction=envelope_fraction,
        skin_fraction=skin_fraction,
    )
    coarse_axes = tuple(
        _coarse_edges(axis, labels.shape[index], stride)
        for index, axis in enumerate(source_axes)
    )
    mesh = tetrahedralize_blocks(
        region_blocks, coarse_axes, max_tetrahedra=max_tetrahedra
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_npz(output, mesh)

    volumes = mesh.cell_volumes()
    report: dict[str, object] = {
        "input": str(Path(input_path)),
        "label_variable": variable,
        "source_grid_shape": list(labels.shape),
        "source_axis_unit": axis_unit,
        "output_coordinate_unit": output_unit,
        "coarse_stride_voxels": stride,
        "coarse_envelope_fraction": envelope_fraction,
        "coarse_skin_fraction": skin_fraction,
        "coarse_block_shape": list(region_blocks.shape),
        "mesh": {
            "output": str(output),
            "points": mesh.n_points,
            "tetrahedra": mesh.n_cells,
            "tag_names": mesh.tag_names,
            "tetrahedra_by_tag": {
                name: int(np.count_nonzero(mesh.cell_tags == tag))
                for name, tag in mesh.tag_names.items()
            },
            "volume_by_tag": {
                name: float(volumes[mesh.cell_tags == tag].sum())
                for name, tag in mesh.tag_names.items()
            },
        },
        "source_label_groups": {
            "SAT": [int(value) for value in sat_labels],
            "SKIN": [int(value) for value in skin_labels],
            "BONE": [int(value) for value in bone_labels],
            "SOFT": "all other non-zero labels",
        },
    }

    if surface_output is not None:
        occupied = _block_occupancy(labels, surface_stride)
        surface_axes = tuple(
            _coarse_edges(axis, labels.shape[index], surface_stride)
            for index, axis in enumerate(source_axes)
        )
        surface = surface_from_blocks(occupied, surface_axes)
        surface_path = Path(surface_output)
        surface_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            surface_path, points=surface.points, triangles=surface.triangles
        )
        report["surface"] = {
            "output": str(surface_path),
            "stride_voxels": surface_stride,
            "points": surface.n_points,
            "triangles": surface.n_triangles,
        }

    if report_path is not None:
        report_file = Path(report_path)
        report_file.parent.mkdir(parents=True, exist_ok=True)
        report_file.write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    return report

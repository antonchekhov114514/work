from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np

from .audit import tetra_quality
from .io import save_npz
from .mesh import TetMesh
from .tissue_groups import labels_to_material_ids, labels_to_mechanical_group_ids
from .voxel_convert import (
    DEFAULT_BONE_LABELS,
    DEFAULT_SAT_LABELS,
    DEFAULT_SKIN_LABELS,
    TAG_NAMES,
    load_voxel_mat,
    map_anatomical_regions,
)


def convert_voxel_mat_adaptive(
    input_path: str | Path,
    output_path: str | Path,
    *,
    report_path: str | Path | None = None,
    variable: str = "MaterialLabelGrid",
    axis_keys: tuple[str, str, str] = ("Axis0", "Axis1", "Axis2"),
    axis_unit: str = "m",
    output_unit: str = "m",
    voxel_size_mm: float = 1.0,
    coarse_stride: int = 20,
    refine_stride: int = 5,
    fine_stride: int | None = None,
    refine_labels: Iterable[int] | None = (1,),
    preserve_labels: Iterable[int] = (),
    audit_report: str | Path | Iterable[str | Path] | None = None,
    volume_error_threshold: float = 5.0,
    refine_halo_blocks: int = 1,
    sat_labels: Iterable[int] = DEFAULT_SAT_LABELS,
    skin_labels: Iterable[int] = DEFAULT_SKIN_LABELS,
    bone_labels: Iterable[int] = DEFAULT_BONE_LABELS,
    max_points: int = 250_000,
    max_tetrahedra: int = 1_500_000,
) -> dict[str, object]:
    """Create a conforming Delaunay mesh with denser samples near tissue boundaries."""
    if not 1 <= refine_stride < coarse_stride:
        raise ValueError("refine_stride must satisfy 1 <= refine_stride < coarse_stride")
    if fine_stride is not None and not 1 <= fine_stride < refine_stride:
        raise ValueError("fine_stride must satisfy 1 <= fine_stride < refine_stride")
    labels, axes = load_voxel_mat(
        input_path,
        variable=variable,
        axis_keys=axis_keys,
        axis_unit=axis_unit,
        output_unit=output_unit,
        voxel_size_mm=voxel_size_mm,
    )
    marked, occupied = _mark_refinement_blocks(
        labels,
        coarse_stride,
        None if refine_labels is None else {int(value) for value in refine_labels},
    )
    if refine_halo_blocks > 0:
        from scipy.ndimage import binary_dilation

        marked = binary_dilation(marked, iterations=refine_halo_blocks) & occupied

    critical_labels = {int(value) for value in preserve_labels}
    audit_selected: list[int] = []
    audit_paths = _as_audit_paths(audit_report)
    if audit_paths:
        selected_from_audits: set[int] = set()
        for audit_path in audit_paths:
            selected_from_audits.update(
                _labels_from_audit(audit_path, volume_error_threshold)
            )
        audit_selected = sorted(selected_from_audits)
        critical_labels.update(selected_from_audits)
    fine_marked = np.zeros_like(marked)
    if fine_stride is not None and critical_labels:
        fine_marked, _ = _mark_refinement_blocks(labels, coarse_stride, critical_labels)
        if refine_halo_blocks > 0:
            from scipy.ndimage import binary_dilation

            fine_marked = binary_dilation(fine_marked, iterations=refine_halo_blocks) & occupied
        marked |= fine_marked

    point_indices = _adaptive_point_indices(
        labels.shape,
        occupied,
        marked,
        coarse_stride,
        refine_stride,
        fine_marked=fine_marked,
        fine_stride=fine_stride,
    )
    if len(point_indices) > max_points:
        raise ValueError(
            f"adaptive sampling would create {len(point_indices):,} points; "
            "increase --coarse-stride/--refine-stride or explicitly raise --max-points"
        )
    points = np.column_stack(
        [axes[axis][point_indices[:, axis]] for axis in range(3)]
    )
    try:
        from scipy.spatial import Delaunay
    except ImportError as exc:
        raise RuntimeError("adaptive conversion requires scipy.spatial.Delaunay") from exc
    simplices = np.asarray(Delaunay(points).simplices, dtype=np.int64)
    centroids = points[simplices].mean(axis=1)
    voxel_indices = np.column_stack(
        [
            np.clip(np.searchsorted(axes[axis], centroids[:, axis], side="right") - 1, 0, labels.shape[axis] - 1)
            for axis in range(3)
        ]
    )
    source_labels = labels[tuple(voxel_indices.T)]
    simplices = simplices[source_labels != 0]
    source_labels = source_labels[source_labels != 0]
    simplices, source_labels = _drop_degenerate(points, simplices, source_labels)
    if len(simplices) > max_tetrahedra:
        raise ValueError(
            f"adaptive Delaunay mesh has {len(simplices):,} tetrahedra; "
            "coarsen sampling or explicitly raise --max-tetrahedra"
        )
    regions = map_anatomical_regions(
        source_labels,
        sat_labels=sat_labels,
        skin_labels=skin_labels,
        bone_labels=bone_labels,
    )
    final_centroids = points[simplices].mean(axis=1)
    final_indices = np.column_stack(
        [
            np.clip(np.searchsorted(axes[axis], final_centroids[:, axis], side="right") - 1, 0, labels.shape[axis] - 1)
            for axis in range(3)
        ]
    )
    mesh = TetMesh(
        points,
        simplices,
        regions,
        TAG_NAMES,
        {
            "source_label": source_labels.astype(np.int16),
            "material_id": labels_to_material_ids(source_labels),
            "mechanical_group_id": labels_to_mechanical_group_ids(source_labels),
            "adaptive_level": np.where(
                _centroids_in_marked_blocks(final_indices, marked, coarse_stride),
                1,
                0,
            ).astype(np.int8),
        },
    )
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    save_npz(target, mesh)
    report: dict[str, object] = {
        "input": str(Path(input_path)),
        "output": str(target),
        "method": "boundary-aware adaptive Delaunay sampling",
        "source_grid_shape": list(labels.shape),
        "coarse_stride_voxels": coarse_stride,
        "refine_stride_voxels": refine_stride,
        "fine_stride_voxels": fine_stride,
        "refine_labels": None if refine_labels is None else [int(value) for value in refine_labels],
        "refine_halo_blocks": refine_halo_blocks,
        "preserve_labels": sorted(critical_labels),
        "audit_selected_labels": audit_selected,
        "audit_reports": [str(path) for path in audit_paths],
        "volume_error_threshold_percent": volume_error_threshold,
        "occupied_coarse_blocks": int(np.count_nonzero(occupied)),
        "refined_coarse_blocks": int(np.count_nonzero(marked)),
        "fine_refined_coarse_blocks": int(np.count_nonzero(fine_marked)),
        "mesh": {
            "points": mesh.n_points,
            "tetrahedra": mesh.n_cells,
            "quality": tetra_quality(mesh),
        },
        "warning": (
            "Delaunay refinement is conforming and boundary-aware, but labels are sampled at "
            "tetrahedron centroids. Use volume-audit and quality-report before production runs."
        ),
    }
    if report_path is not None:
        report_target = Path(report_path)
        report_target.parent.mkdir(parents=True, exist_ok=True)
        report_target.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def _mark_refinement_blocks(
    labels: np.ndarray,
    stride: int,
    refine_labels: set[int] | None,
) -> tuple[np.ndarray, np.ndarray]:
    shape = tuple((size + stride - 1) // stride for size in labels.shape)
    marked = np.zeros(shape, dtype=bool)
    occupied = np.zeros(shape, dtype=bool)
    for i in range(shape[0]):
        xs = slice(i * stride, min((i + 1) * stride, labels.shape[0]))
        for j in range(shape[1]):
            ys = slice(j * stride, min((j + 1) * stride, labels.shape[1]))
            for k in range(shape[2]):
                zs = slice(k * stride, min((k + 1) * stride, labels.shape[2]))
                values = np.unique(labels[xs, ys, zs])
                nonzero = values[values != 0]
                occupied[i, j, k] = len(nonzero) > 0
                if not occupied[i, j, k]:
                    continue
                relevant = refine_labels is None or bool(refine_labels.intersection(nonzero.tolist()))
                marked[i, j, k] = relevant and (len(values) > 1)
    return marked, occupied


def _adaptive_point_indices(
    grid_shape: tuple[int, int, int],
    occupied: np.ndarray,
    marked: np.ndarray,
    coarse_stride: int,
    refine_stride: int,
    *,
    fine_marked: np.ndarray | None = None,
    fine_stride: int | None = None,
) -> np.ndarray:
    points: set[tuple[int, int, int]] = set()
    for block in np.argwhere(occupied):
        starts = block * coarse_stride
        ends = np.minimum(starts + coarse_stride, np.asarray(grid_shape))
        if fine_marked is not None and fine_stride is not None and fine_marked[tuple(block)]:
            stride = fine_stride
        else:
            stride = refine_stride if marked[tuple(block)] else coarse_stride
        coordinates = []
        for start, end in zip(starts, ends):
            values = list(range(int(start), int(end), stride))
            if not values or values[-1] != int(end):
                values.append(int(end))
            coordinates.append(values)
        for i in coordinates[0]:
            for j in coordinates[1]:
                for k in coordinates[2]:
                    points.add((i, j, k))
    return np.asarray(sorted(points), dtype=np.int64)


def _labels_from_audit(path: str | Path, threshold: float) -> list[int]:
    if threshold < 0.0:
        raise ValueError("volume_error_threshold must be non-negative")
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    selected = {int(value) for value in raw.get("summary", {}).get("missing_source_labels", [])}
    for row in raw.get("labels", []):
        error = row.get("mesh_vs_voxel_error_percent")
        if error is not None and abs(float(error)) > threshold:
            selected.add(int(row["label_id"]))
    return sorted(selected)


def _as_audit_paths(
    value: str | Path | Iterable[str | Path] | None,
) -> list[Path]:
    if value is None:
        return []
    if isinstance(value, (str, Path)):
        return [Path(value)]
    return [Path(path) for path in value]


def _drop_degenerate(points, tetra, labels):
    x = points[tetra]
    matrices = np.stack((x[:, 1] - x[:, 0], x[:, 2] - x[:, 0], x[:, 3] - x[:, 0]), axis=2)
    determinant = np.linalg.det(matrices)
    scale = max(float(np.ptp(points, axis=0).max()), 1.0)
    keep = np.abs(determinant) > np.finfo(float).eps * scale**3 * 100.0
    return tetra[keep], labels[keep]


def _centroids_in_marked_blocks(indices, marked, stride):
    blocks = np.minimum(indices // stride, np.asarray(marked.shape) - 1)
    return marked[tuple(blocks.T)]

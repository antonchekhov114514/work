from __future__ import annotations

import csv
import json
from pathlib import Path
import numpy as np

from .io import load_mesh, load_result_npz
from .mesh import TetMesh
from .tissue_groups import ATLAS_LABELS
from .voxel_convert import load_voxel_mat


def voxel_label_volumes(
    labels: np.ndarray,
    axes: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> dict[int, float]:
    """Accumulate exact voxel volumes without allocating a full 3-D weight grid."""
    labels = np.asarray(labels, dtype=np.int64)
    widths = tuple(np.diff(np.asarray(axis, dtype=float)) for axis in axes)
    if any(np.any(width <= 0.0) for width in widths):
        raise ValueError("voxel axes must be strictly increasing")

    maximum = int(labels.max(initial=0))
    totals = np.zeros(maximum + 1, dtype=float)
    yz_area = widths[1][:, None] * widths[2][None, :]
    for index, dx in enumerate(widths[0]):
        slab = labels[index].ravel()
        totals += np.bincount(
            slab,
            weights=(float(dx) * yz_area).ravel(),
            minlength=maximum + 1,
        )
    return {int(label): float(totals[label]) for label in np.flatnonzero(totals)}


def mesh_label_volumes(
    mesh: TetMesh,
    points: np.ndarray | None = None,
) -> dict[int, float]:
    labels = mesh.cell_data.get("source_label")
    if labels is None:
        raise ValueError("mesh has no source_label cell data")
    labels = np.asarray(labels, dtype=np.int64)
    volumes = mesh.cell_volumes(points)
    return {
        int(label): float(volumes[labels == label].sum())
        for label in np.unique(labels)
    }


def tetra_quality(mesh: TetMesh, points: np.ndarray | None = None) -> dict[str, object]:
    """Return scale-independent tetra quality; one is a regular tetrahedron."""
    coordinates = mesh.points if points is None else np.asarray(points, dtype=float)
    x = coordinates[mesh.tetra]
    volumes = mesh.cell_volumes(coordinates)
    edge_pairs = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    squared_edges = np.stack(
        [np.sum((x[:, left] - x[:, right]) ** 2, axis=1) for left, right in edge_pairs],
        axis=1,
    )
    denominator = squared_edges.sum(axis=1)
    quality = np.zeros(mesh.n_cells, dtype=float)
    positive = (volumes > 0.0) & (denominator > 0.0)
    quality[positive] = (
        12.0 * np.power(3.0 * volumes[positive], 2.0 / 3.0) / denominator[positive]
    )
    edge_lengths = np.sqrt(squared_edges)
    edge_ratio = edge_lengths.max(axis=1) / np.maximum(edge_lengths.min(axis=1), 1.0e-30)
    percentiles = [1, 5, 25, 50, 75, 95, 99]
    return {
        "tetrahedra": mesh.n_cells,
        "negative_or_zero_volume_count": int(np.count_nonzero(volumes <= 0.0)),
        "minimum_signed_volume": float(volumes.min()) if len(volumes) else 0.0,
        "mean_ratio_quality_minimum": float(quality.min()) if len(quality) else 0.0,
        "mean_ratio_quality_mean": float(quality.mean()) if len(quality) else 0.0,
        "mean_ratio_quality_percentiles": {
            str(value): float(np.percentile(quality, value)) for value in percentiles
        },
        "edge_ratio_maximum": float(edge_ratio.max()) if len(edge_ratio) else 0.0,
        "edge_ratio_percentiles": {
            str(value): float(np.percentile(edge_ratio, value)) for value in percentiles
        },
        "poor_quality_below_0_1_count": int(np.count_nonzero(quality < 0.1)),
        "poor_quality_below_0_2_count": int(np.count_nonzero(quality < 0.2)),
    }


def build_volume_audit(
    voxel_mat: str | Path,
    mesh_path: str | Path,
    *,
    result_path: str | Path | None = None,
    variable: str = "MaterialLabelGrid",
    axis_keys: tuple[str, str, str] = ("Axis0", "Axis1", "Axis2"),
    axis_unit: str = "m",
    output_unit: str = "m",
    voxel_size_mm: float = 1.0,
) -> dict[str, object]:
    labels, axes = load_voxel_mat(
        voxel_mat,
        variable=variable,
        axis_keys=axis_keys,
        axis_unit=axis_unit,
        output_unit=output_unit,
        voxel_size_mm=voxel_size_mm,
    )
    source = voxel_label_volumes(labels, axes)
    if result_path is None:
        mesh = load_mesh(mesh_path)
        deformed_points = None
    else:
        mesh, _, deformed_points = load_result_npz(result_path)
    reference = mesh_label_volumes(mesh)
    current = mesh_label_volumes(mesh, deformed_points) if deformed_points is not None else {}

    rows: list[dict[str, object]] = []
    for label in sorted(set(source) | set(reference) | set(current)):
        if label == 0:
            continue
        voxel_volume = source.get(label, 0.0)
        mesh_volume = reference.get(label, 0.0)
        current_volume = current.get(label)
        atlas = ATLAS_LABELS.get(label)
        rows.append(
            {
                "label_id": label,
                "tissue": atlas.atlas_name if atlas is not None else f"label_{label}",
                "voxel_volume": voxel_volume,
                "mesh_reference_volume": mesh_volume,
                "mesh_vs_voxel_error_percent": (
                    100.0 * (mesh_volume - voxel_volume) / voxel_volume
                    if voxel_volume > 0.0
                    else None
                ),
                "mesh_current_volume": current_volume,
                "current_vs_reference_ratio": (
                    current_volume / mesh_volume
                    if current_volume is not None and mesh_volume > 0.0
                    else None
                ),
            }
        )

    errors = [
        abs(float(row["mesh_vs_voxel_error_percent"]))
        for row in rows
        if row["mesh_vs_voxel_error_percent"] is not None
    ]
    report: dict[str, object] = {
        "voxel_mat": str(Path(voxel_mat)),
        "mesh": str(Path(mesh_path)),
        "result": None if result_path is None else str(Path(result_path)),
        "coordinate_unit": output_unit,
        "labels": rows,
        "summary": {
            "voxel_tissue_volume": float(sum(v for k, v in source.items() if k != 0)),
            "mesh_reference_volume": float(sum(reference.values())),
            "mesh_current_volume": float(sum(current.values())) if current else None,
            "mean_absolute_label_volume_error_percent": float(np.mean(errors)) if errors else 0.0,
            "maximum_absolute_label_volume_error_percent": float(max(errors, default=0.0)),
            "missing_source_labels": [
                int(label) for label in source if label != 0 and reference.get(label, 0.0) == 0.0
            ],
        },
        "reference_mesh_quality": tetra_quality(mesh),
    }
    if deformed_points is not None:
        report["deformed_mesh_quality"] = tetra_quality(mesh, deformed_points)
    return report


def write_volume_audit(
    report: dict[str, object],
    *,
    json_path: str | Path | None = None,
    csv_path: str | Path | None = None,
) -> None:
    if json_path is not None:
        target = Path(json_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if csv_path is not None:
        target = Path(csv_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        rows = list(report["labels"])
        fields = list(rows[0]) if rows else ["label_id", "tissue"]
        with target.open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

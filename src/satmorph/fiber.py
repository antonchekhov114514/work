from __future__ import annotations

from pathlib import Path

import numpy as np

from .io import load_mesh, save_npz
from .mesh import TetMesh
from .tissue_groups import ATLAS_LABELS


def build_anatomical_fiber_field(
    mesh: TetMesh,
    *,
    longitudinal_axis: int = 2,
) -> np.ndarray:
    labels = mesh.cell_data.get("source_label")
    if labels is None:
        raise ValueError("fiber-field generation requires source_label cell data")
    if longitudinal_axis not in (0, 1, 2):
        raise ValueError("longitudinal_axis must be 0, 1, or 2")
    labels = np.asarray(labels, dtype=np.int64)
    centroids = mesh.points[mesh.tetra].mean(axis=1)
    directions = np.zeros((mesh.n_cells, 3), dtype=float)
    body_center = mesh.points.mean(axis=0)
    longitudinal = np.zeros(3)
    longitudinal[longitudinal_axis] = 1.0

    for label in np.unique(labels):
        entry = ATLAS_LABELS.get(int(label))
        if entry is None:
            continue
        selected = labels == label
        if entry.mechanical_group == "SKIN":
            radial = centroids[selected] - body_center
            radial[:, longitudinal_axis] = 0.0
            tangent = np.cross(longitudinal[None, :], radial)
            directions[selected] = _normalize_rows(tangent)
        elif entry.mechanical_group in {"MUSCLE", "TENDON_LIGAMENT", "CARTILAGE_DISC"}:
            points = centroids[selected]
            if len(points) >= 3:
                covariance = np.cov((points - points.mean(axis=0)).T)
                values, vectors = np.linalg.eigh(covariance)
                axis = vectors[:, int(np.argmax(values))]
            else:
                axis = longitudinal
            directions[selected] = axis
    return _normalize_rows(directions)


def save_mesh_with_fibers(
    input_path: str | Path,
    output_path: str | Path,
    *,
    longitudinal_axis: int = 2,
) -> dict[str, object]:
    mesh = load_mesh(input_path)
    fibers = build_anatomical_fiber_field(mesh, longitudinal_axis=longitudinal_axis)
    cell_data = {name: values.copy() for name, values in mesh.cell_data.items()}
    cell_data["fiber_direction"] = fibers
    enriched = TetMesh(
        mesh.points.copy(), mesh.tetra.copy(), mesh.cell_tags.copy(), mesh.tag_names.copy(), cell_data
    )
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    save_npz(target, enriched)
    return {
        "output": str(target),
        "cells_with_fibers": int(np.count_nonzero(np.linalg.norm(fibers, axis=1) > 0.0)),
        "total_cells": mesh.n_cells,
        "longitudinal_axis": longitudinal_axis,
    }


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    lengths = np.linalg.norm(values, axis=1)
    out = np.zeros_like(values)
    valid = lengths > 0.0
    out[valid] = values[valid] / lengths[valid, None]
    return out

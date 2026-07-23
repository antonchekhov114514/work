from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .material_library import load_physical_materials
from .mesh import TetMesh
from .tissue_groups import ATLAS_LABELS


PHYSICAL_FIELD_NAMES = (
    "mass_density_kg_per_m3",
    "conductivity_s_per_m",
    "relative_permittivity",
    "em_frequency_hz",
)


def attach_physical_properties(
    mesh: TetMesh,
    materials_path: str | Path,
) -> tuple[TetMesh, dict[str, object]]:
    labels = mesh.cell_data.get("source_label")
    if labels is None:
        raise ValueError("source_label cell data is required for physical-property mapping")
    labels = np.asarray(labels, dtype=np.int64)
    records = load_physical_materials(materials_path)

    density = np.full(mesh.n_cells, np.nan, dtype=float)
    conductivity = np.full(mesh.n_cells, np.nan, dtype=float)
    permittivity = np.full(mesh.n_cells, np.nan, dtype=float)
    frequency = np.full(mesh.n_cells, np.nan, dtype=float)
    missing: list[int] = []
    mapped: list[dict[str, object]] = []
    for label in np.unique(labels):
        entry = ATLAS_LABELS.get(int(label))
        record = None if entry is None else records.get(entry.material_id)
        if record is None:
            missing.append(int(label))
            continue
        properties = dict(record.get("properties", {}))
        electric = dict(properties.get("electric", {}))
        mask = labels == label
        density[mask] = float(properties["mass_density_kg_per_m3"])
        conductivity[mask] = float(electric["conductivity_s_per_m"])
        permittivity[mask] = float(electric["relative_permittivity"])
        frequency[mask] = float(electric["frequency_hz"])
        mapped.append(
            {
                "label_id": int(label),
                "tissue": entry.atlas_name,
                "material_id": entry.material_id,
                "database_catalog": record.get("database_catalog"),
                "database_name": record.get("database_name"),
                "mass_density_kg_per_m3": float(properties["mass_density_kg_per_m3"]),
                "conductivity_s_per_m": float(electric["conductivity_s_per_m"]),
                "relative_permittivity": float(electric["relative_permittivity"]),
                "frequency_hz": float(electric["frequency_hz"]),
            }
        )
    if missing:
        raise ValueError(f"no physical material mapping for source labels {missing}")

    cell_data = {name: np.asarray(values).copy() for name, values in mesh.cell_data.items()}
    cell_data.update(
        {
            "mass_density_kg_per_m3": density,
            "conductivity_s_per_m": conductivity,
            "relative_permittivity": permittivity,
            "em_frequency_hz": frequency,
        }
    )
    output = TetMesh(
        mesh.points.copy(),
        mesh.tetra.copy(),
        mesh.cell_tags.copy(),
        mesh.tag_names.copy(),
        cell_data,
    )
    frequencies = sorted(float(value) for value in np.unique(frequency))
    return output, {
        "schema": "satmorph-physical-property-map-v1",
        "materials": str(Path(materials_path)),
        "source_labels": len(mapped),
        "mapped_cells": mesh.n_cells,
        "frequency_hz": frequencies[0] if len(frequencies) == 1 else frequencies,
        "fields": list(PHYSICAL_FIELD_NAMES),
        "labels": mapped,
        "label_policy": "source_label is immutable; properties are reattached by label after remeshing",
    }


def mass_report(
    mesh: TetMesh,
    *,
    current_points: np.ndarray | None = None,
    growth_j: np.ndarray | None = None,
    material_reference_volume: np.ndarray | None = None,
    length_unit: str = "m",
) -> dict[str, object]:
    density = mesh.cell_data.get("mass_density_kg_per_m3")
    labels = mesh.cell_data.get("source_label")
    if density is None or labels is None:
        raise ValueError("mass report requires source_label and mass_density_kg_per_m3 cell data")
    scale = {"m": 1.0, "mm": 1.0e-3}.get(length_unit)
    if scale is None:
        raise ValueError("length_unit must be 'm' or 'mm'")
    density = np.asarray(density, dtype=float)
    labels = np.asarray(labels, dtype=np.int64)
    geometric_reference_volume = mesh.cell_volumes() * scale**3
    reference_volume = (
        geometric_reference_volume
        if material_reference_volume is None
        else np.asarray(material_reference_volume, dtype=float) * scale**3
    )
    if reference_volume.shape != (mesh.n_cells,):
        raise ValueError("material_reference_volume must contain one value per tetrahedron")
    current_volume = (
        geometric_reference_volume
        if current_points is None
        else mesh.cell_volumes(np.asarray(current_points, dtype=float)) * scale**3
    )
    growth = np.ones(mesh.n_cells, dtype=float) if growth_j is None else np.asarray(growth_j, dtype=float)
    if growth.shape != (mesh.n_cells,):
        raise ValueError("growth_j must contain one value per tetrahedron")

    initial_mass = density * reference_volume
    geometric_mass = density * current_volume
    growth_mass = density * reference_volume * growth
    rows: list[dict[str, object]] = []
    for label in np.unique(labels):
        mask = labels == label
        entry = ATLAS_LABELS.get(int(label))
        rows.append(
            {
                "label_id": int(label),
                "tissue": entry.atlas_name if entry is not None else f"label_{label}",
                "density_kg_per_m3": float(np.mean(density[mask])),
                "reference_volume_m3": float(reference_volume[mask].sum()),
                "current_geometric_volume_m3": float(current_volume[mask].sum()),
                "growth_volume_m3": float((reference_volume[mask] * growth[mask]).sum()),
                "initial_mass_kg": float(initial_mass[mask].sum()),
                "geometric_constant_density_mass_kg": float(geometric_mass[mask].sum()),
                "growth_accounted_mass_kg": float(growth_mass[mask].sum()),
                "growth_mass_change_kg": float((growth_mass[mask] - initial_mass[mask]).sum()),
            }
        )
    initial_total = float(initial_mass.sum())
    geometric_total = float(geometric_mass.sum())
    growth_total = float(growth_mass.sum())
    return {
        "schema": "satmorph-mass-report-v1",
        "length_unit": length_unit,
        "summary": {
            "initial_mass_kg": initial_total,
            "geometric_constant_density_mass_kg": geometric_total,
            "growth_accounted_mass_kg": growth_total,
            "growth_mass_change_kg": growth_total - initial_total,
            "geometric_mass_change_kg": geometric_total - initial_total,
        },
        "interpretation": {
            "growth_accounted_mass": (
                "rho times reference volume times J_growth; elastic compression J_elastic "
                "does not create or remove biological mass"
            ),
            "geometric_constant_density_mass": (
                "rho times deformed geometric volume; useful for geometry checks but it "
                "mixes elastic volume change with biological growth"
            ),
        },
        "labels": rows,
    }


def write_json_report(report: dict[str, object], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

from __future__ import annotations

import json
import re
from html import escape
from pathlib import Path

import numpy as np

from .io import load_result_npz
from .surface_map import load_surface, map_surface, save_surface_result
from .surface_ops import save_surface_mesh, smooth_surface
from .tissue_groups import (
    ATLAS_LABELS,
    MECHANICAL_GROUPS,
    MESH_DOMAINS,
    mesh_domain_for_label,
)
from .visual_surface import _marching_cubes_surface
from .voxel_convert import _block_occupancy, _coarse_edges, load_voxel_mat, surface_from_blocks


GROUP_COLORS = {
    "SAT_FAT": "#F5C84C",
    "VISCERAL_FAT": "#E89B32",
    "SKIN": "#E8B09A",
    "MUSCLE": "#B43A3A",
    "ORGAN_SOFT": "#D96868",
    "LUNG_AIRWAY": "#78A9CF",
    "FLUID_BLOOD": "#B51F3A",
    "AIR_LUMEN": "#B9DCE8",
    "TENDON_LIGAMENT": "#EFE7D0",
    "CNS_NERVE": "#D8C7E8",
    "EYE": "#69B7D5",
    "CARTILAGE_DISC": "#65BFA0",
    "BONE_CANCELLOUS": "#E3D7B9",
    "BONE_CORTICAL": "#F4F0E6",
    "TOOTH": "#FFFFFF",
    "MARROW_YELLOW": "#E7B83E",
}


def extract_tissue_surface_bundle(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    include_labels: list[int] | None = None,
    variable: str = "MaterialLabelGrid",
    axis_keys: tuple[str, str, str] = ("Axis0", "Axis1", "Axis2"),
    axis_unit: str = "m",
    output_unit: str = "m",
    voxel_size_mm: float = 1.0,
    surface_stride: int = 2,
    pre_smooth_sigma: float = 0.0,
    smooth_iterations: int = 15,
    suffix: str = ".vtp",
    method: str = "marching-cubes",
) -> dict[str, object]:
    labels, axes = load_voxel_mat(
        input_path,
        variable=variable,
        axis_keys=axis_keys,
        axis_unit=axis_unit,
        output_unit=output_unit,
        voxel_size_mm=voxel_size_mm,
    )
    selected = sorted(
        int(value) for value in (
            np.unique(labels[labels != 0]) if include_labels is None else include_labels
        )
        if np.any(labels == int(value))
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    surface_axes = tuple(
        _coarse_edges(axis, labels.shape[index], surface_stride)
        for index, axis in enumerate(axes)
    )
    entries: list[dict[str, object]] = []
    for label in selected:
        occupied = _block_occupancy((labels == label).astype(np.uint8), surface_stride)
        if not np.any(occupied):
            continue
        if method == "marching-cubes":
            surface = _marching_cubes_surface(
                occupied, surface_axes, pre_smooth_sigma=pre_smooth_sigma
            )
        elif method == "blocks":
            surface = surface_from_blocks(occupied, surface_axes)
        else:
            raise ValueError("method must be 'marching-cubes' or 'blocks'")
        surface = smooth_surface(
            surface,
            method="taubin",
            iterations=smooth_iterations,
            taubin_lambda=0.5,
            taubin_mu=-0.53,
        )
        entry = ATLAS_LABELS.get(label)
        tissue_name = entry.atlas_name if entry is not None else f"Label {label}"
        group = entry.mechanical_group if entry is not None else "ORGAN_SOFT"
        mesh_domain = mesh_domain_for_label(label)
        filename = f"{label:03d}-{_slug(tissue_name)}{suffix}"
        target = output / filename
        save_surface_mesh(
            target,
            surface,
            normals=True,
            point_data={"source_label": np.full(surface.n_points, label, dtype=np.int16)},
            cell_data={
                "source_label": np.full(surface.n_triangles, label, dtype=np.int16),
                "mechanical_group_id": np.full(
                    surface.n_triangles, MECHANICAL_GROUPS.get(group, 0), dtype=np.int16
                ),
                "mesh_domain_tag": np.full(
                    surface.n_triangles, MESH_DOMAINS.get(mesh_domain, 0), dtype=np.int16
                ),
            },
        )
        entries.append(
            {
                "label_id": label,
                "tissue": tissue_name,
                "mechanical_group": group,
                "mesh_domain": mesh_domain,
                "mesh_domain_tag": MESH_DOMAINS.get(mesh_domain, 0),
                "color": GROUP_COLORS.get(group, "#B0B0B0"),
                "opacity": 0.35 if group in {"SKIN", "SAT_FAT"} else 1.0,
                "file": filename,
                "points": surface.n_points,
                "triangles": surface.n_triangles,
            }
        )
    manifest = {
        "format": "satmorph-tissue-surface-bundle-v1",
        "source": str(Path(input_path)),
        "coordinate_unit": output_unit,
        "surface_stride_voxels": surface_stride,
        "method": method,
        "surfaces": entries,
    }
    manifest_path = output / "tissues.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    if suffix.lower() == ".vtp":
        _write_vtm(output / "tissues.vtm", entries)
    return manifest


def map_tissue_surface_bundle(
    coarse_result: str | Path,
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    candidate_cells: int = 64,
    outside_mode: str = "clamp",
) -> dict[str, object]:
    mesh, displacement, _ = load_result_npz(coarse_result)
    manifest_file = Path(manifest_path)
    source_root = manifest_file.parent
    raw = json.loads(manifest_file.read_text(encoding="utf-8"))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    mapped_entries: list[dict[str, object]] = []
    for entry in raw.get("surfaces", []):
        source = source_root / entry["file"]
        surface = load_surface(source)
        result = map_surface(
            mesh,
            displacement,
            surface,
            candidate_cells=candidate_cells,
            outside_mode=outside_mode,
            map_centers=False,
        )
        target = output / entry["file"]
        save_surface_result(target, surface, result)
        mapped = dict(entry)
        mapped["file"] = target.name
        mapped["outside_points"] = int(np.count_nonzero(~result.inside))
        mapped["maximum_displacement"] = float(
            np.linalg.norm(result.displacement, axis=1).max(initial=0.0)
        )
        mapped_entries.append(mapped)
    mapped_manifest = {
        "format": "satmorph-mapped-tissue-surface-bundle-v1",
        "coarse_result": str(Path(coarse_result)),
        "source_manifest": str(manifest_file),
        "surfaces": mapped_entries,
    }
    target_manifest = output / "tissues-deformed.json"
    target_manifest.write_text(
        json.dumps(mapped_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if mapped_entries and Path(mapped_entries[0]["file"]).suffix.lower() == ".vtp":
        _write_vtm(output / "tissues-deformed.vtm", mapped_entries)
    return mapped_manifest


def _write_vtm(path: Path, entries: list[dict[str, object]]) -> None:
    lines = [
        '<?xml version="1.0"?>',
        '<VTKFile type="vtkMultiBlockDataSet" version="1.0" byte_order="LittleEndian">',
        "  <vtkMultiBlockDataSet>",
    ]
    for index, entry in enumerate(entries):
        lines.append(
            f'    <DataSet index="{index}" name="{escape(str(entry["tissue"]))}" '
            f'file="{escape(str(entry["file"]))}"/>'
        )
    lines.extend(["  </vtkMultiBlockDataSet>", "</VTKFile>"])
    path.write_text("\n".join(lines), encoding="utf-8")


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", value).strip("-").lower() or "tissue"

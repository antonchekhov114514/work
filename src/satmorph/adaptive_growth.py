from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np

from .io import save_mesh_vtu, save_npz, save_result_bundle
from .material_library import mechanical_parameters_for_label
from .mesh import TetMesh
from .physical_properties import mass_report
from .remesh import coarsen_marked_tetrahedra, mark_source_labels, refine_marked_tetrahedra
from .solver import Material, SolverOptions, morph_target_region


def literature_cell_materials(mesh: TetMesh) -> np.ndarray:
    labels = mesh.cell_data.get("source_label")
    if labels is None:
        raise ValueError("source_label is required for the 73-label mechanical library")
    labels = np.asarray(labels, dtype=np.int64)
    materials = np.empty(mesh.n_cells, dtype=object)
    for label in np.unique(labels):
        parameters = mechanical_parameters_for_label(int(label))
        materials[labels == label] = Material(**parameters)
    return materials


def remesh_morph_result(
    mesh: TetMesh,
    deformed_points: np.ndarray,
    growth_lambda: np.ndarray,
    j_total: np.ndarray,
    j_elastic: np.ndarray,
    *,
    target_labels: Iterable[int] = (1,),
    max_edges: int = 1_000,
    interface_mode: str = "propagate",
) -> tuple[TetMesh, dict[str, object]]:
    target_labels = [int(value) for value in target_labels]
    current = mesh.copy_with_points(np.asarray(deformed_points, dtype=float))
    marked = mark_source_labels(current, target_labels)
    previous_growth = np.asarray(
        current.cell_data.get("accumulated_growth_J", np.ones(current.n_cells)),
        dtype=float,
    )
    previous_elastic = np.asarray(
        current.cell_data.get(
            "elastic_history_F",
            np.repeat(np.eye(3)[None, :, :], current.n_cells, axis=0),
        ),
        dtype=float,
    )
    if previous_elastic.shape != (current.n_cells, 3, 3):
        raise ValueError("elastic_history_F must have shape (n_cells, 3, 3)")
    material_reference_volume = np.asarray(
        current.cell_data.get("material_reference_volume", mesh.cell_volumes()),
        dtype=float,
    )
    growth_lambda = np.asarray(growth_lambda, dtype=float)
    j_total = np.asarray(j_total, dtype=float)
    j_elastic = np.asarray(j_elastic, dtype=float)
    _, gradients, _ = mesh.reference_geometry()
    incremental_deformation = np.einsum(
        "eai,eaj->eij", np.asarray(deformed_points)[mesh.tetra], gradients, optimize=True
    )
    updated_elastic = np.einsum(
        "eij,ejk->eik", incremental_deformation, previous_elastic, optimize=True
    ) / growth_lambda[:, None, None]
    history_j = np.linalg.det(updated_elastic)
    if np.any(history_j <= 0.0):
        raise ValueError("updated elastic history contains an inverted state")
    history_j_error = np.abs(history_j - j_elastic) / np.maximum(np.abs(j_elastic), 1.0e-12)
    indicator = np.maximum(j_total, 1.0 / np.maximum(j_total, 1.0e-12))
    refined = refine_marked_tetrahedra(
        current,
        marked,
        max_edges=max_edges,
        indicator=indicator,
        interface_mode=interface_mode,
        parent_cell_data={
            "accumulated_growth_J": previous_growth * growth_lambda**3,
            "previous_J_total": j_total,
            "previous_J_elastic": j_elastic,
            "elastic_history_F": updated_elastic,
        },
        extensive_parent_cell_data={
            "material_reference_volume": material_reference_volume,
        },
    )
    report = dict(refined.report)
    report.update(
        {
            "target_labels": target_labels,
            "history_transfer": {
                "accumulated_growth_J": "parent value times current growth_lambda cubed",
                "source_label": "inherited exactly from parent",
                "material_fields": "inherited exactly from parent",
                "material_reference_volume": "split conservatively among child cells",
                "elastic_history_F": (
                    "cumulative elastic deformation inherited exactly by every child; "
                    "the next solve evaluates stress from F_incremental @ elastic_history_F"
                ),
                "contact_state": "not transferred; contact constraints must be rebuilt after remeshing",
            },
            "elastic_history_J_min": float(history_j.min()),
            "elastic_history_J_max": float(history_j.max()),
            "elastic_history_J_consistency_max_relative": float(history_j_error.max()),
        }
    )
    return refined.mesh, report


def coarsen_morph_result(
    mesh: TetMesh,
    deformed_points: np.ndarray,
    growth_lambda: np.ndarray,
    j_total: np.ndarray,
    j_elastic: np.ndarray,
    *,
    target_labels: Iterable[int] = (1,),
    max_collapses: int = 500,
    max_local_volume_drift: float = 0.01,
) -> tuple[TetMesh, dict[str, object]]:
    target_labels = [int(value) for value in target_labels]
    current = mesh.copy_with_points(np.asarray(deformed_points, dtype=float))
    marked = mark_source_labels(current, target_labels)
    previous_growth = np.asarray(
        current.cell_data.get("accumulated_growth_J", np.ones(current.n_cells)), dtype=float
    )
    previous_elastic = np.asarray(
        current.cell_data.get(
            "elastic_history_F",
            np.repeat(np.eye(3)[None, :, :], current.n_cells, axis=0),
        ),
        dtype=float,
    )
    material_reference_volume = np.asarray(
        current.cell_data.get("material_reference_volume", mesh.cell_volumes()), dtype=float
    )
    growth_lambda = np.asarray(growth_lambda, dtype=float)
    j_total = np.asarray(j_total, dtype=float)
    j_elastic = np.asarray(j_elastic, dtype=float)
    _, gradients, _ = mesh.reference_geometry()
    incremental_deformation = np.einsum(
        "eai,eaj->eij", np.asarray(deformed_points)[mesh.tetra], gradients, optimize=True
    )
    updated_elastic = np.einsum(
        "eij,ejk->eik", incremental_deformation, previous_elastic, optimize=True
    ) / growth_lambda[:, None, None]
    history_j = np.linalg.det(updated_elastic)
    if np.any(history_j <= 0.0):
        raise ValueError("updated elastic history contains an inverted state")
    history_j_error = np.abs(history_j - j_elastic) / np.maximum(np.abs(j_elastic), 1.0e-12)
    indicator = np.maximum(j_total, 1.0 / np.maximum(j_total, 1.0e-12))
    coarsened = coarsen_marked_tetrahedra(
        current,
        marked,
        max_collapses=max_collapses,
        indicator=indicator,
        max_local_volume_drift=max_local_volume_drift,
        parent_cell_data={
            "accumulated_growth_J": previous_growth * growth_lambda**3,
            "previous_J_total": j_total,
            "previous_J_elastic": j_elastic,
            "elastic_history_F": updated_elastic,
        },
        extensive_parent_cell_data={
            "material_reference_volume": material_reference_volume,
        },
        project_parent_cell_data=(
            "accumulated_growth_J",
            "previous_J_total",
            "previous_J_elastic",
            "elastic_history_F",
        ),
    )
    report = dict(coarsened.report)
    report.update(
        {
            "target_labels": target_labels,
            "elastic_history_J_min_before_projection": float(history_j.min()),
            "elastic_history_J_max_before_projection": float(history_j.max()),
            "elastic_history_J_consistency_max_relative": float(history_j_error.max()),
            "history_transfer_detail": {
                "refinement": "not used in this operation",
                "coarsening": (
                    "same-label local material-volume-weighted projection onto surviving cells"
                ),
                "contact_state": "not transferred; contact constraints must be rebuilt",
            },
        }
    )
    return coarsened.mesh, report


def solve_incremental_with_remeshing(
    mesh: TetMesh,
    *,
    target_labels: Iterable[int],
    bone_tags: Iterable[int],
    target_growth_volume_ratio: float,
    stages: int,
    max_edges_per_stage: int,
    interface_mode: str,
    output_dir: str | Path,
    options: SolverOptions,
    max_collapses_per_stage: int = 500,
    remesh_mode: str = "auto",
    max_local_volume_drift: float = 0.01,
) -> dict[str, object]:
    if target_growth_volume_ratio <= 0.0:
        raise ValueError("target_growth_volume_ratio must be positive")
    if stages < 1:
        raise ValueError("stages must be at least one")
    if remesh_mode not in {"auto", "refine", "coarsen", "none"}:
        raise ValueError("remesh_mode must be auto, refine, coarsen, or none")
    labels = [int(value) for value in target_labels]
    if not labels:
        raise ValueError("at least one target label is required")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    original = mesh
    original_target = mark_source_labels(original, labels)
    original_volume = float(original.cell_volumes()[original_target].sum())
    current = mesh
    incremental_ratio = float(target_growth_volume_ratio ** (1.0 / stages))
    stage_reports: list[dict[str, object]] = []

    for stage in range(1, stages + 1):
        target = mark_source_labels(current, labels)
        fixed_nodes = current.nodes_for_tags(list(bone_tags))
        result = morph_target_region(
            current,
            target,
            fixed_nodes,
            incremental_ratio,
            cell_materials=literature_cell_materials(current),
            default_material=Material(young=12_000.0, poisson=0.49),
            options=options,
            target_name="source_label:" + ",".join(str(value) for value in labels),
        )
        stage_base = output / f"stage-{stage:02d}-solve"
        solve_paths = save_result_bundle(stage_base, current, result)
        stage_mode = remesh_mode
        if stage_mode == "auto":
            stage_mode = "refine" if incremental_ratio > 1.0 else (
                "coarsen" if incremental_ratio < 1.0 else "none"
            )
        if stage_mode == "coarsen":
            remeshed, remesh_report = coarsen_morph_result(
                current,
                result.points,
                result.growth_lambda,
                result.j_total,
                result.j_elastic,
                target_labels=labels,
                max_collapses=max_collapses_per_stage,
                max_local_volume_drift=max_local_volume_drift,
            )
        elif stage_mode == "refine":
            remeshed, remesh_report = remesh_morph_result(
                current,
                result.points,
                result.growth_lambda,
                result.j_total,
                result.j_elastic,
                target_labels=labels,
                max_edges=max_edges_per_stage,
                interface_mode=interface_mode,
            )
        else:
            remeshed, remesh_report = remesh_morph_result(
                current,
                result.points,
                result.growth_lambda,
                result.j_total,
                result.j_elastic,
                target_labels=labels,
                max_edges=0,
                interface_mode=interface_mode,
            )
        mesh_npz = output / f"stage-{stage:02d}-remeshed.npz"
        mesh_vtu = output / f"stage-{stage:02d}-remeshed.vtu"
        remesh_json = output / f"stage-{stage:02d}-remeshed.json"
        save_npz(mesh_npz, remeshed)
        save_mesh_vtu(mesh_vtu, remeshed)
        remesh_json.write_text(json.dumps(remesh_report, indent=2, ensure_ascii=False), encoding="utf-8")
        current = remeshed
        current_target = mark_source_labels(current, labels)
        current_volume = float(current.cell_volumes()[current_target].sum())
        stage_reports.append(
            {
                "stage": stage,
                "incremental_growth_volume_ratio": incremental_ratio,
                "remesh_mode": stage_mode,
                "actual_stage_volume_ratio": result.actual_volume_ratio,
                "cumulative_geometric_volume_ratio": current_volume / original_volume,
                "target_cells_after_remesh": int(np.count_nonzero(current_target)),
                "solve_outputs": [str(path) for path in solve_paths],
                "remeshed_mesh": str(mesh_npz),
                "remesh_report": str(remesh_json),
                "selected_operations": remesh_report.get(
                    "selected_edges", remesh_report.get("selected_collapses", 0)
                ),
            }
        )

    final_target = mark_source_labels(current, labels)
    final_volume = float(current.cell_volumes()[final_target].sum())
    summary: dict[str, object] = {
        "schema": "satmorph-incremental-remesh-v1",
        "target_labels": labels,
        "target_growth_volume_ratio": target_growth_volume_ratio,
        "stages": stages,
        "incremental_growth_volume_ratio": incremental_ratio,
        "remesh_mode": remesh_mode,
        "tetrahedra_before": original.n_cells,
        "tetrahedra_after": current.n_cells,
        "target_cells_before": int(np.count_nonzero(original_target)),
        "target_cells_after": int(np.count_nonzero(final_target)),
        "actual_cumulative_geometric_volume_ratio": final_volume / original_volume,
        "source_labels_preserved": sorted(
            int(value) for value in np.unique(current.cell_data["source_label"])
        )
        == sorted(int(value) for value in np.unique(original.cell_data["source_label"])),
        "stages_detail": stage_reports,
        "limitations": [
            "Constitutive elastic history is transferred, but contact multipliers and frictional history are not.",
            "Coarsening is deliberately conservative and may reject all collapses in very thin or poorly resolved SAT layers.",
            "Coarsening projects integration-point history locally and therefore introduces a measurable remapping approximation.",
            "The prescribed cumulative growth ratio is not an outer-calibrated final geometric SAT ratio.",
        ],
    }
    if "mass_density_kg_per_m3" in current.cell_data:
        summary["final_mass"] = mass_report(
            current,
            growth_j=np.asarray(current.cell_data["accumulated_growth_J"], dtype=float),
            material_reference_volume=np.asarray(
                current.cell_data["material_reference_volume"], dtype=float
            ),
        )["summary"]
    final_npz = output / "final-remeshed.npz"
    final_vtu = output / "final-remeshed.vtu"
    save_npz(final_npz, current)
    save_mesh_vtu(final_vtu, current)
    summary["final_mesh_npz"] = str(final_npz)
    summary["final_mesh_vtu"] = str(final_vtu)
    summary_path = output / "incremental-remesh-summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary

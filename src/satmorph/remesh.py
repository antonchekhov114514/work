from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from typing import Iterable

import numpy as np

from .audit import tetra_quality
from .mesh import TetMesh


EDGE_PAIRS = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


@dataclass(frozen=True)
class RefinementResult:
    mesh: TetMesh
    parent_cells: np.ndarray
    report: dict[str, object]


@dataclass(frozen=True)
class CoarseningResult:
    mesh: TetMesh
    parent_cells: np.ndarray
    report: dict[str, object]


def _edge(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


def _label_statistics(mesh: TetMesh, labels: np.ndarray) -> dict[int, dict[str, float | int]]:
    volumes = mesh.cell_volumes()
    return {
        int(label): {
            "cells": int(np.count_nonzero(labels == label)),
            "volume": float(volumes[labels == label].sum()),
        }
        for label in np.unique(labels)
    }


def _candidate_edges(
    mesh: TetMesh,
    marked: np.ndarray,
    indicator: np.ndarray,
) -> dict[tuple[int, int], float]:
    candidates: dict[tuple[int, int], float] = {}
    coordinates = mesh.points[mesh.tetra]
    for cell in np.flatnonzero(marked):
        lengths = np.asarray(
            [
                np.linalg.norm(coordinates[cell, left] - coordinates[cell, right])
                for left, right in EDGE_PAIRS
            ]
        )
        left, right = EDGE_PAIRS[int(np.argmax(lengths))]
        key = _edge(int(mesh.tetra[cell, left]), int(mesh.tetra[cell, right]))
        score = float(lengths.max() * max(float(indicator[cell]), 0.0))
        candidates[key] = max(candidates.get(key, -np.inf), score)
    return candidates


def _edge_stars(
    tetra: np.ndarray,
    candidates: set[tuple[int, int]],
) -> dict[tuple[int, int], list[int]]:
    stars = {edge: [] for edge in candidates}
    for cell, nodes in enumerate(tetra):
        for left, right in EDGE_PAIRS:
            key = _edge(int(nodes[left]), int(nodes[right]))
            if key in stars:
                stars[key].append(cell)
    return stars


def refine_marked_tetrahedra(
    mesh: TetMesh,
    marked_cells: np.ndarray,
    *,
    max_edges: int = 1_000,
    indicator: np.ndarray | None = None,
    interface_mode: str = "propagate",
    parent_cell_data: dict[str, np.ndarray] | None = None,
    extensive_parent_cell_data: dict[str, np.ndarray] | None = None,
) -> RefinementResult:
    """Conformingly refine marked tetrahedra by edge-star bisection.

    Each selected edge is bisected in every tetrahedron incident to that edge.
    Selected edge stars are disjoint within one call, so every parent is split at
    most once. Child cells inherit every parent label and material field.
    """
    marked = np.asarray(marked_cells, dtype=bool)
    if marked.shape != (mesh.n_cells,):
        raise ValueError("marked_cells must contain one boolean per tetrahedron")
    if max_edges < 0:
        raise ValueError("max_edges must be non-negative")
    if interface_mode not in {"propagate", "interior-only"}:
        raise ValueError("interface_mode must be 'propagate' or 'interior-only'")
    values = np.ones(mesh.n_cells, dtype=float) if indicator is None else np.asarray(indicator, dtype=float)
    if values.shape != (mesh.n_cells,):
        raise ValueError("indicator must contain one value per tetrahedron")

    candidates = _candidate_edges(mesh, marked, values)
    stars = _edge_stars(mesh.tetra, set(candidates))
    selected: list[tuple[int, int]] = []
    occupied = np.zeros(mesh.n_cells, dtype=bool)
    for edge in ([] if max_edges == 0 else sorted(candidates, key=candidates.get, reverse=True)):
        star = np.asarray(stars[edge], dtype=np.int64)
        if star.size == 0 or np.any(occupied[star]):
            continue
        if interface_mode == "interior-only" and not np.all(marked[star]):
            continue
        selected.append(edge)
        occupied[star] = True
        if len(selected) >= max_edges:
            break

    if not selected:
        unchanged_data = {
            name: np.asarray(values).copy() for name, values in mesh.cell_data.items()
        }
        if parent_cell_data:
            unchanged_data.update(
                {name: np.asarray(values).copy() for name, values in parent_cell_data.items()}
            )
        if extensive_parent_cell_data:
            unchanged_data.update(
                {
                    name: np.asarray(values).copy()
                    for name, values in extensive_parent_cell_data.items()
                }
            )
        unchanged = TetMesh(
            mesh.points.copy(),
            mesh.tetra.copy(),
            mesh.cell_tags.copy(),
            mesh.tag_names.copy(),
            unchanged_data,
        )
        labels = np.asarray(unchanged.cell_data.get("source_label", unchanged.cell_tags), dtype=np.int64)
        stats = _label_statistics(unchanged, labels)
        return RefinementResult(
            mesh=unchanged,
            parent_cells=np.arange(mesh.n_cells, dtype=np.int64),
            report={
                "method": "conforming_edge_star_bisection",
                "interface_mode": interface_mode,
                "selected_edges": 0,
                "points_before": mesh.n_points,
                "points_after": mesh.n_points,
                "tetrahedra_before": mesh.n_cells,
                "tetrahedra_after": mesh.n_cells,
                "label_statistics_before": stats,
                "label_statistics_after": stats,
                "label_set_preserved": True,
                "maximum_label_volume_drift_relative": 0.0,
                "quality_after": tetra_quality(unchanged),
            },
        )

    selected_stars = {edge: stars[edge] for edge in selected}
    midpoint_index = {
        edge: mesh.n_points + index for index, edge in enumerate(selected)
    }
    new_points = np.vstack(
        (
            mesh.points,
            np.asarray([(mesh.points[left] + mesh.points[right]) * 0.5 for left, right in selected]),
        )
    )

    split_edge_by_cell: dict[int, tuple[int, int]] = {}
    for edge, star in selected_stars.items():
        for cell in star:
            split_edge_by_cell[int(cell)] = edge

    children: list[list[int]] = []
    parents: list[int] = []
    for cell, raw_nodes in enumerate(mesh.tetra):
        edge = split_edge_by_cell.get(cell)
        if edge is None:
            children.append([int(value) for value in raw_nodes])
            parents.append(cell)
            continue
        left, right = edge
        other = [int(value) for value in raw_nodes if int(value) not in edge]
        midpoint = midpoint_index[edge]
        children.append([left, midpoint, other[0], other[1]])
        children.append([midpoint, right, other[0], other[1]])
        parents.extend((cell, cell))

    parent_indices = np.asarray(parents, dtype=np.int64)
    merged_data = {name: np.asarray(data) for name, data in mesh.cell_data.items()}
    if parent_cell_data:
        for name, data in parent_cell_data.items():
            values_for_parent = np.asarray(data)
            if values_for_parent.shape[0] != mesh.n_cells:
                raise ValueError(f"parent_cell_data[{name!r}] must have one value per parent cell")
            merged_data[str(name)] = values_for_parent
    child_data = {name: data[parent_indices] for name, data in merged_data.items()}
    if extensive_parent_cell_data:
        child_counts = np.bincount(parent_indices, minlength=mesh.n_cells)
        for name, data in extensive_parent_cell_data.items():
            parent_values = np.asarray(data)
            if parent_values.shape != (mesh.n_cells,):
                raise ValueError(
                    f"extensive_parent_cell_data[{name!r}] must be scalar per parent cell"
                )
            child_data[str(name)] = parent_values[parent_indices] / child_counts[parent_indices]
    child_data["remesh_parent_cell"] = parent_indices
    previous_generation = np.asarray(
        merged_data.get("remesh_generation", np.zeros(mesh.n_cells, dtype=np.int16)),
        dtype=np.int16,
    )
    child_data["remesh_generation"] = previous_generation[parent_indices] + 1

    refined = TetMesh(
        new_points,
        np.asarray(children, dtype=np.int64),
        mesh.cell_tags[parent_indices],
        mesh.tag_names.copy(),
        child_data,
    )

    before_labels = np.asarray(mesh.cell_data.get("source_label", mesh.cell_tags), dtype=np.int64)
    after_labels = np.asarray(refined.cell_data.get("source_label", refined.cell_tags), dtype=np.int64)
    before = _label_statistics(mesh, before_labels)
    after = _label_statistics(refined, after_labels)
    drifts = []
    for label, values_before in before.items():
        initial = float(values_before["volume"])
        final = float(after[label]["volume"])
        drifts.append(abs(final - initial) / max(abs(initial), 1.0e-30))
    report: dict[str, object] = {
        "method": "conforming_edge_star_bisection",
        "interface_mode": interface_mode,
        "selected_edges": len(selected),
        "candidate_edges": len(candidates),
        "split_parent_cells": len(split_edge_by_cell),
        "points_before": mesh.n_points,
        "points_after": refined.n_points,
        "tetrahedra_before": mesh.n_cells,
        "tetrahedra_after": refined.n_cells,
        "label_statistics_before": before,
        "label_statistics_after": after,
        "label_set_preserved": set(before) == set(after),
        "maximum_label_volume_drift_relative": float(max(drifts, default=0.0)),
        "quality_before": tetra_quality(mesh),
        "quality_after": tetra_quality(refined),
        "note": (
            "Interface-adjacent non-target cells may be bisected for conformity, "
            "but every child inherits its parent's source_label and material fields."
        ),
    }
    return RefinementResult(refined, parent_indices, report)


def mark_source_labels(mesh: TetMesh, labels: Iterable[int]) -> np.ndarray:
    source = mesh.cell_data.get("source_label")
    if source is None:
        raise ValueError("source_label cell data is required for label-preserving remeshing")
    return np.isin(np.asarray(source, dtype=np.int64), np.asarray(list(labels), dtype=np.int64))


def _vertex_cells(mesh: TetMesh) -> list[list[int]]:
    incident = [[] for _ in range(mesh.n_points)]
    for cell, nodes in enumerate(mesh.tetra):
        for node in nodes:
            incident[int(node)].append(cell)
    return incident


def _boundary_complex(mesh: TetMesh) -> tuple[set[int], set[tuple[int, int]]]:
    face_counts: dict[tuple[int, int, int], int] = {}
    for nodes in mesh.tetra:
        for face in combinations((int(value) for value in nodes), 3):
            key = tuple(sorted(face))
            face_counts[key] = face_counts.get(key, 0) + 1
    boundary_faces = [face for face, count in face_counts.items() if count == 1]
    vertices = {node for face in boundary_faces for node in face}
    edges = {
        _edge(left, right)
        for face in boundary_faces
        for left, right in combinations(face, 2)
    }
    return vertices, edges


def _link(
    mesh: TetMesh,
    vertices: tuple[int, ...],
    cell_ids: Iterable[int] | None = None,
) -> set[tuple[int, ...]]:
    required = set(vertices)
    simplices: set[tuple[int, ...]] = set()
    cells = range(mesh.n_cells) if cell_ids is None else cell_ids
    for cell in cells:
        nodes = mesh.tetra[int(cell)]
        raw = {int(value) for value in nodes}
        if not required.issubset(raw):
            continue
        rest = sorted(raw.difference(required))
        for size in range(1, len(rest) + 1):
            simplices.update(tuple(choice) for choice in combinations(rest, size))
    return simplices


def _link_condition(
    mesh: TetMesh,
    left: int,
    right: int,
    incident: list[list[int]] | None = None,
) -> bool:
    left_cells = None if incident is None else incident[left]
    right_cells = None if incident is None else incident[right]
    edge_cells = (
        None
        if incident is None
        else set(left_cells).intersection(right_cells)
    )
    return _link(mesh, (left,), left_cells).intersection(
        _link(mesh, (right,), right_cells)
    ) == _link(
        mesh, (left, right), edge_cells
    )


def _signed_six_volumes(points: np.ndarray, tetra: np.ndarray) -> np.ndarray:
    x = points[tetra]
    matrices = np.stack(
        (x[:, 1] - x[:, 0], x[:, 2] - x[:, 0], x[:, 3] - x[:, 0]), axis=2
    )
    return np.linalg.det(matrices)


def coarsen_marked_tetrahedra(
    mesh: TetMesh,
    marked_cells: np.ndarray,
    *,
    max_collapses: int = 500,
    indicator: np.ndarray | None = None,
    max_local_volume_drift: float = 0.01,
    min_survivor_volume_ratio: float = 0.05,
    parent_cell_data: dict[str, np.ndarray] | None = None,
    extensive_parent_cell_data: dict[str, np.ndarray] | None = None,
    project_parent_cell_data: Iterable[str] = (),
) -> CoarseningResult:
    """Coarsen target tissue with conservative, label-safe edge collapses.

    Endpoints must have identical incident label sets and satisfy the tetrahedral
    link condition. Boundary vertices may only collapse along a boundary edge.
    Every surviving cell inherits its parent's label; extensive quantities are
    redistributed only among surviving cells with the same source label.
    """
    marked = np.asarray(marked_cells, dtype=bool)
    if marked.shape != (mesh.n_cells,):
        raise ValueError("marked_cells must contain one boolean per tetrahedron")
    if max_collapses < 0:
        raise ValueError("max_collapses must be non-negative")
    if not (0.0 <= max_local_volume_drift < 1.0):
        raise ValueError("max_local_volume_drift must lie in [0, 1)")
    if not (0.0 < min_survivor_volume_ratio <= 1.0):
        raise ValueError("min_survivor_volume_ratio must lie in (0, 1]")
    values = np.ones(mesh.n_cells, dtype=float) if indicator is None else np.asarray(indicator, dtype=float)
    if values.shape != (mesh.n_cells,):
        raise ValueError("indicator must contain one value per tetrahedron")

    labels = np.asarray(mesh.cell_data.get("source_label", mesh.cell_tags), dtype=np.int64)
    incident = _vertex_cells(mesh)
    boundary_vertices, boundary_edges = _boundary_complex(mesh)
    label_signatures = [frozenset(int(labels[cell]) for cell in cells) for cells in incident]
    candidate_edges = {
        _edge(int(nodes[left]), int(nodes[right]))
        for cell, nodes in enumerate(mesh.tetra)
        if marked[cell]
        for left, right in EDGE_PAIRS
    }
    candidate_scores: dict[tuple[int, int], float] = {}
    for edge in candidate_edges:
        left, right = edge
        edge_cells = sorted(set(incident[left]).intersection(incident[right]))
        marked_edge_cells = [cell for cell in edge_cells if marked[cell]]
        if not marked_edge_cells:
            continue
        length = float(np.linalg.norm(mesh.points[left] - mesh.points[right]))
        candidate_scores[edge] = length / max(float(np.mean(values[marked_edge_cells])), 1.0e-12)

    selected: list[dict[str, object]] = []
    occupied_cells: set[int] = set()
    occupied_vertices: set[int] = set()
    scale = max(float(np.ptp(mesh.points, axis=0).max()), 1.0)
    determinant_floor = np.finfo(float).eps * scale**3 * 100.0
    original_six_volume = _signed_six_volumes(mesh.points, mesh.tetra)
    canonical_cells = [tuple(sorted(int(value) for value in nodes)) for nodes in mesh.tetra]
    canonical_counts = Counter(canonical_cells)

    rejection_counts: dict[str, int] = {}

    def reject(reason: str) -> None:
        rejection_counts[reason] = rejection_counts.get(reason, 0) + 1

    ordered = [] if max_collapses == 0 else sorted(candidate_scores, key=candidate_scores.get)
    for left, right in ordered:
        patch = sorted(set(incident[left]).union(incident[right]))
        if left in occupied_vertices or right in occupied_vertices or occupied_cells.intersection(patch):
            reject("overlapping_patch")
            continue
        if label_signatures[left] != label_signatures[right]:
            reject("different_label_neighborhood")
            continue
        left_boundary = left in boundary_vertices
        right_boundary = right in boundary_vertices
        if left_boundary != right_boundary:
            reject("boundary_to_interior")
            continue
        if left_boundary and _edge(left, right) not in boundary_edges:
            reject("non_boundary_chord")
            continue
        if not _link_condition(mesh, left, right, incident):
            reject("link_condition")
            continue

        edge_cells = sorted(set(incident[left]).intersection(incident[right]))
        survivors = [cell for cell in patch if cell not in edge_cells]
        if not survivors:
            reject("empty_patch")
            continue
        removed_labels = set(int(labels[cell]) for cell in edge_cells)
        survivor_labels = set(int(labels[cell]) for cell in survivors)
        if not removed_labels.issubset(survivor_labels):
            reject("would_remove_label_patch")
            continue

        midpoint = 0.5 * (mesh.points[left] + mesh.points[right])
        trial_points = mesh.points.copy()
        trial_points[left] = midpoint
        trial_tetra = mesh.tetra[survivors].copy()
        trial_tetra[trial_tetra == right] = left
        if any(len(set(int(value) for value in nodes)) < 4 for nodes in trial_tetra):
            reject("degenerate_survivor")
            continue
        trial_six = _signed_six_volumes(trial_points, trial_tetra)
        if np.any(trial_six <= determinant_floor):
            reject("inversion_or_degeneracy")
            continue
        ratios = trial_six / original_six_volume[np.asarray(survivors, dtype=np.int64)]
        if np.any(ratios < min_survivor_volume_ratio):
            reject("small_survivor")
            continue
        local_keys = [tuple(sorted(int(value) for value in nodes)) for nodes in trial_tetra]
        patch_counts = Counter(canonical_cells[cell] for cell in patch)
        collides_outside = any(
            canonical_counts[key] - patch_counts.get(key, 0) > 0 for key in local_keys
        )
        if len(local_keys) != len(set(local_keys)) or collides_outside:
            reject("duplicate_tetrahedron")
            continue
        local_drift = 0.0
        for label in label_signatures[left]:
            before = float(
                original_six_volume[
                    [cell for cell in patch if int(labels[cell]) == int(label)]
                ].sum()
            )
            after = float(
                trial_six[
                    [index for index, cell in enumerate(survivors) if int(labels[cell]) == int(label)]
                ].sum()
            )
            local_drift = max(local_drift, abs(after - before) / max(abs(before), 1.0e-30))
        if local_drift > max_local_volume_drift:
            reject("local_label_volume_drift")
            continue

        selected.append(
            {
                "left": left,
                "right": right,
                "point": midpoint,
                "patch": np.asarray(patch, dtype=np.int64),
                "removed": np.asarray(edge_cells, dtype=np.int64),
            }
        )
        occupied_vertices.update((left, right))
        occupied_cells.update(patch)
        if len(selected) >= max_collapses:
            break

    if not selected:
        unchanged_data = {name: np.asarray(data).copy() for name, data in mesh.cell_data.items()}
        if parent_cell_data:
            unchanged_data.update({name: np.asarray(data).copy() for name, data in parent_cell_data.items()})
        if extensive_parent_cell_data:
            unchanged_data.update(
                {name: np.asarray(data).copy() for name, data in extensive_parent_cell_data.items()}
            )
        unchanged = TetMesh(
            mesh.points.copy(), mesh.tetra.copy(), mesh.cell_tags.copy(), mesh.tag_names.copy(), unchanged_data
        )
        stats = _label_statistics(unchanged, labels)
        return CoarseningResult(
            unchanged,
            np.arange(mesh.n_cells, dtype=np.int64),
            {
                "method": "label_safe_edge_collapse",
                "selected_collapses": 0,
                "candidate_edges": len(candidate_scores),
                "rejection_counts": rejection_counts,
                "points_before": mesh.n_points,
                "points_after": mesh.n_points,
                "tetrahedra_before": mesh.n_cells,
                "tetrahedra_after": mesh.n_cells,
                "label_statistics_before": stats,
                "label_statistics_after": stats,
                "label_set_preserved": True,
                "maximum_label_volume_drift_relative": 0.0,
                "quality_after": tetra_quality(unchanged),
            },
        )

    new_points = mesh.points.copy()
    point_map = np.arange(mesh.n_points, dtype=np.int64)
    removed_cells = np.zeros(mesh.n_cells, dtype=bool)
    for collapse in selected:
        left = int(collapse["left"])
        right = int(collapse["right"])
        new_points[left] = np.asarray(collapse["point"], dtype=float)
        point_map[right] = left
        removed_cells[np.asarray(collapse["removed"], dtype=np.int64)] = True
    surviving_parents = np.flatnonzero(~removed_cells)
    new_tetra = point_map[mesh.tetra[surviving_parents]]
    used_points = np.unique(new_tetra)
    compact = np.full(mesh.n_points, -1, dtype=np.int64)
    compact[used_points] = np.arange(len(used_points), dtype=np.int64)
    new_tetra = compact[new_tetra]

    merged_data = {name: np.asarray(data) for name, data in mesh.cell_data.items()}
    if parent_cell_data:
        for name, data in parent_cell_data.items():
            parent_values = np.asarray(data)
            if parent_values.shape[0] != mesh.n_cells:
                raise ValueError(f"parent_cell_data[{name!r}] must have one value per parent cell")
            merged_data[str(name)] = parent_values
    child_data = {name: data[surviving_parents].copy() for name, data in merged_data.items()}
    child_data["remesh_parent_cell"] = surviving_parents.copy()
    previous_generation = np.asarray(
        merged_data.get("remesh_generation", np.zeros(mesh.n_cells, dtype=np.int16)), dtype=np.int16
    )
    child_data["remesh_generation"] = previous_generation[surviving_parents]
    parent_to_child = np.full(mesh.n_cells, -1, dtype=np.int64)
    parent_to_child[surviving_parents] = np.arange(len(surviving_parents), dtype=np.int64)

    extensive = {} if extensive_parent_cell_data is None else {
        str(name): np.asarray(data) for name, data in extensive_parent_cell_data.items()
    }
    for name, parent_values in extensive.items():
        if parent_values.shape != (mesh.n_cells,):
            raise ValueError(f"extensive_parent_cell_data[{name!r}] must be scalar per parent cell")
        child_data[name] = parent_values[surviving_parents].copy()
    project_names = {str(name) for name in project_parent_cell_data}
    coarsened_geometry = TetMesh(
        new_points[used_points],
        new_tetra,
        mesh.cell_tags[surviving_parents],
        mesh.tag_names.copy(),
        child_data,
    )
    coarsened_volumes = coarsened_geometry.cell_volumes()

    for collapse in selected:
        patch = np.asarray(collapse["patch"], dtype=np.int64)
        child_positions = parent_to_child[patch]
        child_positions = child_positions[child_positions >= 0]
        patch_labels = np.unique(labels[patch])
        for label in patch_labels:
            parent_group = patch[labels[patch] == label]
            child_group = child_positions[
                labels[surviving_parents[child_positions]] == label
            ]
            if len(child_group) == 0:
                raise RuntimeError("safe collapse lost a source-label patch")
            geometric_weights = coarsened_volumes[child_group]
            geometric_weights = geometric_weights / geometric_weights.sum()
            for name, parent_values in extensive.items():
                child_data[name][child_group] = float(parent_values[parent_group].sum()) * geometric_weights
            projection_weights = (
                extensive.get("material_reference_volume", mesh.cell_volumes())[parent_group]
            )
            projection_weights = projection_weights / projection_weights.sum()
            for name in project_names:
                if name not in merged_data:
                    continue
                average = np.average(merged_data[name][parent_group], axis=0, weights=projection_weights)
                if name == "elastic_history_F" and np.linalg.det(average) <= 0.0:
                    average = merged_data[name][parent_group[int(np.argmax(projection_weights))]]
                child_data[name][child_group] = average
            child_data["remesh_generation"][child_group] += 1

    coarsened = TetMesh(
        new_points[used_points],
        new_tetra,
        mesh.cell_tags[surviving_parents],
        mesh.tag_names.copy(),
        child_data,
    )
    before = _label_statistics(mesh, labels)
    after_labels = np.asarray(coarsened.cell_data.get("source_label", coarsened.cell_tags), dtype=np.int64)
    after = _label_statistics(coarsened, after_labels)
    drifts = [
        abs(float(after[label]["volume"]) - float(values_before["volume"]))
        / max(abs(float(values_before["volume"])), 1.0e-30)
        for label, values_before in before.items()
    ]
    extensive_errors = {
        name: float(
            abs(child_data[name].sum() - parent_values.sum())
            / max(abs(float(parent_values.sum())), 1.0e-30)
        )
        for name, parent_values in extensive.items()
    }
    report: dict[str, object] = {
        "method": "label_safe_edge_collapse",
        "selected_collapses": len(selected),
        "candidate_edges": len(candidate_scores),
        "rejection_counts": rejection_counts,
        "points_before": mesh.n_points,
        "points_after": coarsened.n_points,
        "tetrahedra_before": mesh.n_cells,
        "tetrahedra_after": coarsened.n_cells,
        "removed_tetrahedra": int(np.count_nonzero(removed_cells)),
        "label_statistics_before": before,
        "label_statistics_after": after,
        "label_set_preserved": set(before) == set(after),
        "maximum_label_volume_drift_relative": float(max(drifts, default=0.0)),
        "extensive_quantity_conservation_relative": extensive_errors,
        "quality_before": tetra_quality(mesh),
        "quality_after": tetra_quality(coarsened),
        "history_transfer": (
            "surviving cells inherit parent state; selected projected fields use a "
            "same-label material-volume-weighted local projection"
        ),
        "note": (
            "Collapses never change source_label. Endpoints must share the same label "
            "neighborhood and satisfy boundary, topology, volume and Jacobian checks."
        ),
    }
    return CoarseningResult(coarsened, surviving_parents, report)

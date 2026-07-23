from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Mapping

import numpy as np

from .material import fiber_reinforcement_growth, lame_from_young_poisson, neo_hookean_growth
from .mesh import TetMesh
from .contact import ContactConstraints

try:
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import MatrixRankWarning, spsolve

    HAVE_SCIPY = True
except ImportError:  # Small validation cases can use the dense fallback.
    HAVE_SCIPY = False
    coo_matrix = None
    spsolve = None
    MatrixRankWarning = Warning


@dataclass(frozen=True)
class Material:
    young: float
    poisson: float = 0.45
    fiber_stiffness: float = 0.0
    fiber_direction: tuple[float, float, float] | None = None
    fiber_tension_only: bool = True

    @property
    def lame(self) -> tuple[float, float]:
        return lame_from_young_poisson(self.young, self.poisson)


@dataclass
class SolverOptions:
    increments: int = 12
    max_iterations: int = 30
    relative_tolerance: float = 1.0e-7
    absolute_tolerance: float = 1.0e-8
    min_line_search: float = 1.0e-7
    jacobian_floor: float = 1.0e-8
    regularization: float = 1.0e-10
    dense_dof_limit: int = 4000
    bulk_modulus_ratio_cap: float | None = 100.0
    verbose: bool = True


@dataclass
class IterationRecord:
    increment: int
    fraction: float
    iteration: int
    residual_norm: float
    energy: float
    step_length: float
    min_jacobian: float


@dataclass
class MorphResult:
    points: np.ndarray
    displacement: np.ndarray
    growth_lambda: np.ndarray
    j_total: np.ndarray
    j_elastic: np.ndarray
    target_reference_volume: float
    target_current_volume: float
    target_volume_ratio: float
    target_name: str = "SAT"
    desired_volume_ratio: float | None = None
    calibration_records: list[dict[str, float]] = field(default_factory=list)
    contact_summary: dict[str, object] = field(default_factory=dict)
    locking_control: dict[str, object] = field(default_factory=dict)
    volume_by_region: dict[str, dict[str, float]] = field(default_factory=dict)
    volume_by_source_label: dict[str, dict[str, float]] = field(default_factory=dict)
    records: list[IterationRecord] = field(default_factory=list)

    @property
    def actual_volume_ratio(self) -> float:
        return self.target_current_volume / self.target_reference_volume

    @property
    def sat_reference_volume(self) -> float:
        return self.target_reference_volume

    @property
    def sat_current_volume(self) -> float:
        return self.target_current_volume

    def summary(self) -> dict[str, object]:
        desired = (
            self.target_volume_ratio
            if self.desired_volume_ratio is None
            else self.desired_volume_ratio
        )
        return {
            "target_name": self.target_name,
            "target_volume_ratio_unconstrained": self.target_volume_ratio,
            "actual_target_volume_ratio": self.actual_volume_ratio,
            "desired_target_volume_ratio": desired,
            "target_reference_volume": self.target_reference_volume,
            "target_current_volume": self.target_current_volume,
            "target_volume_error_percent": float(
                100.0 * (self.actual_volume_ratio - desired) / desired
            ),
            "actual_sat_volume_ratio": self.actual_volume_ratio,
            "sat_reference_volume": self.target_reference_volume,
            "sat_current_volume": self.target_current_volume,
            "minimum_total_jacobian": float(np.min(self.j_total)),
            "maximum_displacement": float(np.linalg.norm(self.displacement, axis=1).max()),
            "volume_by_region": self.volume_by_region,
            "volume_by_source_label": self.volume_by_source_label,
            "calibration_iterations": self.calibration_records,
            "contact": self.contact_summary,
            "locking_control": self.locking_control,
            "iterations": [asdict(record) for record in self.records],
        }


class _Assembler:
    def __init__(
        self,
        mesh: TetMesh,
        materials: Mapping[int, Material],
        default_material: Material,
        dense_dof_limit: int,
        cell_materials: np.ndarray | None = None,
        contact: ContactConstraints | None = None,
        bulk_modulus_ratio_cap: float | None = None,
    ) -> None:
        self.mesh = mesh
        self.volumes, self.grads, _ = mesh.reference_geometry()
        self.n_dof = mesh.n_points * 3
        self.contact = contact
        self.use_sparse = HAVE_SCIPY
        if not self.use_sparse and self.n_dof > dense_dof_limit:
            raise RuntimeError(
                f"SciPy is required for {self.n_dof} DOFs; dense fallback limit is {dense_dof_limit}"
            )

        if cell_materials is not None and len(cell_materials) != mesh.n_cells:
            raise ValueError("cell_materials must have one Material per tetrahedron")
        self.mu = np.empty(mesh.n_cells, dtype=float)
        self.kappa = np.empty(mesh.n_cells, dtype=float)
        self.fiber_stiffness = np.zeros(mesh.n_cells, dtype=float)
        self.fiber_tension_only = np.ones(mesh.n_cells, dtype=bool)
        self.fiber_direction = np.zeros((mesh.n_cells, 3), dtype=float)
        mesh_fibers = mesh.cell_data.get("fiber_direction")
        raw_history = mesh.cell_data.get("elastic_history_F")
        self.elastic_history = (
            np.repeat(np.eye(3)[None, :, :], mesh.n_cells, axis=0)
            if raw_history is None
            else np.asarray(raw_history, dtype=float)
        )
        if self.elastic_history.shape != (mesh.n_cells, 3, 3):
            raise ValueError("elastic_history_F must have shape (n_cells, 3, 3)")
        if np.any(np.linalg.det(self.elastic_history) <= 0.0):
            raise ValueError("elastic_history_F must have positive determinants")
        raw_material_volume = mesh.cell_data.get("material_reference_volume")
        self.integration_volumes = (
            self.volumes.copy()
            if raw_material_volume is None
            else np.asarray(raw_material_volume, dtype=float)
        )
        if self.integration_volumes.shape != (mesh.n_cells,):
            raise ValueError("material_reference_volume must contain one value per cell")
        if np.any(self.integration_volumes <= 0.0):
            raise ValueError("material_reference_volume must be positive")
        self.bulk_modulus_capped_cells = 0
        for cell, tag in enumerate(mesh.cell_tags):
            material = cell_materials[cell] if cell_materials is not None else materials.get(int(tag), default_material)
            self.mu[cell], self.kappa[cell] = material.lame
            if bulk_modulus_ratio_cap is not None and bulk_modulus_ratio_cap > 0.0:
                capped = bulk_modulus_ratio_cap * self.mu[cell]
                if self.kappa[cell] > capped:
                    self.kappa[cell] = capped
                    self.bulk_modulus_capped_cells += 1
            self.fiber_stiffness[cell] = material.fiber_stiffness
            self.fiber_tension_only[cell] = material.fiber_tension_only
            if mesh_fibers is not None:
                self.fiber_direction[cell] = np.asarray(mesh_fibers[cell], dtype=float)
            elif material.fiber_direction is not None:
                self.fiber_direction[cell] = np.asarray(material.fiber_direction, dtype=float)

        local = np.arange(3, dtype=np.int64)
        self.element_dofs = (
            mesh.tetra[:, :, None] * 3 + local[None, None, :]
        ).reshape(mesh.n_cells, 12)

    def assemble(
        self,
        displacement: np.ndarray,
        growth: np.ndarray,
        need_stiffness: bool,
    ) -> tuple[float, np.ndarray, object | None, np.ndarray, np.ndarray]:
        current = self.mesh.points + displacement
        if not need_stiffness:
            return self._assemble_vectorized_residual(current, growth)
        residual = np.zeros(self.n_dof, dtype=float)
        energy_total = 0.0
        j_total = np.empty(self.mesh.n_cells, dtype=float)
        j_elastic = np.empty(self.mesh.n_cells, dtype=float)

        if need_stiffness and self.use_sparse:
            entries = self.mesh.n_cells * 144
            rows = np.empty(entries, dtype=np.int64)
            cols = np.empty(entries, dtype=np.int64)
            values = np.empty(entries, dtype=float)
        elif need_stiffness:
            stiffness = np.zeros((self.n_dof, self.n_dof), dtype=float)

        offset = 0
        for cell, nodes in enumerate(self.mesh.tetra):
            x = current[nodes]
            f = x.T @ self.grads[cell]
            w, p, tangent, jt, je = neo_hookean_growth(
                f,
                float(growth[cell]),
                self.mu[cell],
                self.kappa[cell],
                elastic_history=self.elastic_history[cell],
            )
            if self.fiber_stiffness[cell] > 0.0:
                wf, pf, af = fiber_reinforcement_growth(
                    f,
                    float(growth[cell]),
                    self.fiber_direction[cell],
                    self.fiber_stiffness[cell],
                    tension_only=bool(self.fiber_tension_only[cell]),
                    elastic_history=self.elastic_history[cell],
                )
                w += wf
                p += pf
                tangent += af
            volume = self.integration_volumes[cell]
            energy_total += volume * w
            j_total[cell] = jt
            j_elastic[cell] = je

            local_residual = volume * np.einsum(
                "ij,aj->ai", p, self.grads[cell], optimize=True
            )
            dofs = self.element_dofs[cell]
            np.add.at(residual, dofs, local_residual.ravel())

            if need_stiffness:
                local_stiffness = volume * np.einsum(
                    "aJ,iJkL,bL->aibk",
                    self.grads[cell],
                    tangent,
                    self.grads[cell],
                    optimize=True,
                )
                local_matrix = local_stiffness.reshape(12, 12)
                if self.use_sparse:
                    block = slice(offset, offset + 144)
                    rows[block] = np.repeat(dofs, 12)
                    cols[block] = np.tile(dofs, 12)
                    values[block] = local_matrix.ravel()
                    offset += 144
                else:
                    stiffness[np.ix_(dofs, dofs)] += local_matrix

        matrix = None
        if need_stiffness:
            if self.use_sparse:
                matrix = coo_matrix((values, (rows, cols)), shape=(self.n_dof, self.n_dof)).tocsr()
            else:
                matrix = stiffness
        if self.contact is not None:
            contact_energy, contact_residual, contact_matrix = self._assemble_contact(
                current, need_stiffness
            )
            energy_total += contact_energy
            residual += contact_residual
            if need_stiffness:
                matrix = matrix + contact_matrix
        return energy_total, residual, matrix, j_total, j_elastic

    def _assemble_vectorized_residual(self, current: np.ndarray, growth: np.ndarray):
        x = current[self.mesh.tetra]
        deformation = np.einsum("eai,eaj->eij", x, self.grads, optimize=True)
        j_total = np.linalg.det(deformation)
        history_increment = self.elastic_history / growth[:, None, None]
        fe = np.einsum("eij,ejk->eik", deformation, history_increment, optimize=True)
        j_elastic = np.linalg.det(fe)
        if np.any(j_total <= 0.0) or np.any(j_elastic <= 0.0):
            raise FloatingPointError("inverted element")
        log_j = np.log(j_elastic)
        i1 = np.sum(fe * fe, axis=(1, 2))
        energy_density = (
            0.5 * self.mu * (i1 - 3.0)
            - self.mu * log_j
            + 0.5 * self.kappa * log_j**2
        )
        inv_fe_t = np.linalg.inv(fe).transpose(0, 2, 1)
        coefficient_b = -self.mu + self.kappa * log_j
        elastic_stress = (
            self.mu[:, None, None] * fe
            + coefficient_b[:, None, None] * inv_fe_t
        )
        stress = np.einsum(
            "eik,ejk->eij", elastic_stress, history_increment, optimize=True
        )
        for cell in np.flatnonzero(self.fiber_stiffness > 0.0):
            wf, pf, _ = fiber_reinforcement_growth(
                deformation[cell],
                float(growth[cell]),
                self.fiber_direction[cell],
                self.fiber_stiffness[cell],
                tension_only=bool(self.fiber_tension_only[cell]),
                elastic_history=self.elastic_history[cell],
            )
            energy_density[cell] += wf
            stress[cell] += pf
        local_residual = self.integration_volumes[:, None, None] * np.einsum(
            "eij,eaj->eai", stress, self.grads, optimize=True
        )
        residual = np.zeros(self.n_dof, dtype=float)
        np.add.at(residual, self.element_dofs.ravel(), local_residual.ravel())
        energy = float(np.dot(self.integration_volumes, energy_density))
        if self.contact is not None:
            contact_energy, contact_residual, _ = self._assemble_contact(current, False)
            energy += contact_energy
            residual += contact_residual
        return energy, residual, None, j_total, j_elastic

    def _assemble_contact(self, current: np.ndarray, need_stiffness: bool):
        residual = np.zeros(self.n_dof, dtype=float)
        state, gaps = self.contact.evaluate(current)
        active = np.flatnonzero(gaps < 0.0)
        if self.use_sparse:
            from scipy.sparse import coo_matrix

            matrix = coo_matrix((self.n_dof, self.n_dof)).tocsr()
        else:
            matrix = np.zeros((self.n_dof, self.n_dof), dtype=float)
        energy = 0.0
        rows: list[int] = []
        cols: list[int] = []
        values: list[float] = []
        for index in active:
            nodes = state.node_ids[index]
            gradient = (
                state.coefficients[index, :, None]
                * state.normals[index][None, :]
            ).reshape(-1)
            dofs = (nodes[:, None] * 3 + np.arange(3)[None, :]).reshape(-1)
            stiffness = float(state.penalty[index])
            gap = float(gaps[index])
            energy += 0.5 * stiffness * gap * gap
            np.add.at(residual, dofs, stiffness * gap * gradient)
            if need_stiffness:
                local = stiffness * np.outer(gradient, gradient)
                if self.use_sparse:
                    rows.extend(np.repeat(dofs, len(dofs)).tolist())
                    cols.extend(np.tile(dofs, len(dofs)).tolist())
                    values.extend(local.ravel().tolist())
                else:
                    matrix[np.ix_(dofs, dofs)] += local
        if need_stiffness and self.use_sparse and rows:
            matrix = coo_matrix(
                (values, (rows, cols)), shape=(self.n_dof, self.n_dof)
            ).tocsr()
        return energy, residual, matrix


def _solve_direction(matrix: object, rhs: np.ndarray, regularization: float) -> np.ndarray:
    if HAVE_SCIPY and hasattr(matrix, "tocsr"):
        diag = np.asarray(matrix.diagonal())
        scale = max(float(np.mean(np.abs(diag))), 1.0)
        from scipy.sparse import eye

        regularized = matrix + (regularization * scale) * eye(matrix.shape[0], format="csr")
        direction = np.asarray(spsolve(regularized, rhs), dtype=float)
    else:
        dense = np.asarray(matrix, dtype=float)
        scale = max(float(np.mean(np.abs(np.diag(dense)))), 1.0)
        direction = np.linalg.solve(
            dense + regularization * scale * np.eye(len(dense)), rhs
        )
    if not np.all(np.isfinite(direction)):
        raise RuntimeError("linear solve produced non-finite values; check constraints and mesh quality")
    return direction


def morph_target_region(
    mesh: TetMesh,
    target_cells: np.ndarray,
    fixed_nodes: np.ndarray,
    target_volume_ratio: float,
    materials: Mapping[int, Material] | None = None,
    default_material: Material = Material(young=10_000.0, poisson=0.45),
    cell_materials: np.ndarray | None = None,
    options: SolverOptions | None = None,
    target_name: str = "target",
    contact: ContactConstraints | None = None,
    initial_displacement: np.ndarray | None = None,
    initial_growth_volume_ratio: float = 1.0,
) -> MorphResult:
    """Morph selected cells by prescribed isotropic inelastic strain."""
    options = options or SolverOptions()
    target_cells = np.asarray(target_cells, dtype=bool)
    fixed_nodes = np.unique(np.asarray(fixed_nodes, dtype=np.int64))
    if target_cells.shape != (mesh.n_cells,):
        raise ValueError("target_cells must have one boolean per tetrahedron")
    if not np.any(target_cells):
        raise ValueError("target selection is empty")
    if fixed_nodes.size == 0:
        raise ValueError("at least one fixed node set is required to remove rigid-body modes")
    if fixed_nodes.min() < 0 or fixed_nodes.max() >= mesh.n_points:
        raise ValueError("fixed_nodes contains an invalid point index")
    if target_volume_ratio <= 0.0:
        raise ValueError("target_volume_ratio must be positive")
    if initial_growth_volume_ratio <= 0.0:
        raise ValueError("initial_growth_volume_ratio must be positive")
    if options.increments < 1:
        raise ValueError("increments must be at least one")

    assembler = _Assembler(
        mesh,
        materials or {},
        default_material,
        dense_dof_limit=options.dense_dof_limit,
        cell_materials=cell_materials,
        contact=contact,
        bulk_modulus_ratio_cap=options.bulk_modulus_ratio_cap,
    )
    target_lambda = target_volume_ratio ** (1.0 / 3.0)
    starting_lambda = initial_growth_volume_ratio ** (1.0 / 3.0)
    displacement = (
        np.zeros_like(mesh.points)
        if initial_displacement is None
        else np.asarray(initial_displacement, dtype=float).copy()
    )
    if displacement.shape != mesh.points.shape:
        raise ValueError("initial_displacement must match mesh points")
    displacement[fixed_nodes] = 0.0

    fixed_dofs = (fixed_nodes[:, None] * 3 + np.arange(3)[None, :]).ravel()
    free_mask = np.ones(mesh.n_points * 3, dtype=bool)
    free_mask[fixed_dofs] = False
    free_dofs = np.flatnonzero(free_mask)
    if free_dofs.size == 0:
        raise ValueError("all nodes are fixed")

    records: list[IterationRecord] = []
    final_growth = np.ones(mesh.n_cells, dtype=float)
    final_jt = np.ones(mesh.n_cells, dtype=float)
    final_je = np.ones(mesh.n_cells, dtype=float)

    for increment in range(1, options.increments + 1):
        fraction = increment / options.increments
        current_lambda = float(
            np.exp(
                (1.0 - fraction) * np.log(starting_lambda)
                + fraction * np.log(target_lambda)
            )
        )
        growth = np.ones(mesh.n_cells, dtype=float)
        growth[target_cells] = current_lambda
        first_norm: float | None = None

        for iteration in range(options.max_iterations + 1):
            energy, residual, stiffness, jt, je = assembler.assemble(
                displacement, growth, need_stiffness=True
            )
            free_residual = residual[free_dofs]
            norm = float(np.linalg.norm(free_residual))
            if first_norm is None:
                first_norm = max(norm, 1.0)
            threshold = options.absolute_tolerance + options.relative_tolerance * first_norm
            if options.verbose:
                print(
                    f"increment {increment:02d}/{options.increments:02d} "
                    f"iteration {iteration:02d}: |R|={norm:.6e}, "
                    f"E={energy:.6e}, min(J)={jt.min():.6e}"
                )
            if norm <= threshold:
                records.append(
                    IterationRecord(
                        increment, fraction, iteration, norm, energy, 0.0, float(jt.min())
                    )
                )
                final_growth, final_jt, final_je = growth, jt, je
                break
            if iteration == options.max_iterations:
                raise RuntimeError(
                    f"Newton solver did not converge at increment {increment}; "
                    "increase --increments or inspect mesh/material parameters"
                )

            if HAVE_SCIPY and hasattr(stiffness, "tocsr"):
                reduced = stiffness[free_dofs][:, free_dofs]
            else:
                reduced = stiffness[np.ix_(free_dofs, free_dofs)]
            direction_free = _solve_direction(
                reduced, -free_residual, options.regularization
            )
            directional_derivative = float(free_residual @ direction_free)
            if directional_derivative >= 0.0:
                raise RuntimeError(
                    "Newton direction is not an energy descent direction; "
                    "increase increments or reduce Poisson ratio"
                )

            full_direction = np.zeros(mesh.n_points * 3, dtype=float)
            full_direction[free_dofs] = direction_free
            full_direction = full_direction.reshape((-1, 3))
            step = 1.0
            accepted = False
            while step >= options.min_line_search:
                trial = displacement + step * full_direction
                try:
                    trial_energy, _, _, trial_jt, _ = assembler.assemble(
                        trial, growth, need_stiffness=False
                    )
                except FloatingPointError:
                    step *= 0.5
                    continue
                armijo = energy + 1.0e-4 * step * directional_derivative
                if trial_jt.min() > options.jacobian_floor and trial_energy <= armijo:
                    displacement = trial
                    accepted = True
                    records.append(
                        IterationRecord(
                            increment,
                            fraction,
                            iteration,
                            norm,
                            energy,
                            step,
                            float(trial_jt.min()),
                        )
                    )
                    break
                step *= 0.5
            if not accepted:
                raise RuntimeError(
                    f"line search failed at increment {increment}, iteration {iteration}; "
                    "increase --increments or improve tetrahedral mesh quality"
                )
        else:  # pragma: no cover
            raise RuntimeError("unexpected Newton loop termination")

    current_points = mesh.points + displacement
    reference_cell_volumes = mesh.cell_volumes()
    current_cell_volumes = mesh.cell_volumes(current_points)
    target_reference = float(reference_cell_volumes[target_cells].sum())
    target_current = float(current_cell_volumes[target_cells].sum())
    return MorphResult(
        points=current_points,
        displacement=displacement,
        growth_lambda=final_growth,
        j_total=final_jt,
        j_elastic=final_je,
        target_reference_volume=target_reference,
        target_current_volume=target_current,
        target_volume_ratio=target_volume_ratio,
        target_name=target_name,
        contact_summary=contact.summary(current_points) if contact is not None else {},
        locking_control={
            "method": "bulk_modulus_ratio_cap",
            "kappa_over_mu_cap": options.bulk_modulus_ratio_cap,
            "capped_cell_count": assembler.bulk_modulus_capped_cells,
            "note": "locking mitigation for linear tetrahedra; not a mixed u-p element",
        },
        volume_by_region=_volume_table(mesh.cell_tags, reference_cell_volumes, current_cell_volumes, mesh.tag_names),
        volume_by_source_label=_volume_table(
            mesh.cell_data["source_label"],
            reference_cell_volumes,
            current_cell_volumes,
            {},
        )
        if "source_label" in mesh.cell_data
        else {},
        records=records,
    )


def morph_sat(
    mesh: TetMesh,
    sat_cells: np.ndarray,
    fixed_nodes: np.ndarray,
    target_volume_ratio: float,
    materials: Mapping[int, Material] | None = None,
    default_material: Material = Material(young=10_000.0, poisson=0.45),
    cell_materials: np.ndarray | None = None,
    options: SolverOptions | None = None,
) -> MorphResult:
    """Backward-compatible SAT entry point."""
    return morph_target_region(
        mesh,
        sat_cells,
        fixed_nodes,
        target_volume_ratio,
        materials=materials,
        default_material=default_material,
        cell_materials=cell_materials,
        options=options,
        target_name="SAT",
    )


def _volume_table(
    labels: np.ndarray,
    reference_cell_volumes: np.ndarray,
    current_cell_volumes: np.ndarray,
    tag_names: Mapping[str, int],
) -> dict[str, dict[str, float]]:
    names_by_value = {int(value): str(name) for name, value in tag_names.items()}
    out: dict[str, dict[str, float]] = {}
    for raw_value in np.unique(labels):
        value = int(raw_value)
        mask = labels == value
        reference = float(reference_cell_volumes[mask].sum())
        current = float(current_cell_volumes[mask].sum())
        key = names_by_value.get(value, str(value))
        out[key] = {
            "id": value,
            "reference_volume": reference,
            "current_volume": current,
            "volume_ratio": current / reference if reference > 0.0 else 0.0,
            "volume_change": current - reference,
            "volume_change_percent": 100.0 * (current - reference) / reference
            if reference > 0.0
            else 0.0,
        }
    return out

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Mapping

import numpy as np

from .material import lame_from_young_poisson, neo_hookean_growth
from .mesh import TetMesh

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
    sat_reference_volume: float
    sat_current_volume: float
    target_volume_ratio: float
    records: list[IterationRecord] = field(default_factory=list)

    @property
    def actual_volume_ratio(self) -> float:
        return self.sat_current_volume / self.sat_reference_volume

    def summary(self) -> dict[str, object]:
        return {
            "target_volume_ratio_unconstrained": self.target_volume_ratio,
            "actual_sat_volume_ratio": self.actual_volume_ratio,
            "sat_reference_volume": self.sat_reference_volume,
            "sat_current_volume": self.sat_current_volume,
            "minimum_total_jacobian": float(np.min(self.j_total)),
            "maximum_displacement": float(np.linalg.norm(self.displacement, axis=1).max()),
            "iterations": [asdict(record) for record in self.records],
        }


class _Assembler:
    def __init__(
        self,
        mesh: TetMesh,
        materials: Mapping[int, Material],
        default_material: Material,
        dense_dof_limit: int,
    ) -> None:
        self.mesh = mesh
        self.volumes, self.grads, _ = mesh.reference_geometry()
        self.n_dof = mesh.n_points * 3
        self.use_sparse = HAVE_SCIPY
        if not self.use_sparse and self.n_dof > dense_dof_limit:
            raise RuntimeError(
                f"SciPy is required for {self.n_dof} DOFs; dense fallback limit is {dense_dof_limit}"
            )

        self.mu = np.empty(mesh.n_cells, dtype=float)
        self.kappa = np.empty(mesh.n_cells, dtype=float)
        for cell, tag in enumerate(mesh.cell_tags):
            material = materials.get(int(tag), default_material)
            self.mu[cell], self.kappa[cell] = material.lame

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
                f, float(growth[cell]), self.mu[cell], self.kappa[cell]
            )
            volume = self.volumes[cell]
            energy_total += volume * w
            j_total[cell] = jt
            j_elastic[cell] = je

            local_residual = np.empty((4, 3), dtype=float)
            for a in range(4):
                local_residual[a] = volume * (p @ self.grads[cell, a])
            dofs = self.element_dofs[cell]
            np.add.at(residual, dofs, local_residual.ravel())

            if need_stiffness:
                local_stiffness = np.empty((4, 3, 4, 3), dtype=float)
                for a in range(4):
                    for b in range(4):
                        local_stiffness[a, :, b, :] = volume * np.einsum(
                            "J,iJkL,L->ik",
                            self.grads[cell, a],
                            tangent,
                            self.grads[cell, b],
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
        return energy_total, residual, matrix, j_total, j_elastic


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


def morph_sat(
    mesh: TetMesh,
    sat_cells: np.ndarray,
    fixed_nodes: np.ndarray,
    target_volume_ratio: float,
    materials: Mapping[int, Material] | None = None,
    default_material: Material = Material(young=10_000.0, poisson=0.45),
    options: SolverOptions | None = None,
) -> MorphResult:
    """Morph SAT by prescribed isotropic inelastic strain and static equilibrium."""
    options = options or SolverOptions()
    sat_cells = np.asarray(sat_cells, dtype=bool)
    fixed_nodes = np.unique(np.asarray(fixed_nodes, dtype=np.int64))
    if sat_cells.shape != (mesh.n_cells,):
        raise ValueError("sat_cells must have one boolean per tetrahedron")
    if not np.any(sat_cells):
        raise ValueError("SAT selection is empty")
    if fixed_nodes.size == 0:
        raise ValueError("at least one fixed node set is required to remove rigid-body modes")
    if fixed_nodes.min() < 0 or fixed_nodes.max() >= mesh.n_points:
        raise ValueError("fixed_nodes contains an invalid point index")
    if target_volume_ratio <= 0.0:
        raise ValueError("target_volume_ratio must be positive")
    if options.increments < 1:
        raise ValueError("increments must be at least one")

    assembler = _Assembler(
        mesh, materials or {}, default_material, dense_dof_limit=options.dense_dof_limit
    )
    target_lambda = target_volume_ratio ** (1.0 / 3.0)
    displacement = np.zeros_like(mesh.points)

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
        current_lambda = float(np.exp(fraction * np.log(target_lambda)))
        growth = np.ones(mesh.n_cells, dtype=float)
        growth[sat_cells] = current_lambda
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
    sat_reference = float(reference_cell_volumes[sat_cells].sum())
    sat_current = float(current_cell_volumes[sat_cells].sum())
    return MorphResult(
        points=current_points,
        displacement=displacement,
        growth_lambda=final_growth,
        j_total=final_jt,
        j_elastic=final_je,
        sat_reference_volume=sat_reference,
        sat_current_volume=sat_current,
        target_volume_ratio=target_volume_ratio,
        records=records,
    )


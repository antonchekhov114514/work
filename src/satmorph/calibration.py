from __future__ import annotations

from dataclasses import replace
from typing import Mapping

import numpy as np

from .mesh import TetMesh
from .solver import Material, MorphResult, SolverOptions, morph_target_region


def calibrate_target_volume(
    mesh: TetMesh,
    target_cells: np.ndarray,
    fixed_nodes: np.ndarray,
    desired_volume_ratio: float,
    *,
    materials: Mapping[int, Material] | None = None,
    default_material: Material = Material(young=10_000.0, poisson=0.45),
    cell_materials: np.ndarray | None = None,
    options: SolverOptions | None = None,
    target_name: str = "target",
    tolerance: float = 2.5e-3,
    max_corrections: int = 4,
    relaxation: float = 0.8,
    ratio_bounds: tuple[float, float] = (0.1, 4.0),
    max_solver_retries: int = 3,
    increment_retry_factor: int = 2,
    contact=None,
) -> MorphResult:
    """Correct prescribed growth until the constrained target volume is reached."""
    if desired_volume_ratio <= 0.0:
        raise ValueError("desired_volume_ratio must be positive")
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
    if max_corrections < 0:
        raise ValueError("max_corrections must be non-negative")
    if max_solver_retries < 0:
        raise ValueError("max_solver_retries must be non-negative")
    if increment_retry_factor < 2:
        raise ValueError("increment_retry_factor must be at least two")
    lower, upper = ratio_bounds
    if not 0.0 < lower < upper:
        raise ValueError("ratio_bounds must satisfy 0 < lower < upper")

    requested = float(np.clip(desired_volume_ratio, lower, upper))
    history: list[dict[str, float]] = []
    result: MorphResult | None = None
    previous_result: MorphResult | None = None
    previous_requested = 1.0
    current_options = options or SolverOptions()
    for iteration in range(max_corrections + 1):
        solver_retries = 0
        while True:
            try:
                result = morph_target_region(
                    mesh,
                    target_cells,
                    fixed_nodes,
                    requested,
                    materials=materials,
                    default_material=default_material,
                    cell_materials=cell_materials,
                    options=current_options,
                    target_name=target_name,
                    contact=contact,
                    initial_displacement=(
                        None if previous_result is None else previous_result.displacement
                    ),
                    initial_growth_volume_ratio=previous_requested,
                )
                break
            except RuntimeError as exc:
                retryable = any(
                    marker in str(exc)
                    for marker in (
                        "line search failed",
                        "did not converge",
                        "not an energy descent direction",
                    )
                )
                if not retryable or solver_retries >= max_solver_retries:
                    raise
                solver_retries += 1
                previous_increments = current_options.increments
                current_options = replace(
                    current_options,
                    increments=previous_increments * increment_retry_factor,
                )
                if current_options.verbose:
                    print(
                        "\nInner solve did not converge; retrying calibration "
                        f"iteration {iteration} with "
                        f"{current_options.increments} increments "
                        f"(retry {solver_retries}/{max_solver_retries})."
                    )
        actual = result.actual_volume_ratio
        relative_error = (actual - desired_volume_ratio) / desired_volume_ratio
        history.append(
            {
                "iteration": float(iteration),
                "prescribed_growth_volume_ratio": requested,
                "actual_volume_ratio": actual,
                "relative_error": relative_error,
                "solver_increments": float(current_options.increments),
                "solver_retries": float(solver_retries),
            }
        )
        previous_result = result
        previous_requested = requested
        if abs(relative_error) <= tolerance:
            break

        if len(history) >= 2:
            older, newer = history[-2], history[-1]
            delta_actual = newer["actual_volume_ratio"] - older["actual_volume_ratio"]
            if abs(delta_actual) > 1.0e-8:
                candidate = newer["prescribed_growth_volume_ratio"] + (
                    desired_volume_ratio - newer["actual_volume_ratio"]
                ) * (
                    newer["prescribed_growth_volume_ratio"]
                    - older["prescribed_growth_volume_ratio"]
                ) / delta_actual
            else:
                candidate = requested * (desired_volume_ratio / actual) ** relaxation
        else:
            candidate = requested * (desired_volume_ratio / actual) ** relaxation
        requested = float(np.clip(candidate, lower, upper))

    assert result is not None
    result.desired_volume_ratio = desired_volume_ratio
    result.calibration_records = history
    return result

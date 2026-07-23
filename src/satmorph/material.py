from __future__ import annotations

import numpy as np


def lame_from_young_poisson(young: float, poisson: float) -> tuple[float, float]:
    if young <= 0.0:
        raise ValueError("Young's modulus must be positive")
    if not (-1.0 < poisson < 0.5):
        raise ValueError("Poisson's ratio must lie in (-1, 0.5)")
    mu = young / (2.0 * (1.0 + poisson))
    kappa = young / (3.0 * (1.0 - 2.0 * poisson))
    return float(mu), float(kappa)


def neo_hookean_growth(
    deformation_gradient: np.ndarray,
    growth_lambda: float,
    mu: float,
    kappa: float,
    elastic_history: np.ndarray | None = None,
) -> tuple[float, np.ndarray, np.ndarray, float, float]:
    """Compressible Neo-Hookean energy, stress, and consistent tangent.

    The prescribed incremental inelastic deformation is
    F_in = growth_lambda * I. If an elastic history H is supplied, the
    current elastic deformation is F_el = F * H * inv(F_in). This updated
    reference form preserves constitutive stress across remeshing.
    """
    f = np.asarray(deformation_gradient, dtype=float)
    if f.shape != (3, 3):
        raise ValueError("deformation_gradient must have shape (3, 3)")
    if growth_lambda <= 0.0:
        raise ValueError("growth_lambda must be positive")

    history = np.eye(3) if elastic_history is None else np.asarray(elastic_history, dtype=float)
    if history.shape != (3, 3):
        raise ValueError("elastic_history must have shape (3, 3)")
    history_increment = history / growth_lambda
    fe = f @ history_increment
    j_total = float(np.linalg.det(f))
    j_elastic = float(np.linalg.det(fe))
    if j_total <= 0.0 or j_elastic <= 0.0:
        raise FloatingPointError("inverted element")

    log_je = float(np.log(j_elastic))
    i1 = float(np.sum(fe * fe))
    energy = 0.5 * mu * (i1 - 3.0) - mu * log_je + 0.5 * kappa * log_je**2

    inv_fe_t = np.linalg.inv(fe).T
    b = -mu + kappa * log_je
    elastic_piola = mu * fe + b * inv_fe_t
    first_piola = elastic_piola @ history_increment.T

    identity = np.eye(3)
    elastic_tangent = mu * np.einsum("ik,MN->iMkN", identity, identity)
    elastic_tangent += kappa * np.einsum(
        "kN,iM->iMkN", inv_fe_t, inv_fe_t
    )
    elastic_tangent -= b * np.einsum(
        "kM,iN->iMkN", inv_fe_t, inv_fe_t
    )
    tangent = np.einsum(
        "iMkN,JM,LN->iJkL",
        elastic_tangent,
        history_increment,
        history_increment,
        optimize=True,
    )
    return float(energy), first_piola, tangent, j_total, j_elastic


def fiber_reinforcement_growth(
    deformation_gradient: np.ndarray,
    growth_lambda: float,
    fiber_direction: np.ndarray,
    fiber_stiffness: float,
    *,
    tension_only: bool = True,
    elastic_history: np.ndarray | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Quadratic fiber reinforcement expressed in the reference direction."""
    if fiber_stiffness <= 0.0:
        return 0.0, np.zeros((3, 3)), np.zeros((3, 3, 3, 3))
    direction = np.asarray(fiber_direction, dtype=float)
    length = float(np.linalg.norm(direction))
    if length <= 0.0:
        return 0.0, np.zeros((3, 3)), np.zeros((3, 3, 3, 3))
    a = direction / length
    f = np.asarray(deformation_gradient, dtype=float)
    history = np.eye(3) if elastic_history is None else np.asarray(elastic_history, dtype=float)
    if history.shape != (3, 3):
        raise ValueError("elastic_history must have shape (3, 3)")
    material_direction = history @ a / growth_lambda
    v = f @ material_direction
    i4 = float(v @ v)
    strain = i4 - 1.0
    if tension_only and strain <= 0.0:
        return 0.0, np.zeros((3, 3)), np.zeros((3, 3, 3, 3))
    energy = 0.5 * fiber_stiffness * strain**2
    first_piola = 2.0 * fiber_stiffness * strain * np.outer(v, material_direction)
    tangent = (
        4.0
        * fiber_stiffness
        * np.einsum(
            "i,J,k,L->iJkL", v, material_direction, v, material_direction
        )
    )
    tangent += (
        2.0
        * fiber_stiffness
        * strain
        * np.einsum(
            "ik,J,L->iJkL", np.eye(3), material_direction, material_direction
        )
    )
    return float(energy), first_piola, tangent

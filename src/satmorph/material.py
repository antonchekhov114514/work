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
) -> tuple[float, np.ndarray, np.ndarray, float, float]:
    """Compressible Neo-Hookean energy, stress, and consistent tangent.

    The prescribed inelastic deformation is F_in = growth_lambda * I and
    F_el = F * inv(F_in). Energy is measured per unit reference volume.
    """
    f = np.asarray(deformation_gradient, dtype=float)
    if f.shape != (3, 3):
        raise ValueError("deformation_gradient must have shape (3, 3)")
    if growth_lambda <= 0.0:
        raise ValueError("growth_lambda must be positive")

    j_total = float(np.linalg.det(f))
    j_elastic = j_total / growth_lambda**3
    if j_total <= 0.0 or j_elastic <= 0.0:
        raise FloatingPointError("inverted element")

    fe = f / growth_lambda
    log_je = float(np.log(j_elastic))
    i1 = float(np.sum(fe * fe))
    energy = 0.5 * mu * (i1 - 3.0) - mu * log_je + 0.5 * kappa * log_je**2

    inv_ft = np.linalg.inv(f).T
    a = mu / growth_lambda**2
    b = -mu + kappa * log_je
    first_piola = a * f + b * inv_ft

    identity = np.eye(3)
    tangent = a * np.einsum("ik,JL->iJkL", identity, identity)
    tangent += kappa * np.einsum("kL,iJ->iJkL", inv_ft, inv_ft)
    tangent -= b * np.einsum("kJ,iL->iJkL", inv_ft, inv_ft)
    return float(energy), first_piola, tangent, j_total, j_elastic


def fiber_reinforcement_growth(
    deformation_gradient: np.ndarray,
    growth_lambda: float,
    fiber_direction: np.ndarray,
    fiber_stiffness: float,
    *,
    tension_only: bool = True,
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
    v = f @ a
    i4 = float(v @ v) / growth_lambda**2
    strain = i4 - 1.0
    if tension_only and strain <= 0.0:
        return 0.0, np.zeros((3, 3)), np.zeros((3, 3, 3, 3))
    energy = 0.5 * fiber_stiffness * strain**2
    first_piola = (
        2.0 * fiber_stiffness * strain / growth_lambda**2 * np.outer(v, a)
    )
    tangent = (
        4.0
        * fiber_stiffness
        / growth_lambda**4
        * np.einsum("i,J,k,L->iJkL", v, a, v, a)
    )
    tangent += (
        2.0
        * fiber_stiffness
        * strain
        / growth_lambda**2
        * np.einsum("ik,J,L->iJkL", np.eye(3), a, a)
    )
    return float(energy), first_piola, tangent

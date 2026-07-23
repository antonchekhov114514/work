import unittest

import numpy as np

from satmorph.material import fiber_reinforcement_growth, neo_hookean_growth


class MaterialTests(unittest.TestCase):
    def test_consistent_tangent_matches_finite_difference(self):
        f = np.asarray(
            [[1.08, 0.03, -0.01], [0.02, 0.94, 0.04], [0.00, -0.02, 1.05]],
            dtype=float,
        )
        _, _, tangent, _, _ = neo_hookean_growth(f, 1.12, 3000.0, 12000.0)
        eps = 1.0e-7
        for k in range(3):
            for ell in range(3):
                perturbation = np.zeros((3, 3))
                perturbation[k, ell] = eps
                _, plus, _, _, _ = neo_hookean_growth(
                    f + perturbation, 1.12, 3000.0, 12000.0
                )
                _, minus, _, _, _ = neo_hookean_growth(
                    f - perturbation, 1.12, 3000.0, 12000.0
                )
                numerical = (plus - minus) / (2.0 * eps)
                np.testing.assert_allclose(
                    numerical, tangent[:, :, k, ell], rtol=2.0e-6, atol=2.0e-5
                )

    def test_fiber_tangent_matches_finite_difference(self):
        f = np.asarray([[1.12, 0.02, 0.0], [0.01, 1.03, 0.0], [0.0, 0.0, 0.98]])
        direction = np.asarray([1.0, 0.2, 0.0])
        _, _, tangent = fiber_reinforcement_growth(f, 1.0, direction, 25_000.0)
        eps = 1.0e-7
        for k in range(3):
            for ell in range(3):
                perturbation = np.zeros((3, 3))
                perturbation[k, ell] = eps
                _, plus, _ = fiber_reinforcement_growth(
                    f + perturbation, 1.0, direction, 25_000.0
                )
                _, minus, _ = fiber_reinforcement_growth(
                    f - perturbation, 1.0, direction, 25_000.0
                )
                numerical = (plus - minus) / (2.0 * eps)
                np.testing.assert_allclose(
                    numerical, tangent[:, :, k, ell], rtol=2.0e-5, atol=2.0e-5
                )

    def test_history_aware_tangent_matches_finite_difference(self):
        f = np.asarray(
            [[1.02, 0.01, 0.00], [-0.02, 1.01, 0.03], [0.01, 0.00, 0.99]]
        )
        history = np.asarray(
            [[1.08, 0.04, 0.00], [0.01, 0.95, 0.02], [0.00, -0.01, 1.03]]
        )
        _, _, tangent, _, _ = neo_hookean_growth(
            f, 1.03, 3000.0, 12000.0, elastic_history=history
        )
        eps = 1.0e-7
        for k in range(3):
            for ell in range(3):
                perturbation = np.zeros((3, 3))
                perturbation[k, ell] = eps
                _, plus, _, _, _ = neo_hookean_growth(
                    f + perturbation,
                    1.03,
                    3000.0,
                    12000.0,
                    elastic_history=history,
                )
                _, minus, _, _, _ = neo_hookean_growth(
                    f - perturbation,
                    1.03,
                    3000.0,
                    12000.0,
                    elastic_history=history,
                )
                numerical = (plus - minus) / (2.0 * eps)
                np.testing.assert_allclose(
                    numerical, tangent[:, :, k, ell], rtol=3.0e-6, atol=3.0e-5
                )

    def test_elastic_history_retains_nonzero_constitutive_stress(self):
        previous_f = np.diag([1.08, 0.97, 1.01])
        previous_growth = 1.02
        old_energy, _, _, _, old_je = neo_hookean_growth(
            previous_f, previous_growth, 3000.0, 12000.0
        )
        history = previous_f / previous_growth
        new_energy, new_stress, _, _, new_je = neo_hookean_growth(
            np.eye(3), 1.0, 3000.0, 12000.0, elastic_history=history
        )
        self.assertAlmostEqual(new_energy, old_energy, places=12)
        self.assertAlmostEqual(new_je, old_je, places=12)
        self.assertGreater(np.linalg.norm(new_stress), 1.0)


if __name__ == "__main__":
    unittest.main()

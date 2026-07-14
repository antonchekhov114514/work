import unittest

import numpy as np

from satmorph.material import neo_hookean_growth


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


if __name__ == "__main__":
    unittest.main()


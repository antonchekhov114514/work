import unittest

import numpy as np

from satmorph.demo import BONE, SAT, SKIN, SOFT, layered_torso_mesh
from satmorph.io import load_mesh, save_npz
from satmorph.solver import Material, SolverOptions, _Assembler, morph_sat


class SolverTests(unittest.TestCase):
    def setUp(self):
        self.mesh = layered_torso_mesh(3, 2)
        self.materials = {
            BONE: Material(10_000.0, 0.40),
            SOFT: Material(12_000.0, 0.40),
            SAT: Material(5_000.0, 0.40),
            SKIN: Material(30_000.0, 0.40),
        }

    def test_identity_growth_has_zero_residual(self):
        assembler = _Assembler(
            self.mesh, self.materials, Material(10_000.0, 0.40), dense_dof_limit=4000
        )
        _, residual, _, jt, je = assembler.assemble(
            np.zeros_like(self.mesh.points), np.ones(self.mesh.n_cells), False
        )
        self.assertLess(np.linalg.norm(residual), 1.0e-9)
        np.testing.assert_allclose(jt, 1.0, atol=1.0e-12)
        np.testing.assert_allclose(je, 1.0, atol=1.0e-12)

    def test_reference_gradients_reproduce_affine_deformation(self):
        _, gradients, _ = self.mesh.reference_geometry()
        prescribed = np.asarray(
            [[1.05, 0.02, 0.01], [0.00, 0.97, -0.03], [0.01, 0.00, 1.02]]
        )
        translated = np.asarray([0.3, -0.2, 0.1])
        current = self.mesh.points @ prescribed.T + translated
        for cell, nodes in enumerate(self.mesh.tetra):
            recovered = current[nodes].T @ gradients[cell]
            np.testing.assert_allclose(recovered, prescribed, atol=2.0e-12)

    def test_npz_round_trip(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            path = Path(directory) / "mesh.npz"
            save_npz(path, self.mesh)
            loaded = load_mesh(path)
        np.testing.assert_allclose(loaded.points, self.mesh.points)
        np.testing.assert_array_equal(loaded.tetra, self.mesh.tetra)
        np.testing.assert_array_equal(loaded.cell_tags, self.mesh.cell_tags)
        self.assertEqual(loaded.tag_names, self.mesh.tag_names)

    def test_sat_expands_and_bone_nodes_stay_fixed(self):
        sat = self.mesh.cell_tags == SAT
        fixed = self.mesh.nodes_for_tags([BONE])
        result = morph_sat(
            self.mesh,
            sat,
            fixed,
            target_volume_ratio=1.12,
            materials=self.materials,
            options=SolverOptions(
                increments=4,
                max_iterations=25,
                relative_tolerance=1.0e-7,
                absolute_tolerance=1.0e-8,
                verbose=False,
            ),
        )
        self.assertGreater(result.actual_volume_ratio, 1.0)
        np.testing.assert_allclose(result.displacement[fixed], 0.0, atol=1.0e-12)
        self.assertGreater(result.j_total.min(), 0.0)


if __name__ == "__main__":
    unittest.main()

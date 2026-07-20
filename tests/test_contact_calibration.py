import unittest

import numpy as np

from satmorph.calibration import calibrate_target_volume
from satmorph.contact import ContactConstraints, DynamicContact
from satmorph.demo import BONE, SAT, SKIN, SOFT, layered_torso_mesh
from satmorph.solver import Material, SolverOptions, _Assembler


class ContactCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.mesh = layered_torso_mesh(3, 2)
        self.materials = {
            BONE: Material(10_000.0, 0.40),
            SOFT: Material(12_000.0, 0.40),
            SAT: Material(5_000.0, 0.40),
            SKIN: Material(30_000.0, 0.40),
        }

    def test_active_contact_adds_energy_and_residual(self):
        points = self.mesh.points
        slave = int(np.argmax(points[:, 0]))
        masters = np.argsort(points[:, 0])[:3]
        normal = np.asarray([[1.0, 0.0, 0.0]])
        initial_gap = points[slave, 0] - points[masters].mean(axis=0)[0]
        contact = ContactConstraints(
            [[slave, *masters]],
            [[1.0, -1 / 3, -1 / 3, -1 / 3]],
            normal,
            [initial_gap + 0.1],
            [1.0e4],
        )
        assembler = _Assembler(
            self.mesh,
            self.materials,
            Material(10_000.0, 0.4),
            dense_dof_limit=4000,
            contact=contact,
        )
        energy, residual, matrix, _, _ = assembler.assemble(
            np.zeros_like(points), np.ones(self.mesh.n_cells), True
        )
        self.assertGreater(energy, 0.0)
        self.assertGreater(np.linalg.norm(residual), 0.0)
        self.assertEqual(matrix.shape, (self.mesh.n_points * 3, self.mesh.n_points * 3))

    def test_outer_calibration_records_actual_target(self):
        result = calibrate_target_volume(
            self.mesh,
            self.mesh.cell_tags == SAT,
            self.mesh.nodes_for_tags([BONE]),
            1.06,
            materials=self.materials,
            options=SolverOptions(
                increments=3,
                max_iterations=20,
                relative_tolerance=1.0e-7,
                absolute_tolerance=1.0e-8,
                verbose=False,
            ),
            tolerance=5.0e-3,
            max_corrections=2,
        )
        self.assertEqual(result.desired_volume_ratio, 1.06)
        self.assertGreaterEqual(len(result.calibration_records), 1)
        self.assertLess(abs(result.actual_volume_ratio - 1.06) / 1.06, 0.02)

    def test_dynamic_contact_updates_gap_after_sliding(self):
        points = np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.2, 0.2, 0.1]]
        )
        contact = DynamicContact([3], [[0, 1, 2]], 1.0e4, 1.0)
        _, initial_gap = contact.evaluate(points)
        self.assertGreater(initial_gap[0], 0.0)
        moved = points.copy()
        moved[3] = [0.6, 0.2, -0.05]
        state, current_gap = contact.evaluate(moved)
        self.assertEqual(state.count, 1)
        self.assertLess(current_gap[0], 0.0)


if __name__ == "__main__":
    unittest.main()

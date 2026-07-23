import unittest
from unittest.mock import patch

import numpy as np

from satmorph.calibration import calibrate_target_volume
from satmorph.contact import ContactConstraints, DynamicContact
from satmorph.demo import BONE, SAT, SKIN, SOFT, layered_torso_mesh
from satmorph.solver import Material, MorphResult, SolverOptions, _Assembler


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

    def test_calibration_retries_failed_inner_solve_with_more_increments(self):
        cells = self.mesh.n_cells
        points = self.mesh.points
        successful = MorphResult(
            points=points.copy(),
            displacement=np.zeros_like(points),
            growth_lambda=np.ones(cells),
            j_total=np.ones(cells),
            j_elastic=np.ones(cells),
            target_reference_volume=1.0,
            target_current_volume=1.05,
            target_volume_ratio=1.05,
        )
        with patch(
            "satmorph.calibration.morph_target_region",
            side_effect=[
                RuntimeError("line search failed at increment 2, iteration 3"),
                successful,
            ],
        ) as mocked:
            result = calibrate_target_volume(
                self.mesh,
                self.mesh.cell_tags == SAT,
                self.mesh.nodes_for_tags([BONE]),
                1.05,
                materials=self.materials,
                options=SolverOptions(increments=2, verbose=False),
                max_corrections=0,
                max_solver_retries=2,
                increment_retry_factor=2,
            )
        self.assertEqual(mocked.call_count, 2)
        self.assertEqual(mocked.call_args_list[0].kwargs["options"].increments, 2)
        self.assertEqual(mocked.call_args_list[1].kwargs["options"].increments, 4)
        self.assertEqual(result.calibration_records[0]["solver_retries"], 1.0)
        self.assertEqual(result.calibration_records[0]["solver_increments"], 4.0)


if __name__ == "__main__":
    unittest.main()

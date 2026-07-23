import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.io import savemat

from satmorph.demo import BONE, SAT
from satmorph.io import load_mesh
from satmorph.voxel_convert import convert_voxel_mat


class VoxelConvertTests(unittest.TestCase):
    def test_two_voxels_become_conforming_tagged_tetrahedra(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "atlas.mat"
            output = root / "mesh.npz"
            surface = root / "surface.npz"
            labels = np.asarray([[[1]], [[68]]], dtype=np.uint8)
            savemat(
                source,
                {
                    "MaterialLabelGrid": labels,
                    "Axis0": np.asarray([0.0, 0.001, 0.002]),
                    "Axis1": np.asarray([0.0, 0.001]),
                    "Axis2": np.asarray([0.0, 0.001]),
                },
            )

            report = convert_voxel_mat(
                source,
                output,
                surface_output=surface,
                stride=1,
                surface_stride=1,
            )
            mesh = load_mesh(output)
            self.assertEqual(mesh.n_cells, 12)
            self.assertEqual(np.count_nonzero(mesh.cell_tags == SAT), 6)
            self.assertEqual(np.count_nonzero(mesh.cell_tags == BONE), 6)
            self.assertIn("source_label", mesh.cell_data)
            self.assertIn("mechanical_group_id", mesh.cell_data)
            self.assertIn("mesh_domain_tag", mesh.cell_data)
            self.assertEqual(np.count_nonzero(mesh.cell_data["source_label"] == 1), 6)
            self.assertEqual(np.count_nonzero(mesh.cell_data["source_label"] == 68), 6)
            self.assertIn("mesh_domains", report)
            self.assertIn("mesh_domain_policy", report)
            self.assertEqual(mesh.n_points, 12)
            self.assertEqual(report["surface"]["triangles"], 20)
            with np.load(surface) as data:
                self.assertEqual(data["triangles"].shape, (20, 3))
                self.assertAlmostEqual(float(data["points"][:, 0].max()), 0.002)


if __name__ == "__main__":
    unittest.main()

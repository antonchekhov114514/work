import tempfile
import unittest
import json
from pathlib import Path

import numpy as np
from scipy.io import savemat

from satmorph.adaptive_voxel import convert_voxel_mat_adaptive
from satmorph.io import load_mesh


class AdaptiveVoxelTests(unittest.TestCase):
    def test_boundary_blocks_receive_refined_conforming_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "atlas.mat"
            output = root / "adaptive.npz"
            labels = np.ones((4, 4, 4), dtype=np.uint8)
            labels[2:, :, :] = 68
            savemat(
                source,
                {
                    "MaterialLabelGrid": labels,
                    "Axis0": np.arange(5) * 0.001,
                    "Axis1": np.arange(5) * 0.001,
                    "Axis2": np.arange(5) * 0.001,
                },
            )
            report = convert_voxel_mat_adaptive(
                source,
                output,
                coarse_stride=4,
                refine_stride=2,
                refine_labels=[1, 68],
                refine_halo_blocks=0,
            )
            mesh = load_mesh(output)
            self.assertGreater(mesh.n_points, 8)
            self.assertGreater(mesh.n_cells, 6)
            self.assertEqual(set(np.unique(mesh.cell_data["source_label"])), {1, 68})
            self.assertIn("mesh_domain_tag", mesh.cell_data)
            self.assertIn("mesh_domains", report)
            self.assertEqual(report["refined_coarse_blocks"], 1)

            audit = root / "audit.json"
            audit.write_text(
                json.dumps(
                    {
                        "summary": {"missing_source_labels": [68]},
                        "labels": [{"label_id": 68, "mesh_vs_voxel_error_percent": -100.0}],
                    }
                ),
                encoding="utf-8",
            )
            fine_output = root / "adaptive-fine.npz"
            fine_report = convert_voxel_mat_adaptive(
                source,
                fine_output,
                coarse_stride=4,
                refine_stride=2,
                fine_stride=1,
                refine_labels=[1],
                audit_report=audit,
                refine_halo_blocks=0,
            )
            fine_mesh = load_mesh(fine_output)
            self.assertIn(68, fine_report["audit_selected_labels"])
            self.assertEqual(fine_report["fine_refined_coarse_blocks"], 1)
            self.assertGreater(fine_mesh.n_points, mesh.n_points)


if __name__ == "__main__":
    unittest.main()

import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.io import savemat

from satmorph.audit import build_volume_audit, tetra_quality
from satmorph.voxel_convert import convert_voxel_mat


class AuditTests(unittest.TestCase):
    def test_stride_one_preserves_per_label_voxel_volume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "atlas.mat"
            mesh = root / "mesh.npz"
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
            convert_voxel_mat(source, mesh, stride=1)
            report = build_volume_audit(source, mesh)
            rows = {row["label_id"]: row for row in report["labels"]}
            self.assertAlmostEqual(rows[1]["mesh_vs_voxel_error_percent"], 0.0, places=8)
            self.assertAlmostEqual(rows[68]["mesh_vs_voxel_error_percent"], 0.0, places=8)
            self.assertGreater(report["reference_mesh_quality"]["mean_ratio_quality_minimum"], 0.0)


if __name__ == "__main__":
    unittest.main()

import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.io import savemat

from satmorph.surface_map import SurfaceMesh
from satmorph.surface_ops import compute_point_normals, smooth_surface
from satmorph.visual_surface import extract_visual_surface_from_voxel_mat


class VisualSurfaceTests(unittest.TestCase):
    def test_normals_and_smoothing_preserve_surface_topology(self):
        surface = SurfaceMesh(
            points=np.asarray(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ),
            triangles=np.asarray(
                [
                    [0, 2, 1],
                    [0, 1, 3],
                    [1, 2, 3],
                    [2, 0, 3],
                ],
                dtype=np.int64,
            ),
        )
        normals = compute_point_normals(surface.points, surface.triangles)
        self.assertEqual(normals.shape, surface.points.shape)
        self.assertTrue(np.all(np.linalg.norm(normals, axis=1) > 0.0))

        smoothed = smooth_surface(surface, method="taubin", iterations=2)
        self.assertEqual(smoothed.triangles.shape, surface.triangles.shape)
        self.assertEqual(smoothed.points.shape, surface.points.shape)

    def test_extract_visual_surface_blocks_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "atlas.mat"
            output = root / "surface.npz"
            report = root / "surface.json"
            labels = np.ones((2, 2, 2), dtype=np.uint8)
            savemat(
                source,
                {
                    "MaterialLabelGrid": labels,
                    "Axis0": np.asarray([0.0, 0.001, 0.002]),
                    "Axis1": np.asarray([0.0, 0.001, 0.002]),
                    "Axis2": np.asarray([0.0, 0.001, 0.002]),
                },
            )

            summary = extract_visual_surface_from_voxel_mat(
                source,
                output,
                report_path=report,
                method="blocks",
                smooth_method="none",
                surface_stride=1,
            )
            self.assertEqual(summary["method"], "blocks")
            self.assertTrue(output.exists())
            self.assertTrue(report.exists())
            with np.load(output) as data:
                self.assertIn("normals", data.files)
                self.assertEqual(data["triangles"].shape[1], 3)


if __name__ == "__main__":
    unittest.main()

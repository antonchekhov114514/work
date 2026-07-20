import tempfile
import unittest
from pathlib import Path

import numpy as np

from satmorph.metrics import mapped_surface_metrics, sat_thickness_metrics


class MetricsTests(unittest.TestCase):
    def test_scaled_closed_surface_reports_volume_and_distance(self):
        points = np.asarray(
            [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1], [0, 0, 0]],
            dtype=float,
        )
        triangles = np.asarray(
            [[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4], [3, 7, 6], [3, 6, 2], [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5]]
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mapped.npz"
            np.savez_compressed(path, points=points, deformed_points=1.1 * points, triangles=triangles)
            report = mapped_surface_metrics(path, slice_count=9)
            self.assertAlmostEqual(report["reference"]["enclosed_volume"], 1.0)
            self.assertAlmostEqual(report["deformed"]["enclosed_volume"], 1.1**3)
            self.assertGreater(report["surface_distance"]["hausdorff"], 0.0)
            thickness = sat_thickness_metrics(path, path)
            self.assertAlmostEqual(thickness["mean_change"], 0.0)


if __name__ == "__main__":
    unittest.main()

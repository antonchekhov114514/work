import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.io import savemat

from satmorph.surface_map import load_surface
from satmorph.tissue_surface import extract_tissue_surface_bundle


class TissueSurfaceTests(unittest.TestCase):
    def test_bundle_writes_independent_vtp_and_vtm(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "atlas.mat"
            labels = np.zeros((3, 2, 2), dtype=np.uint8)
            labels[:2] = 1
            labels[2:] = 68
            savemat(source, {"MaterialLabelGrid": labels})
            output = root / "tissues"
            report = extract_tissue_surface_bundle(
                source,
                output,
                include_labels=[1, 68],
                method="blocks",
                surface_stride=1,
                suffix=".vtp",
            )
            self.assertEqual(len(report["surfaces"]), 2)
            self.assertTrue((output / "tissues.vtm").exists())
            for entry in report["surfaces"]:
                surface = load_surface(output / entry["file"])
                self.assertGreater(surface.n_triangles, 0)


if __name__ == "__main__":
    unittest.main()

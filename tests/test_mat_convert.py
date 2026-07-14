import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.io import savemat

from satmorph.io import load_mesh
from satmorph.mat_convert import convert_mat


class MatConvertTests(unittest.TestCase):
    def test_matlab_style_arrays_convert_to_required_npz(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = directory / "model.mat"
            output = directory / "model.npz"
            surface = directory / "surface.npz"
            savemat(
                source,
                {
                    "nodes": np.asarray(
                        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float
                    ).T,
                    "elements": np.asarray([[1], [2], [3], [4]], dtype=np.int64),
                    "materials": np.asarray([3], dtype=np.int64),
                    "skin_vertices": np.asarray(
                        [[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float
                    ),
                    "skin_faces": np.asarray([1, 2, 3], dtype=np.int64),
                },
            )
            report = convert_mat(source, output, surface_output=surface)
            mesh = load_mesh(output)
            self.assertEqual(mesh.points.shape, (4, 3))
            np.testing.assert_array_equal(mesh.tetra, [[0, 1, 2, 3]])
            np.testing.assert_array_equal(mesh.cell_tags, [3])
            self.assertTrue(report["mesh"]["matlab_one_based_indices_converted"])
            with np.load(surface) as data:
                np.testing.assert_array_equal(data["triangles"], [[0, 1, 2]])


if __name__ == "__main__":
    unittest.main()

import unittest

import numpy as np

from satmorph.mesh import TetMesh
from satmorph.surface_map import SurfaceMesh, map_points_to_tet_displacement, map_surface


class SurfaceMapTests(unittest.TestCase):
    def setUp(self):
        self.mesh = TetMesh(
            points=np.asarray(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ),
            tetra=np.asarray([[0, 1, 2, 3]], dtype=np.int64),
            cell_tags=np.asarray([1], dtype=np.int64),
        )
        matrix = np.asarray(
            [
                [0.10, 0.02, -0.01],
                [0.00, -0.03, 0.04],
                [0.02, 0.01, 0.05],
            ]
        )
        offset = np.asarray([0.01, -0.02, 0.03])
        self.affine = lambda x: x @ matrix.T + offset
        self.displacement = self.affine(self.mesh.points)

    def test_internal_points_reproduce_affine_displacement(self):
        query = np.asarray(
            [
                [0.25, 0.25, 0.25],
                [0.10, 0.20, 0.30],
                [0.70, 0.10, 0.10],
            ]
        )
        mapped = map_points_to_tet_displacement(self.mesh, self.displacement, query)
        np.testing.assert_allclose(mapped.displacement, self.affine(query), atol=1.0e-14)
        self.assertTrue(np.all(mapped.inside))
        np.testing.assert_allclose(mapped.barycentric.sum(axis=1), 1.0, atol=1.0e-14)

    def test_triangle_centers_are_mapped(self):
        surface = SurfaceMesh(
            points=np.asarray(
                [
                    [0.10, 0.10, 0.10],
                    [0.40, 0.10, 0.10],
                    [0.10, 0.40, 0.10],
                ]
            ),
            triangles=np.asarray([[0, 1, 2]], dtype=np.int64),
        )
        result = map_surface(self.mesh, self.displacement, surface, map_centers=True)
        centers = surface.triangle_centers()
        np.testing.assert_allclose(
            result.center_displacement, self.affine(centers), atol=1.0e-14
        )
        np.testing.assert_allclose(result.points, surface.points + self.affine(surface.points))
        self.assertEqual(result.summary()["mapped_triangle_centers"], 1)

    def test_outside_clamp_keeps_displacement_finite(self):
        query = np.asarray([[1.5, 0.2, 0.2]])
        mapped = map_points_to_tet_displacement(
            self.mesh, self.displacement, query, outside_mode="clamp"
        )
        self.assertFalse(bool(mapped.inside[0]))
        self.assertTrue(np.all(np.isfinite(mapped.displacement)))
        np.testing.assert_allclose(mapped.barycentric.sum(axis=1), 1.0, atol=1.0e-14)


if __name__ == "__main__":
    unittest.main()

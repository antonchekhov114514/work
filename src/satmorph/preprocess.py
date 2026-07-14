from __future__ import annotations

from pathlib import Path

import numpy as np


def repair_surface(input_path: str | Path, output_path: str | Path) -> None:
    """Repair a triangle surface with the paper-cited MeshFix implementation."""
    try:
        import meshio
        import pymeshfix
    except ImportError as exc:
        raise RuntimeError(
            "surface repair requires meshio and pymeshfix; install sat-morphing-fem[preprocess]"
        ) from exc

    source = meshio.read(input_path)
    triangles = [block.data for block in source.cells if block.type == "triangle"]
    if not triangles:
        raise ValueError("input surface contains no triangle cells")
    faces = np.vstack(triangles).astype(np.int64)
    fixer = pymeshfix.MeshFix(np.asarray(source.points[:, :3], dtype=float), faces)
    fixer.repair(verbose=False)
    meshio.write_points_cells(output_path, fixer.v, [("triangle", fixer.f)])


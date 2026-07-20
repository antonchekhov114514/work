from __future__ import annotations

from pathlib import Path
from typing import Iterable
import json

import numpy as np

from .surface_map import load_surface


_VIEWS = {
    "front": (0.0, -90.0),
    "side": (0.0, 0.0),
    "back": (0.0, 90.0),
    "oblique": (12.0, -55.0),
}


def render_surface_comparison(
    inputs: Iterable[str | Path],
    output: str | Path,
    *,
    views: tuple[str, ...] = ("front", "side", "oblique"),
    color_by: str = "displacement",
    dpi: int = 300,
    max_triangles: int = 45_000,
    include_reference: bool = True,
    labels: list[str] | None = None,
) -> dict[str, object]:
    """Render aligned surface cases with shared bounds, camera, and color limits."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib import colors
        from matplotlib.cm import ScalarMappable
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    except ImportError as exc:
        raise RuntimeError(
            "paper figures require matplotlib; install with python -m pip install -e '.[figure]'"
        ) from exc

    paths = [Path(path) for path in inputs]
    if not paths:
        raise ValueError("at least one mapped surface NPZ is required")
    invalid_views = sorted(set(views).difference(_VIEWS))
    if invalid_views:
        raise ValueError(f"unknown views: {invalid_views}")
    cases = [_load_case(path, color_by) for path in paths]
    case_labels = labels or [path.stem for path in paths]
    if len(case_labels) != len(cases):
        raise ValueError("labels must match the number of inputs")

    all_points = np.vstack([case["deformed_points"] for case in cases])
    bounds_min = all_points.min(axis=0)
    bounds_max = all_points.max(axis=0)
    center = 0.5 * (bounds_min + bounds_max)
    radius = 0.52 * float(np.max(bounds_max - bounds_min))
    values = np.concatenate([case["values"] for case in cases])
    value_min = float(np.nanmin(values)) if len(values) else 0.0
    value_max = float(np.nanmax(values)) if len(values) else 1.0
    if value_max <= value_min:
        value_max = value_min + 1.0
    normalization = colors.Normalize(value_min, value_max)
    cmap = plt.get_cmap("viridis")

    figure = plt.figure(
        figsize=(3.2 * len(views), 4.6 * len(cases)),
        facecolor="white",
        constrained_layout=True,
    )
    axes = []
    for row, (case, case_label) in enumerate(zip(cases, case_labels)):
        triangle_indices = _sample_triangles(len(case["triangles"]), max_triangles)
        triangles = case["triangles"][triangle_indices]
        face_values = case["values"][triangles].mean(axis=1)
        face_colors = cmap(normalization(face_values))
        for column, view in enumerate(views):
            axis = figure.add_subplot(len(cases), len(views), row * len(views) + column + 1, projection="3d")
            axes.append(axis)
            collection = Poly3DCollection(
                case["deformed_points"][triangles],
                facecolors=face_colors,
                edgecolors="none",
                linewidths=0.0,
                antialiased=False,
            )
            axis.add_collection3d(collection)
            if include_reference:
                reference_sample = triangles[::_max_step(len(triangles), 2500)]
                outline = Poly3DCollection(
                    case["points"][reference_sample],
                    facecolors=(0, 0, 0, 0),
                    edgecolors=(0.15, 0.15, 0.15, 0.12),
                    linewidths=0.18,
                )
                axis.add_collection3d(outline)
            axis.set_xlim(center[0] - radius, center[0] + radius)
            axis.set_ylim(center[1] - radius, center[1] + radius)
            axis.set_zlim(center[2] - radius, center[2] + radius)
            axis.set_box_aspect((1, 1, 1))
            axis.set_proj_type("ortho")
            axis.view_init(*_VIEWS[view])
            axis.set_axis_off()
            if row == 0:
                axis.set_title(view.capitalize(), fontsize=10, pad=2, color="#202124")
            if column == 0:
                axis.text2D(
                    0.02,
                    0.97,
                    case_label,
                    transform=axis.transAxes,
                    ha="left",
                    va="top",
                    fontsize=10,
                    fontweight="bold",
                    color="#202124",
                )
    scalar = ScalarMappable(norm=normalization, cmap=cmap)
    scalar.set_array([])
    colorbar = figure.colorbar(scalar, ax=axes, location="bottom", shrink=0.55, pad=0.015, aspect=40)
    colorbar.set_label(_colorbar_label(color_by), fontsize=9)
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return {
        "output": str(target),
        "cases": [str(path) for path in paths],
        "views": list(views),
        "shared_bounds_min": bounds_min.tolist(),
        "shared_bounds_max": bounds_max.tolist(),
        "color_by": color_by,
        "color_range": [value_min, value_max],
        "dpi": dpi,
        "reference_overlay": include_reference,
    }


def render_tissue_bundle(
    manifest: str | Path,
    output: str | Path,
    *,
    views: tuple[str, ...] = ("front", "side", "oblique"),
    dpi: int = 300,
    max_triangles_per_tissue: int = 20_000,
    include_labels: list[int] | None = None,
) -> dict[str, object]:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    except ImportError as exc:
        raise RuntimeError("tissue figures require matplotlib") from exc
    manifest_path = Path(manifest)
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    tissues = []
    for entry in raw.get("surfaces", []):
        if include_labels is not None and int(entry["label_id"]) not in include_labels:
            continue
        surface = load_surface(manifest_path.parent / entry["file"])
        tissues.append((entry, surface))
    if not tissues:
        raise ValueError("tissue manifest contains no surfaces")
    all_points = np.vstack([surface.points for _, surface in tissues])
    bounds_min, bounds_max = all_points.min(axis=0), all_points.max(axis=0)
    center = 0.5 * (bounds_min + bounds_max)
    radius = 0.52 * float(np.max(bounds_max - bounds_min))
    centers_by_label = {
        int(entry["label_id"]): surface.points.mean(axis=0)
        for entry, surface in tissues
    }
    invert_longitudinal = (
        11 in centers_by_label
        and 35 in centers_by_label
        and centers_by_label[11][2] < centers_by_label[35][2]
    )
    figure = plt.figure(
        figsize=(4.2 * len(views), 6.2), facecolor="white", constrained_layout=True
    )
    for column, view in enumerate(views):
        axis = figure.add_subplot(1, len(views), column + 1, projection="3d")
        for entry, surface in tissues:
            indices = _sample_triangles(surface.n_triangles, max_triangles_per_tissue)
            triangles = surface.triangles[indices]
            opacity = float(entry.get("opacity", 1.0))
            group = entry.get("mechanical_group")
            opacity_caps = {
                "SKIN": 0.08,
                "SAT_FAT": 0.14,
                "VISCERAL_FAT": 0.20,
                "MUSCLE": 0.32,
            }
            if group in opacity_caps:
                opacity = min(opacity, opacity_caps[group])
            collection = Poly3DCollection(
                surface.points[triangles],
                facecolors=entry.get("color", "#B0B0B0"),
                edgecolors="none",
                linewidths=0.0,
                alpha=opacity,
                antialiased=False,
            )
            axis.add_collection3d(collection)
        axis.set_xlim(center[0] - radius, center[0] + radius)
        axis.set_ylim(center[1] - radius, center[1] + radius)
        axis.set_zlim(center[2] - radius, center[2] + radius)
        if invert_longitudinal:
            axis.set_zlim(center[2] + radius, center[2] - radius)
        axis.set_box_aspect((1, 1, 1))
        axis.set_proj_type("ortho")
        axis.view_init(*_VIEWS[view])
        axis.set_axis_off()
        axis.set_title(view.capitalize(), fontsize=11, color="#202124")
    legend = [
        Patch(
            facecolor=entry.get("color", "#B0B0B0"),
            label=f"{entry['label_id']}: {entry['tissue']}",
        )
        for entry, _ in tissues
    ]
    figure.legend(
        handles=legend,
        loc="lower center",
        ncol=min(4, len(legend)),
        frameon=False,
        fontsize=8,
    )
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return {
        "manifest": str(manifest_path),
        "output": str(target),
        "tissue_count": len(tissues),
        "included_labels": None if include_labels is None else include_labels,
        "views": list(views),
        "dpi": dpi,
        "bounds_min": bounds_min.tolist(),
        "bounds_max": bounds_max.tolist(),
        "longitudinal_axis_inverted": bool(invert_longitudinal),
    }


def _load_case(path: Path, color_by: str) -> dict[str, np.ndarray]:
    if path.suffix.lower() != ".npz":
        raise ValueError("paper-figure currently reads mapped surface NPZ files")
    with np.load(path, allow_pickle=False) as data:
        points = np.asarray(data["points"], dtype=float)
        deformed = np.asarray(data.get("deformed_points", points), dtype=float)
        triangles = np.asarray(data["triangles"], dtype=np.int64)
        if color_by == "displacement":
            displacement = np.asarray(data.get("displacement", deformed - points), dtype=float)
            values = np.linalg.norm(displacement, axis=1)
        else:
            key = color_by if color_by in data else f"mapped_{color_by}"
            if key not in data:
                raise KeyError(f"{path} has no point array {color_by!r}")
            values = np.asarray(data[key], dtype=float).reshape(-1)
    if len(values) != len(points):
        raise ValueError(f"{path}: color array must have one value per surface point")
    return {"points": points, "deformed_points": deformed, "triangles": triangles, "values": values}


def _sample_triangles(count: int, maximum: int) -> np.ndarray:
    if count <= maximum:
        return np.arange(count, dtype=np.int64)
    return np.linspace(0, count - 1, maximum, dtype=np.int64)


def _max_step(count: int, target: int) -> int:
    return max(1, int(np.ceil(count / max(target, 1))))


def _colorbar_label(color_by: str) -> str:
    return "Displacement magnitude (model units)" if color_by == "displacement" else color_by

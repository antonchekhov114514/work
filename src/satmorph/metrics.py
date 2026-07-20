from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def mapped_surface_metrics(
    path: str | Path,
    *,
    longitudinal_axis: int = 2,
    slice_count: int = 31,
    slice_range: tuple[float, float] = (0.35, 0.65),
) -> dict[str, object]:
    reference, deformed, triangles = _load_mapped_surface(path)
    displacement = deformed - reference
    reference_profile = circumference_profile(
        reference, triangles, longitudinal_axis, slice_count, slice_range
    )
    deformed_profile = circumference_profile(
        deformed, triangles, longitudinal_axis, slice_count, slice_range
    )
    reference_waist = _minimum_valid_profile(reference_profile)
    deformed_waist = _minimum_valid_profile(deformed_profile)
    forward = cKDTree(reference).query(deformed, k=1)[0]
    reverse = cKDTree(deformed).query(reference, k=1)[0]
    magnitude = np.linalg.norm(displacement, axis=1)
    return {
        "input": str(Path(path)),
        "longitudinal_axis": longitudinal_axis,
        "reference": _surface_geometry(reference, triangles),
        "deformed": _surface_geometry(deformed, triangles),
        "displacement": _distribution(magnitude),
        "surface_distance": {
            "hausdorff": float(max(forward.max(initial=0.0), reverse.max(initial=0.0))),
            "symmetric_chamfer_mean": float(0.5 * (forward.mean() + reverse.mean())),
            "deformed_to_reference": _distribution(forward),
            "reference_to_deformed": _distribution(reverse),
        },
        "waist": {
            "reference": reference_waist,
            "deformed": deformed_waist,
            "change": deformed_waist["circumference"] - reference_waist["circumference"],
            "change_percent": (
                100.0
                * (deformed_waist["circumference"] - reference_waist["circumference"])
                / reference_waist["circumference"]
                if reference_waist["circumference"] > 0.0
                else 0.0
            ),
        },
        "circumference_profile": {
            "reference": reference_profile,
            "deformed": deformed_profile,
        },
    }


def sat_thickness_metrics(
    outer_path: str | Path,
    inner_path: str | Path,
) -> dict[str, object]:
    outer_ref, outer_def, _ = _load_mapped_surface(outer_path)
    inner_ref, inner_def, _ = _load_mapped_surface(inner_path)
    reference = cKDTree(inner_ref).query(outer_ref, k=1)[0]
    deformed = cKDTree(inner_def).query(outer_def, k=1)[0]
    return {
        "outer": str(Path(outer_path)),
        "inner": str(Path(inner_path)),
        "method": "nearest-surface Euclidean distance",
        "reference_thickness": _distribution(reference),
        "deformed_thickness": _distribution(deformed),
        "mean_change": float(deformed.mean() - reference.mean()),
        "mean_change_percent": (
            100.0 * (deformed.mean() - reference.mean()) / reference.mean()
            if reference.mean() > 0.0
            else 0.0
        ),
    }


def circumference_profile(
    points: np.ndarray,
    triangles: np.ndarray,
    axis: int,
    count: int,
    fraction_range: tuple[float, float],
) -> list[dict[str, float]]:
    if axis not in (0, 1, 2) or count < 2:
        raise ValueError("axis must be 0/1/2 and slice_count must be at least two")
    lower, upper = fraction_range
    if not 0.0 <= lower < upper <= 1.0:
        raise ValueError("slice_range must lie within [0, 1]")
    minimum, maximum = float(points[:, axis].min()), float(points[:, axis].max())
    levels = minimum + np.linspace(lower, upper, count) * (maximum - minimum)
    return [
        {"coordinate": float(level), "circumference": _slice_length(points, triangles, axis, level)}
        for level in levels
    ]


def write_metrics(
    report: dict[str, object],
    output: str | Path,
    *,
    profile_csv: str | Path | None = None,
) -> None:
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if profile_csv is None or "circumference_profile" not in report:
        return
    reference = report["circumference_profile"]["reference"]
    deformed = report["circumference_profile"]["deformed"]
    rows = [
        {
            "reference_coordinate": left["coordinate"],
            "reference_circumference": left["circumference"],
            "deformed_coordinate": right["coordinate"],
            "deformed_circumference": right["circumference"],
        }
        for left, right in zip(reference, deformed)
    ]
    csv_target = Path(profile_csv)
    csv_target.parent.mkdir(parents=True, exist_ok=True)
    with csv_target.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _surface_geometry(points: np.ndarray, triangles: np.ndarray) -> dict[str, object]:
    tri = points[triangles]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = 0.5 * np.linalg.norm(cross, axis=1).sum()
    signed_volume = np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6.0
    bounds_min, bounds_max = points.min(axis=0), points.max(axis=0)
    return {
        "surface_area": float(area),
        "enclosed_volume": float(abs(signed_volume)),
        "bounds_min": bounds_min.tolist(),
        "bounds_max": bounds_max.tolist(),
        "dimensions": (bounds_max - bounds_min).tolist(),
    }


def _slice_length(points, triangles, axis, level):
    total = 0.0
    for nodes in triangles:
        triangle = points[nodes]
        intersections: list[np.ndarray] = []
        for left, right in ((0, 1), (1, 2), (2, 0)):
            dl = triangle[left, axis] - level
            dr = triangle[right, axis] - level
            if dl * dr < 0.0:
                fraction = dl / (dl - dr)
                intersections.append(triangle[left] + fraction * (triangle[right] - triangle[left]))
        if len(intersections) == 2:
            total += float(np.linalg.norm(intersections[1] - intersections[0]))
    return total


def _minimum_valid_profile(profile):
    valid = [row for row in profile if row["circumference"] > 0.0]
    return min(valid, key=lambda row: row["circumference"]) if valid else {"coordinate": 0.0, "circumference": 0.0}


def _distribution(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    return {
        "minimum": float(values.min()) if len(values) else 0.0,
        "mean": float(values.mean()) if len(values) else 0.0,
        "median": float(np.median(values)) if len(values) else 0.0,
        "p05": float(np.percentile(values, 5)) if len(values) else 0.0,
        "p95": float(np.percentile(values, 95)) if len(values) else 0.0,
        "maximum": float(values.max()) if len(values) else 0.0,
    }


def _load_mapped_surface(path: str | Path):
    with np.load(path, allow_pickle=False) as data:
        points = np.asarray(data["points"], dtype=float)
        deformed = np.asarray(data.get("deformed_points", points), dtype=float)
        triangles = np.asarray(data["triangles"], dtype=np.int64)
    return points, deformed, triangles

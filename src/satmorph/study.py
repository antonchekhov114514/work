from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable


def summarize_result_jsons(paths: Iterable[str | Path]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in paths:
        source = Path(path)
        data = json.loads(source.read_text(encoding="utf-8"))
        row: dict[str, object] = {
            "path": str(source),
            "case": source.stem,
            "target_name": data.get("target_name", "SAT"),
            "target_volume_ratio": data.get("target_volume_ratio_unconstrained"),
            "desired_target_volume_ratio": data.get(
                "desired_target_volume_ratio",
                data.get("target_volume_ratio_unconstrained"),
            ),
            "actual_target_volume_ratio": data.get(
                "actual_target_volume_ratio",
                data.get("actual_sat_volume_ratio"),
            ),
            "target_volume_error_percent": data.get("target_volume_error_percent"),
            "minimum_total_jacobian": data.get("minimum_total_jacobian"),
            "maximum_displacement": data.get("maximum_displacement"),
            "calibration_solve_count": len(data.get("calibration_iterations", [])),
        }
        for key in ("constraint_count", "active_constraint_count", "maximum_penetration"):
            if key in data.get("contact", {}):
                row[f"contact_{key}"] = data["contact"][key]
        for label, values in data.get("volume_by_source_label", {}).items():
            prefix = f"source_label_{label}"
            row[f"{prefix}_volume_ratio"] = values.get("volume_ratio")
            row[f"{prefix}_volume_change_percent"] = values.get("volume_change_percent")
        for region, values in data.get("volume_by_region", {}).items():
            prefix = f"region_{region}"
            row[f"{prefix}_volume_ratio"] = values.get("volume_ratio")
            row[f"{prefix}_volume_change_percent"] = values.get("volume_change_percent")
        rows.append(row)
    return rows


def write_summary_csv(path: str | Path, rows: list[dict[str, object]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for preferred in (
        "case",
        "target_name",
        "target_volume_ratio",
        "desired_target_volume_ratio",
        "actual_target_volume_ratio",
        "target_volume_error_percent",
        "minimum_total_jacobian",
        "maximum_displacement",
        "calibration_solve_count",
        "path",
    ):
        if any(preferred in row for row in rows):
            fieldnames.append(preferred)
    extras = sorted({key for row in rows for key in row if key not in fieldnames})
    fieldnames.extend(extras)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

import csv
import json
import tempfile
import unittest
from pathlib import Path

from satmorph.study import summarize_result_jsons, write_summary_csv


class StudyTests(unittest.TestCase):
    def test_summarize_result_jsons_keeps_label_and_region_volumes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "case-120.json"
            path.write_text(
                json.dumps(
                    {
                        "target_name": "source_label:1",
                        "target_volume_ratio_unconstrained": 1.2,
                        "actual_target_volume_ratio": 1.18,
                        "target_volume_error_percent": -1.6667,
                        "minimum_total_jacobian": 0.91,
                        "maximum_displacement": 0.004,
                        "volume_by_source_label": {
                            "1": {"volume_ratio": 1.18, "volume_change_percent": 18.0}
                        },
                        "volume_by_region": {
                            "SAT": {"volume_ratio": 1.18, "volume_change_percent": 18.0}
                        },
                    }
                ),
                encoding="utf-8",
            )

            rows = summarize_result_jsons([path])
            self.assertEqual(rows[0]["case"], "case-120")
            self.assertEqual(rows[0]["target_name"], "source_label:1")
            self.assertEqual(rows[0]["source_label_1_volume_ratio"], 1.18)
            self.assertEqual(rows[0]["region_SAT_volume_change_percent"], 18.0)
            self.assertEqual(rows[0]["desired_target_volume_ratio"], 1.2)

            csv_path = root / "summary.csv"
            write_summary_csv(csv_path, rows)
            with csv_path.open(encoding="utf-8", newline="") as handle:
                table = list(csv.DictReader(handle))
            self.assertEqual(table[0]["case"], "case-120")
            self.assertIn("source_label_1_volume_ratio", table[0])


if __name__ == "__main__":
    unittest.main()

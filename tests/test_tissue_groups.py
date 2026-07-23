import unittest

import numpy as np

from satmorph.demo import BONE, SAT, SOFT
from satmorph.tissue_groups import (
    MECHANICAL_GROUPS,
    MESH_DOMAINS,
    labels_for_role,
    labels_to_mechanical_group_ids,
    labels_to_mesh_domain_ids,
    labels_to_solver_roles,
    mesh_domain_for_label,
    mesh_domain_report,
)


class TissueGroupTests(unittest.TestCase):
    def test_sat_and_fat_are_kept_separate(self):
        labels = np.asarray([1, 3, 70], dtype=np.int16)
        roles = labels_to_solver_roles(labels)
        groups = labels_to_mechanical_group_ids(labels)

        self.assertEqual(int(roles[0]), SAT)
        self.assertEqual(int(roles[1]), SOFT)
        self.assertEqual(int(roles[2]), SOFT)
        self.assertEqual(int(groups[0]), MECHANICAL_GROUPS["SAT_FAT"])
        self.assertEqual(int(groups[1]), MECHANICAL_GROUPS["VISCERAL_FAT"])
        self.assertEqual(int(groups[2]), MECHANICAL_GROUPS["MARROW_YELLOW"])

    def test_yellow_marrow_is_not_a_fixed_bone_anchor(self):
        self.assertEqual(labels_for_role("BONE"), [19, 68, 69])
        labels = np.asarray([19, 68, 69, 70], dtype=np.int16)
        roles = labels_to_solver_roles(labels)
        np.testing.assert_array_equal(roles, np.asarray([BONE, BONE, BONE, SOFT]))

    def test_mesh_domains_keep_key_interfaces_distinct(self):
        labels = np.asarray([1, 2, 3, 24, 50, 55, 68, 69, 70], dtype=np.int16)
        domains = labels_to_mesh_domain_ids(labels)

        np.testing.assert_array_equal(
            domains,
            np.asarray(
                [
                    MESH_DOMAINS["SAT"],
                    MESH_DOMAINS["SKIN"],
                    MESH_DOMAINS["ADIPOSE_OTHER"],
                    MESH_DOMAINS["CSF"],
                    MESH_DOMAINS["NERVE"],
                    MESH_DOMAINS["HEART_MUSCLE"],
                    MESH_DOMAINS["CANCELLOUS_BONE"],
                    MESH_DOMAINS["CORTICAL_BONE"],
                    MESH_DOMAINS["YELLOW_MARROW"],
                ],
                dtype=np.int16,
            ),
        )
        self.assertEqual(mesh_domain_for_label(71), "CONNECTIVE_CARTILAGE")

    def test_mesh_domain_report_covers_all_domains(self):
        report = mesh_domain_report()
        names = {row["mesh_domain"] for row in report}

        self.assertEqual(len(report), len(MESH_DOMAINS))
        self.assertIn("SAT", names)
        sat_row = next(row for row in report if row["mesh_domain"] == "SAT")
        self.assertEqual(sat_row["source_labels"], [1])
        self.assertEqual(sat_row["interface_policy"], "protected")


if __name__ == "__main__":
    unittest.main()

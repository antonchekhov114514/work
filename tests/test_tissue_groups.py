import unittest

import numpy as np

from satmorph.demo import BONE, SAT, SOFT
from satmorph.tissue_groups import (
    MECHANICAL_GROUPS,
    labels_for_role,
    labels_to_mechanical_group_ids,
    labels_to_solver_roles,
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


if __name__ == "__main__":
    unittest.main()

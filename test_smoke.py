"""M6: check the M1 gap closure, M2 integer Chern numbers, and M5 edge counts."""

import unittest

import numpy as np

from haldane import CRITICAL_MASS, T2
from m2_chern import GRID_SIZE, band_chern_numbers
from m5_edge import edge_branches, projected_bulk, ribbon_spectrum


def count_gap_branches(mass):
    # A multiple of six includes both projected valleys and the crossing at π.
    q = np.linspace(0, 2 * np.pi, 121)
    bulk_min, bulk_max = projected_bulk(q, mass)
    gap = bulk_max[:, 0].max(), bulk_min[:, 1].min()
    frequencies, weights, velocities = ribbon_spectrum(q, mass, gap)
    return sum(len(branch[0]) > 0 for branch in edge_branches(frequencies, weights, velocities, gap))


class PhysicsSmokeTests(unittest.TestCase):
    def test_critical_mass(self):
        self.assertAlmostEqual(CRITICAL_MASS, 3 * np.sqrt(3) * T2, places=12)

    def test_integer_chern_numbers(self):
        for mass in (0, 1):
            with self.subTest(mass=mass):
                chern = np.asarray(band_chern_numbers(mass, GRID_SIZE))
                self.assertLess(np.max(np.abs(chern - np.rint(chern))), 1e-6)

    def test_topological_ribbon_has_two_gap_branches(self):
        self.assertEqual(count_gap_branches(0), 2)

    def test_trivial_ribbon_has_no_gap_branch(self):
        self.assertEqual(count_gap_branches(1), 0)


if __name__ == "__main__":
    unittest.main()

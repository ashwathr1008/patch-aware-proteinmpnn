"""Check the SAP surrogate's geometry, accessibility algorithm and identity independence."""
import math
import unittest

import numpy as np

import sap


def ideal_residue(offset=(0.0, 0.0, 0.0)):
    """One residue with roughly ideal backbone geometry, translated by offset."""
    base = np.array([
        [-0.525, 1.363, 0.000],   # N
        [0.000, 0.000, 0.000],    # CA
        [1.526, 0.000, 0.000],    # C
        [2.130, 1.070, 0.000],    # O
    ])
    return base + np.asarray(offset, dtype="float64")


def chain(count, spacing=3.8):
    """A straight chain of residues spaced along x, far enough apart to stay exposed."""
    return np.stack([ideal_residue((index * spacing, 0.0, 0.0)) for index in range(count)])


class SpherePointsTest(unittest.TestCase):
    def test_points_lie_on_the_unit_sphere(self):
        points = sap.sphere_points(200)
        self.assertEqual(points.shape, (200, 3))
        np.testing.assert_allclose(np.linalg.norm(points, axis=1), 1.0, atol=1e-12)

    def test_points_are_spread_rather_than_clustered(self):
        points = sap.sphere_points(500)
        np.testing.assert_allclose(points.mean(axis=0), 0.0, atol=0.05)

    def test_a_nonpositive_count_is_rejected(self):
        with self.assertRaises(ValueError):
            sap.sphere_points(0)


class AtomAccessibilityTest(unittest.TestCase):
    def setUp(self):
        self.points = sap.sphere_points(2000)

    def test_an_isolated_atom_has_its_full_analytic_area(self):
        radius = sap.ATOM_RADII["C"] + sap.PROBE_RADIUS
        area = sap.atom_sasa(np.zeros((1, 3)), [sap.ATOM_RADII["C"]], self.points)
        self.assertAlmostEqual(area[0], 4 * math.pi * radius ** 2, places=6)

    def test_distant_atoms_do_not_occlude_each_other(self):
        coordinates = np.array([[0.0, 0.0, 0.0], [50.0, 0.0, 0.0]])
        areas = sap.atom_sasa(coordinates, [1.7, 1.7], self.points)
        self.assertAlmostEqual(areas[0], areas[1], places=9)
        self.assertAlmostEqual(areas[0], 4 * math.pi * (1.7 + sap.PROBE_RADIUS) ** 2, places=6)

    def test_a_fully_enclosed_atom_loses_its_area(self):
        # A small atom at the centre of a large one is completely occluded.
        coordinates = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        areas = sap.atom_sasa(coordinates, [0.5, 3.0], self.points)
        self.assertAlmostEqual(areas[0], 0.0, places=9)

    def test_touching_atoms_lose_part_of_their_area(self):
        coordinates = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        areas = sap.atom_sasa(coordinates, [1.7, 1.7], self.points)
        full = 4 * math.pi * (1.7 + sap.PROBE_RADIUS) ** 2
        self.assertLess(areas[0], full)
        self.assertGreater(areas[0], 0.5 * full)

    def test_mismatched_coordinate_and_radius_counts_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "counts differ"):
            sap.atom_sasa(np.zeros((2, 3)), [1.7], self.points)


class VirtualCarbonBetaTest(unittest.TestCase):
    def test_the_bond_length_is_physically_reasonable(self):
        backbone = chain(1)
        distance = np.linalg.norm(sap.virtual_cb(backbone)[0] - backbone[0, 1])
        self.assertAlmostEqual(distance, 1.53, delta=0.05)

    def test_the_construction_is_translation_equivariant(self):
        shift = np.array([5.0, -2.0, 3.0])
        first = sap.virtual_cb(chain(1))[0]
        second = sap.virtual_cb(chain(1) + shift)[0]
        np.testing.assert_allclose(second - first, shift, atol=1e-9)

    def test_it_does_not_depend_on_residue_identity(self):
        """There is no identity argument at all; this pins that property down."""
        backbone = chain(3)
        np.testing.assert_allclose(sap.virtual_cb(backbone), sap.virtual_cb(backbone.copy()))

    def test_a_wrong_shape_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            sap.virtual_cb(np.zeros((5, 3, 3)))

    def test_nonfinite_coordinates_are_rejected(self):
        backbone = chain(2)
        backbone[0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            sap.virtual_cb(backbone)


class BurialProfileTest(unittest.TestCase):
    def test_relative_accessibility_stays_within_bounds(self):
        burial = sap.burial_profile(chain(4), sphere_point_count=200)
        self.assertEqual(burial.shape, (4,))
        self.assertTrue(np.all(burial >= 0.0) and np.all(burial <= 1.0))

    def test_an_interior_residue_is_less_exposed_than_a_terminal_one(self):
        burial = sap.burial_profile(chain(5), sphere_point_count=400)
        self.assertLess(burial[2], burial[0])
        self.assertLess(burial[2], burial[-1])

    def test_a_crowded_residue_is_more_buried_than_an_isolated_one(self):
        spread = sap.burial_profile(chain(3, spacing=12.0), sphere_point_count=400)
        packed = sap.burial_profile(chain(3, spacing=4.2), sphere_point_count=400)
        self.assertLess(packed[1], spread[1])


class HydrophobicityTest(unittest.TestCase):
    def test_glycine_sits_at_zero(self):
        self.assertAlmostEqual(sap.HYDROPHOBICITY["G"], 0.0)

    def test_the_scale_has_the_expected_polarity(self):
        self.assertGreater(sap.HYDROPHOBICITY["F"], 0.0)
        self.assertGreater(sap.HYDROPHOBICITY["I"], sap.HYDROPHOBICITY["A"])
        self.assertLess(sap.HYDROPHOBICITY["R"], 0.0)
        self.assertLess(sap.HYDROPHOBICITY["D"], 0.0)

    def test_all_twenty_canonical_residues_are_covered(self):
        self.assertEqual(len(sap.HYDROPHOBICITY), 20)

    def test_noncanonical_residues_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "noncanonical"):
            sap.hydrophobicity_vector("AXL")


class DescriptorTest(unittest.TestCase):
    def setUp(self):
        self.backbone = chain(6, spacing=4.2)
        self.burial = sap.burial_profile(self.backbone, sphere_point_count=400)
        self.window = sap.window_matrix(self.backbone, radius=10.0)

    def test_the_window_is_symmetric_and_includes_each_residue(self):
        self.assertTrue(np.array_equal(self.window, self.window.T))
        self.assertTrue(np.all(np.diag(self.window)))

    def test_a_composition_matched_permutation_preserves_total_hydrophobicity(self):
        first = sap.hydrophobicity_vector("FILKRD").sum()
        second = sap.hydrophobicity_vector("KRDFIL").sum()
        self.assertAlmostEqual(first, second)

    def test_a_permutation_changes_the_score_when_burial_varies(self):
        """Composition is identical, so any change comes from the arrangement."""
        buried_first = sap.score(self.burial, "FILKRD", self.window)
        exposed_first = sap.score(self.burial, "KRDFIL", self.window)
        self.assertNotAlmostEqual(buried_first, exposed_first)

    def test_a_permutation_leaves_the_simple_score_alone_under_uniform_burial(self):
        uniform = np.full(6, 0.5)
        self.assertAlmostEqual(sap.score(uniform, "FILKRD"), sap.score(uniform, "KRDFIL"))

    def test_putting_hydrophobics_at_the_most_exposed_positions_raises_the_score(self):
        order = np.argsort(self.burial)
        sequence = ["G"] * 6
        for rank, position in enumerate(order):
            sequence[position] = "R" if rank < 3 else "F"
        exposed_hydrophobic = sap.score(self.burial, "".join(sequence))
        flipped = list(sequence)
        for rank, position in enumerate(order):
            flipped[position] = "F" if rank < 3 else "R"
        self.assertGreater(exposed_hydrophobic, sap.score(self.burial, "".join(flipped)))

    def test_patch_descriptors_cover_only_the_selected_positions(self):
        result = sap.descriptors(self.burial, "FILKRD", self.window, positions=[0, 1])
        partial = sap.sap_profile(self.burial, "FILKRD")[:2].sum()
        self.assertAlmostEqual(result["sap_simple_patch"], partial)
        self.assertIn("burial_patch_mean", result)

    def test_an_out_of_range_patch_position_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "out of range"):
            sap.descriptors(self.burial, "FILKRD", self.window, positions=[0, 99])

    def test_a_length_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "lengths differ"):
            sap.sap_profile(self.burial, "FIL")


if __name__ == "__main__":
    unittest.main()

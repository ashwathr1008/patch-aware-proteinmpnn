"""Check variant-name parsing, the duplicate policy, and contrast formation.

This module had no tests, which is how two defects survived in it. One silently overwrote
repeated measurements of the same variant, keeping whichever row the reader reached last. The
other, introduced while fixing the first, split variant names on the wrong delimiter and turned
every domain into a mutation string. Both are regression-tested here.
"""
import csv
import math
from pathlib import Path
import tempfile
import unittest

import extract_swap_contrasts as extract


def variants_file(rows, directory, name_column="name_original"):
    path = Path(directory) / "variants.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[name_column, "deltaG", "deltaG_95CI"])
        writer.writeheader()
        for name, stability, interval in rows:
            writer.writerow({name_column: name, "deltaG": stability,
                             "deltaG_95CI": interval})
    return path


class TestDomainOf(unittest.TestCase):
    """A pipe means two different things, and the order of splitting decides correctness."""

    def test_pipe_separating_two_mutations_is_not_a_chain_prefix(self):
        self.assertEqual(
            extract.domain_of("1GL5.pdb_dmutv5_10F:53Y_F10Q|Y53Q"), "1GL5")

    def test_pipe_before_the_pdb_suffix_is_a_chain_prefix(self):
        self.assertEqual(extract.domain_of("EA|run2_0325_0005.pdb"), "run2_0325_0005")
        self.assertEqual(extract.domain_of("EA|run2_0325_0005.pdb_D1Q"), "run2_0325_0005")

    def test_plain_name(self):
        self.assertEqual(extract.domain_of("1A0N.pdb_hnet0"), "1A0N")


class TestConsistency(unittest.TestCase):
    """Replicates agree exactly when their difference would not be called resolved."""

    def test_agreement_within_combined_error(self):
        # Half-widths 0.05 each, combined 0.0707, so anything under 0.1414 agrees.
        self.assertTrue(extract.consistent([(1.00, 0.1), (1.10, 0.1)]))

    def test_disagreement_beyond_combined_error(self):
        self.assertFalse(extract.consistent([(1.00, 0.1), (1.50, 0.1)]))

    def test_boundary_is_inclusive(self):
        combined = math.sqrt(0.05 ** 2 + 0.05 ** 2)
        self.assertTrue(extract.consistent([(0.0, 0.1), (2 * combined, 0.1)]))
        self.assertFalse(extract.consistent([(0.0, 0.1), (2 * combined + 1e-9, 0.1)]))

    def test_wide_errors_tolerate_wide_disagreement(self):
        self.assertTrue(extract.consistent([(1.0, 2.0), (2.0, 2.0)]))

    def test_every_pair_must_agree(self):
        """Three replicates where the extremes disagree, though neighbours do not."""
        self.assertFalse(extract.consistent([(0.0, 0.1), (0.13, 0.1), (0.30, 0.1)]))

    def test_matches_the_resolved_rule(self):
        """Consistency is the negation of the contrast filter, by construction."""
        a, b = (1.0, 0.1), (1.5, 0.1)
        uncertainty = math.sqrt(a[1] ** 2 + b[1] ** 2) / 2
        resolved = abs(a[0] - b[0]) > 2 * uncertainty
        self.assertEqual(extract.consistent([a, b]), not resolved)


class TestCombine(unittest.TestCase):

    def test_mean_and_narrowed_error(self):
        stability, width = extract.combine([(1.0, 0.2), (1.2, 0.2)])
        self.assertAlmostEqual(stability, 1.1)
        # full width of the mean = sqrt(sum i^2) / N
        self.assertAlmostEqual(width, math.sqrt(0.2 ** 2 + 0.2 ** 2) / 2)
        self.assertLess(width, 0.2, "averaging must reduce the error, not inherit it")

    def test_single_value_is_unchanged(self):
        self.assertEqual(extract.combine([(1.0, 0.2)]), (1.0, 0.2))


class TestDuplicatePolicy(unittest.TestCase):
    """One site pair, both arrangements, with the forward arrangement measured twice."""

    def rows(self, second_forward):
        return [("X.pdb_dm_A10C|D20E", 1.0, 0.1),
                ("X.pdb_dm_A10C|D20E", second_forward, 0.1),
                ("X.pdb_dm_A10E|D20C", 0.0, 0.1)]

    def parse(self, rows, policy):
        with tempfile.TemporaryDirectory() as directory:
            path = variants_file(rows, directory)
            table, _, _, conflicts, merged = extract.parse_variants(
                path, "deltaG", "deltaG_95CI", policy)
            return table, conflicts, merged

    def test_consistent_replicates_are_averaged(self):
        table, conflicts, merged = self.parse(self.rows(1.10), "average-if-consistent")
        self.assertEqual(merged, 1)
        self.assertEqual(conflicts, [])
        self.assertAlmostEqual(table[("X", 10, 20)][("C", "E")][0], 1.05)

    def test_inconsistent_replicates_are_dropped_even_when_averaging(self):
        table, conflicts, merged = self.parse(self.rows(9.9), "average-if-consistent")
        self.assertEqual(merged, 0)
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0]["resolution"], "dropped")
        self.assertNotIn(("C", "E"), table[("X", 10, 20)])

    def test_drop_policy_never_averages(self):
        table, conflicts, merged = self.parse(self.rows(1.10), "drop")
        self.assertEqual(merged, 0)
        self.assertEqual(len(conflicts), 1)
        self.assertNotIn(("C", "E"), table[("X", 10, 20)])

    def test_exact_repeats_are_collapsed_not_flagged(self):
        table, conflicts, merged = self.parse(self.rows(1.0), "average-if-consistent")
        self.assertEqual((merged, conflicts), (0, []))
        self.assertAlmostEqual(table[("X", 10, 20)][("C", "E")][0], 1.0)

    def test_a_dropped_arrangement_removes_the_contrast(self):
        """The contrast needs both arrangements, so losing one must lose the pair."""
        table, _, _ = self.parse(self.rows(9.9), "average-if-consistent")
        self.assertEqual(extract.swap_contrasts(table), [])

    def test_averaging_keeps_the_contrast(self):
        table, _, _ = self.parse(self.rows(1.10), "average-if-consistent")
        contrasts = extract.swap_contrasts(table)
        self.assertEqual(len(contrasts), 1)
        self.assertAlmostEqual(contrasts[0]["difference"], 1.05)


class TestNameColumn(unittest.TestCase):

    def test_either_name_column_is_accepted(self):
        for column in ("name_original", "name"):
            with tempfile.TemporaryDirectory() as directory:
                path = variants_file([("X.pdb_dm_A10C|D20E", 1.0, 0.1)], directory, column)
                table, rows, _, _, _ = extract.parse_variants(
                    path, "deltaG", "deltaG_95CI", "drop")
                self.assertEqual(rows, 1)
                self.assertIn(("X", 10, 20), table)

    def test_a_missing_name_column_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.csv"
            path.write_text("other,deltaG,deltaG_95CI\nx,1,0.1\n", encoding="utf-8")
            with self.assertRaises(SystemExit):
                extract.parse_variants(path, "deltaG", "deltaG_95CI", "drop")


if __name__ == "__main__":
    unittest.main()

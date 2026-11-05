"""Check first-order scoring arithmetic, the decomposition and the outside-patch invariant."""
import csv
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

import score_first_order as control


def log_probs(rows):
    """A normalized (L, 21) array of log probabilities from per-position dictionaries.

    Each mapping names the probabilities of a few letters; the remaining mass is
    spread evenly over the rest of the alphabet, so every row sums to one.
    """
    table = []
    for mapping in rows:
        remainder = 1.0 - sum(mapping.values())
        if remainder <= 0:
            raise ValueError("Named probabilities must leave mass for the other letters")
        spread = remainder / (len(control.ALPHABET) - len(mapping))
        table.append([math.log(mapping.get(letter, spread)) for letter in control.ALPHABET])
    return np.asarray(table, dtype="float64")


def score_row(pair_id, model_type, order_index, gap, length=10):
    return {"pair_id": pair_id, "model_type": model_type, "order_index": str(order_index),
            "length": str(length), "full_nll_gap": repr(gap / length)}


def write(rows, directory):
    path = Path(directory) / "pair_scores.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


class ScoreAssignmentTest(unittest.TestCase):
    def setUp(self):
        # Position 0 strongly prefers A, position 1 strongly prefers L, the rest are flat.
        self.table = log_probs([{"A": 0.8, "L": 0.1}, {"A": 0.1, "L": 0.8},
                                {"A": 0.45, "L": 0.45}, {"A": 0.45, "L": 0.45}])

    def test_total_is_the_sum_of_position_negative_log_probabilities(self):
        total = control.score_assignment(self.table, "ALAL")
        expected = -(math.log(0.8) + math.log(0.8) + math.log(0.45) + math.log(0.45))
        self.assertAlmostEqual(total, expected)

    def test_selected_positions_restrict_the_sum(self):
        patch = control.score_assignment(self.table, "ALAL", [0, 1])
        self.assertAlmostEqual(patch, -(math.log(0.8) + math.log(0.8)))

    def test_swapping_identities_between_positions_changes_the_total(self):
        # Composition is identical; only the arrangement differs.
        good = control.score_assignment(self.table, "ALAL")
        swapped = control.score_assignment(self.table, "LAAL")
        self.assertGreater(swapped, good)
        self.assertAlmostEqual(swapped - good, 2 * (math.log(0.8) - math.log(0.1)))

    def test_identical_assignments_give_a_zero_gap(self):
        self.assertAlmostEqual(control.score_assignment(self.table, "ALAL"),
                               control.score_assignment(self.table, "ALAL"))

    def test_positions_outside_a_patch_contribute_nothing_to_the_gap(self):
        """The invariant the run asserts: unchanged identities cancel exactly."""
        original, permuted = "ALAL", "LAAL"
        patch = [0, 1]
        chain_gap = (control.score_assignment(self.table, permuted)
                     - control.score_assignment(self.table, original))
        patch_gap = (control.score_assignment(self.table, permuted, patch)
                     - control.score_assignment(self.table, original, patch))
        self.assertAlmostEqual(chain_gap - patch_gap, 0.0)


class FullGapReaderTest(unittest.TestCase):
    def read(self, rows):
        with tempfile.TemporaryDirectory() as directory:
            return control.read_full_gaps(write(rows, directory))

    def test_gaps_are_order_averaged_chain_totals(self):
        gaps = self.read([score_row("p1", "vanilla", 1, 10.0),
                          score_row("p1", "vanilla", 2, 20.0)])
        self.assertAlmostEqual(gaps[("p1", "vanilla")]["delta_full_total"], 15.0)
        self.assertEqual(gaps[("p1", "vanilla")]["order_count"], 2)

    def test_checkpoints_are_kept_apart(self):
        gaps = self.read([score_row("p1", "vanilla", 1, 10.0),
                          score_row("p1", "soluble", 1, -4.0)])
        self.assertAlmostEqual(gaps[("p1", "soluble")]["delta_full_total"], -4.0)

    def test_duplicate_order_index_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate order_index"):
            self.read([score_row("p1", "vanilla", 1, 1.0), score_row("p1", "vanilla", 1, 2.0)])

    def test_gapped_order_indices_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "complete run"):
            self.read([score_row("p1", "vanilla", 1, 1.0), score_row("p1", "vanilla", 3, 2.0)])

    def test_missing_columns_are_reported(self):
        rows = [{k: v for k, v in score_row("p1", "vanilla", 1, 1.0).items() if k != "length"}]
        with self.assertRaisesRegex(ValueError, "missing columns"):
            self.read(rows)


class SummaryTest(unittest.TestCase):
    def rows(self, values):
        """One row per (first-order, full) pair, with the joint term derived as the code does."""
        built = []
        for index, (first_order, full) in enumerate(values):
            built.append({
                "pair_id": "p{}".format(index), "structure_id": "s1", "chain": "A",
                "split": "development", "group": "g1", "model_type": "vanilla",
                "delta_fo_total": first_order, "delta_full_total": full,
                "delta_joint_total": full - first_order,
                "delta_fo_share": first_order / full if abs(full) > control.TOLERANCE else None,
            })
        return built

    def test_sign_test_separates_the_joint_term_from_the_full_gap(self):
        # Every full gap is positive, but the joint residual is negative in two of four.
        checkpoints, _ = control.summarize(self.rows([(9.0, 10.0), (11.0, 10.0),
                                                      (9.0, 10.0), (11.0, 10.0)]))
        block = checkpoints["vanilla"]
        self.assertEqual(block["delta_full_sign_test"]["positive"], 4)
        self.assertEqual(block["delta_joint_sign_test"]["positive"], 2)
        self.assertEqual(block["delta_joint_sign_test"]["negative"], 2)

    def test_first_order_dominance_is_counted_on_the_signed_share(self):
        checkpoints, _ = control.summarize(self.rows([(9.0, 10.0), (1.0, 10.0), (6.0, 10.0)]))
        self.assertEqual(checkpoints["vanilla"]["pairs_where_first_order_dominates"], 2)

    def test_a_first_order_term_opposing_the_full_gap_is_flagged(self):
        checkpoints, _ = control.summarize(self.rows([(-3.0, 10.0), (9.0, 10.0)]))
        self.assertEqual(checkpoints["vanilla"]["pairs_where_joint_reverses_sign"], 1)

    def test_perfect_agreement_gives_unit_r_squared_and_slope(self):
        checkpoints, _ = control.summarize(self.rows([(2.0, 2.0), (4.0, 4.0), (8.0, 8.0)]))
        fit = checkpoints["vanilla"]["full_on_first_order_fit"]
        self.assertAlmostEqual(fit["r_squared"], 1.0)
        self.assertAlmostEqual(fit["slope"], 1.0)

    def test_per_protein_block_counts_positive_joint_terms(self):
        _, proteins = control.summarize(self.rows([(9.0, 10.0), (11.0, 10.0), (9.0, 10.0)]))
        self.assertEqual(len(proteins), 1)
        self.assertEqual(proteins[0]["pair_count"], 3)
        self.assertEqual(proteins[0]["joint_positive_count"], 2)


if __name__ == "__main__":
    unittest.main()

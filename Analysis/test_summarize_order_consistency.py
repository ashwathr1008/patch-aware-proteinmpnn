"""Check order-spread arithmetic, region decomposition and shared-order enforcement."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

import summarize_order_consistency as analysis


def row(pair_id, model_type, order_index, original_full, permuted_full,
        original_patch, permuted_patch, length=10, patch_size=4, structure="p1"):
    """One score row, with the per-residue means the real CSV stores."""
    return {
        "pair_id": pair_id, "structure_id": structure, "chain": "A", "split": "development",
        "group": "g1", "patch_id": structure + "_patch", "model_type": model_type,
        "model_name": "v_48_020", "order_index": str(order_index),
        "order_sha256": "{:064x}".format(order_index), "length": str(length),
        "patch_size": str(patch_size),
        "original_full_nll": repr(original_full / length),
        "permuted_full_nll": repr(permuted_full / length),
        "full_nll_gap": repr((permuted_full - original_full) / length),
        "original_patch_nll": repr(original_patch / patch_size),
        "permuted_patch_nll": repr(permuted_patch / patch_size),
        "patch_nll_gap": repr((permuted_patch - original_patch) / patch_size),
    }


def fixture():
    """Two orders per pair; totals chosen so the expected statistics are exact."""
    return [
        # Chain totals 10 and 14 for the original: mean 12, population SD 2.
        row("p1_1", "vanilla", 1, 10.0, 20.0, 4.0, 9.0),
        row("p1_1", "vanilla", 2, 14.0, 24.0, 4.0, 9.0),
        row("p1_2", "vanilla", 1, 10.0, 8.0, 4.0, 3.0),
        row("p1_2", "vanilla", 2, 10.0, 8.0, 4.0, 3.0),
    ]


def write(rows, directory, name="pair_scores.csv"):
    path = Path(directory) / name
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


class OrderConsistencyTest(unittest.TestCase):
    def analyze(self, rows):
        with tempfile.TemporaryDirectory() as directory:
            path = write(rows, directory)
            output = Path(directory) / "out"
            analysis.run(analysis.argparse.Namespace(
                scores=path, output=output, no_figure=True))
            summary = json.loads((output / "order_consistency_summary.json").read_text())
            pairs = list(csv.DictReader((output / "order_consistency_pair.csv").open()))
            sequences = list(csv.DictReader((output / "order_consistency_sequence.csv").open()))
        return summary, pairs, sequences

    def test_order_spread_uses_chain_totals_not_per_residue_means(self):
        _, pairs, sequences = self.analyze(fixture())
        original = next(s for s in sequences
                        if s["pair_id"] == "p1_1" and s["label"] == "original")
        self.assertAlmostEqual(float(original["chain_total_nll_mean"]), 12.0)
        self.assertAlmostEqual(float(original["chain_total_nll_sd"]), 2.0)
        self.assertAlmostEqual(float(original["per_residue_nll_sd"]), 0.2)
        first = next(p for p in pairs if p["pair_id"] == "p1_1")
        self.assertAlmostEqual(float(first["order_sd_original"]), 2.0)
        self.assertAlmostEqual(float(first["order_sd_permuted"]), 2.0)

    def test_outside_patch_gap_is_the_chain_gap_minus_the_patch_gap(self):
        _, pairs, _ = self.analyze(fixture())
        first = next(p for p in pairs if p["pair_id"] == "p1_1")
        # Chain gap 10, patch gap 5, so the outside remainder is 5.
        self.assertAlmostEqual(float(first["full_gap_total_mean"]), 10.0)
        self.assertAlmostEqual(float(first["patch_gap_total_mean"]), 5.0)
        self.assertAlmostEqual(float(first["outside_gap_total_mean"]), 5.0)
        self.assertAlmostEqual(float(first["outside_share_of_full_gap"]), 0.5)
        self.assertAlmostEqual(float(first["inconsistency_ratio"]), 0.2)

    def test_negative_gap_pair_keeps_its_sign(self):
        _, pairs, _ = self.analyze(fixture())
        second = next(p for p in pairs if p["pair_id"] == "p1_2")
        self.assertAlmostEqual(float(second["full_gap_total_mean"]), -2.0)
        self.assertAlmostEqual(float(second["outside_gap_total_mean"]), -1.0)
        self.assertEqual(second["outside_gap_sign"], "-1")
        self.assertAlmostEqual(float(second["order_sd_mean"]), 0.0)

    def test_sign_test_counts_and_two_sided_p_value(self):
        summary, _, _ = self.analyze(fixture())
        block = summary["checkpoints"]["vanilla"]
        self.assertEqual(block["outside_gap_sign_test"]["positive"], 1)
        self.assertEqual(block["outside_gap_sign_test"]["negative"], 1)
        self.assertAlmostEqual(block["outside_gap_sign_test"]["p_value"], 1.0)
        self.assertEqual(block["pairs_with_order_sd_above_gap"], 0)

    def test_exact_sign_test_matches_the_binomial_tail(self):
        self.assertAlmostEqual(analysis.sign_test([1.0] * 5)["p_value"], 2 / 32)
        self.assertAlmostEqual(analysis.sign_test([1.0, 1.0, -1.0])["p_value"], 1.0)
        self.assertEqual(analysis.sign_test([0.0, 0.0])["p_value"], None)
        self.assertEqual(analysis.sign_test([0.0, 1.0])["ties"], 1)

    def test_stated_and_derived_gaps_must_agree(self):
        rows = fixture()
        rows[0]["full_nll_gap"] = repr(float(rows[0]["full_nll_gap"]) + 0.5)
        with self.assertRaisesRegex(ValueError, "gaps disagree"):
            self.analyze(rows)

    def test_orders_must_be_shared_within_a_structure(self):
        rows = fixture()
        rows[2]["order_sha256"] = "{:064x}".format(99)
        with self.assertRaisesRegex(ValueError, "not shared"):
            self.analyze(rows)

    def test_duplicate_order_index_is_rejected(self):
        rows = fixture()
        rows[1]["order_index"] = "1"
        with self.assertRaisesRegex(ValueError, "Duplicate order_index"):
            self.analyze(rows)

    def test_unequal_order_counts_are_rejected(self):
        # Three orders against the fixture's two, so the count check fires rather
        # than the separate guard against a pair scored under a single order.
        rows = fixture() + [row("p1_3", "vanilla", index, 10.0, 11.0, 4.0, 5.0)
                            for index in (1, 2, 3)]
        with self.assertRaisesRegex(ValueError, "same number of orders"):
            self.analyze(rows)

    def test_a_single_order_pair_is_rejected(self):
        rows = fixture()[:2] + [row("p1_2", "vanilla", 1, 10.0, 8.0, 4.0, 3.0)]
        with self.assertRaisesRegex(ValueError, "at least two shared orders"):
            self.analyze(rows)

    def test_a_patch_covering_the_whole_chain_is_rejected(self):
        rows = fixture()
        for item in rows:
            item["patch_size"] = item["length"]
        with self.assertRaisesRegex(ValueError, "outside region"):
            self.analyze(rows)

    def test_missing_columns_are_reported(self):
        rows = [{k: v for k, v in item.items() if k != "patch_nll_gap"} for item in fixture()]
        with self.assertRaisesRegex(ValueError, "missing columns"):
            self.analyze(rows)


if __name__ == "__main__":
    unittest.main()

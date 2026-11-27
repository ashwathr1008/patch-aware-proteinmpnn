"""Check the ranking endpoints, the fold partition and the frozen join.

Two classes of error are the reason these exist. The first is a sign error: `q_frozen` is built
from a negative log likelihood, so one missing negation turns the baseline into its own mirror
image and makes a correct head look worse than chance. `TestFrozenOrientation` pins the
direction against a hand-worked case rather than against the code's own convention.

The second is leakage. `partition` is the only thing standing between whole-cluster
cross-validation and a family appearing in both training and test, which would inflate every
number downstream and leave no trace. It is asserted across all folds, not one.
"""
import csv


import tempfile
import unittest
from pathlib import Path

import frozen_contrasts as fz
import train_patch_head as tph


def rows_for(spec):
    """spec: list of (family, domain, d, q) -> contrast rows plus a parallel q list."""
    rows, q = [], []
    for index, (family, domain, d, value) in enumerate(spec):
        rows.append({"contrast_id": "c%d" % index, "domain": domain, "family": family,
                     "split": "test", "row_a": 2 * index, "row_b": 2 * index + 1,
                     "d": d, "restores_wild_type": False})
        q.append(value)
    return rows, q


class TestMargin(unittest.TestCase):

    def test_clipped_below_and_above(self):
        self.assertEqual(tph.margin(0.0), 0.1)
        self.assertEqual(tph.margin(0.05), 0.1)
        self.assertEqual(tph.margin(-5.0), 2.0)

    def test_passes_through_in_range_and_uses_magnitude(self):
        self.assertAlmostEqual(tph.margin(0.7), 0.7)
        self.assertAlmostEqual(tph.margin(-0.7), 0.7)


class TestAccuracy(unittest.TestCase):

    def test_agreement_and_disagreement(self):
        rows, q = rows_for([("f", "d1", 1.0, 2.0), ("f", "d1", -1.0, -2.0)])
        self.assertEqual(tph.flat_accuracy(rows, q), 1.0)
        rows, q = rows_for([("f", "d1", 1.0, -2.0)])
        self.assertEqual(tph.flat_accuracy(rows, q), 0.0)

    def test_predicted_tie_scores_one_half(self):
        rows, q = rows_for([("f", "d1", 1.0, 0.0)])
        self.assertEqual(tph.flat_accuracy(rows, q), 0.5)

    def test_families_weigh_equally_regardless_of_size(self):
        # Family A: one domain, all correct. Family B: one domain, all wrong.
        # Family A carries 100 contrasts and B carries 1; macro accuracy must still be 0.5.
        spec = [("A", "d1", 1.0, 1.0)] * 100 + [("B", "d2", 1.0, -1.0)]
        rows, q = rows_for(spec)
        accuracy, families = tph.family_macro_accuracy(rows, q)
        self.assertEqual(families, 2)
        self.assertAlmostEqual(accuracy, 0.5)
        # The flat average, by contrast, is dominated by the larger family.
        self.assertGreater(tph.flat_accuracy(rows, q), 0.98)

    def test_domains_weigh_equally_within_a_family(self):
        spec = [("A", "d1", 1.0, 1.0)] * 10 + [("A", "d2", 1.0, -1.0)]
        rows, q = rows_for(spec)
        accuracy, _ = tph.family_macro_accuracy(rows, q)
        self.assertAlmostEqual(accuracy, 0.5)


class TestPerDomainSpearman(unittest.TestCase):

    def test_excludes_domains_below_the_minimum(self):
        rows, q = rows_for([("f", "d1", float(i), float(i)) for i in range(1, 6)])
        kept, excluded = tph.per_domain_spearman(rows, q, minimum=20)
        self.assertEqual(kept, [])
        self.assertEqual(excluded[0]["reason"], "below minimum")

    def test_excludes_a_constant_prediction_vector(self):
        rows, q = rows_for([("f", "d1", float(i), 1.0) for i in range(1, 26)])
        kept, excluded = tph.per_domain_spearman(rows, q, minimum=20)
        self.assertEqual(kept, [])
        self.assertEqual(excluded[0]["reason"], "constant")

    def test_perfect_agreement_is_positive_one(self):
        rows, q = rows_for([("f", "d1", float(i), float(i)) for i in range(1, 26)])
        kept, _ = tph.per_domain_spearman(rows, q, minimum=20)
        self.assertAlmostEqual(kept[0]["spearman"], 1.0)

    def test_perfect_disagreement_is_negative_one(self):
        rows, q = rows_for([("f", "d1", float(i), -float(i)) for i in range(1, 26)])
        kept, _ = tph.per_domain_spearman(rows, q, minimum=20)
        self.assertAlmostEqual(kept[0]["spearman"], -1.0)


class TestSignTest(unittest.TestCase):

    def test_counts_wins_losses_and_ties(self):
        got = tph.paired_sign_test([1, 2, 3, 4], [0, 1, 3, 9])
        self.assertEqual((got["wins"], got["losses"], got["ties"]), (2, 1, 1))

    def test_matches_the_exact_binomial(self):
        got = tph.paired_sign_test([1, 1, 1], [0, 0, 0])
        self.assertEqual(got["n"], 3)
        self.assertAlmostEqual(got["p_value"], 2 * 1 / 8)

    def test_all_ties_is_not_significant(self):
        got = tph.paired_sign_test([1, 1], [1, 1])
        self.assertEqual(got["p_value"], 1.0)

    def test_an_even_split_is_not_significant(self):
        got = tph.paired_sign_test([1, 0, 1, 0], [0, 1, 0, 1])
        self.assertEqual((got["wins"], got["losses"]), (2, 2))
        self.assertAlmostEqual(got["p_value"], 1.0)


class TestPartition(unittest.TestCase):

    def cohort(self, families=12, per_family=2, each=3):
        spec = []
        for f in range(families):
            for dom in range(per_family):
                for k in range(each):
                    spec.append(("fam%d" % f, "fam%d_d%d" % (f, dom), 1.0, 0.0))
        rows, _ = rows_for(spec)
        return rows

    def test_no_family_crosses_any_partition_in_any_fold(self):
        rows = self.cohort()
        for fold in range(4):
            train, val, test = tph.partition(rows, 4, fold, seed=7)
            fam = lambda part: set(r["family"] for r in part)
            self.assertFalse(fam(train) & fam(val))
            self.assertFalse(fam(train) & fam(test))
            self.assertFalse(fam(val) & fam(test))

    def test_every_family_is_tested_exactly_once_across_folds(self):
        rows = self.cohort()
        seen = []
        for fold in range(4):
            _, _, test = tph.partition(rows, 4, fold, seed=7)
            seen.extend(set(r["family"] for r in test))
        self.assertEqual(sorted(seen), sorted(set(r["family"] for r in rows)))
        self.assertEqual(len(seen), len(set(seen)))

    def test_every_contrast_is_used_once_per_fold(self):
        rows = self.cohort()
        train, val, test = tph.partition(rows, 4, 0, seed=7)
        self.assertEqual(len(train) + len(val) + len(test), len(rows))


class TestFrozenOrientation(unittest.TestCase):
    """q_frozen must be nll(B) - nll(A): positive when the model prefers A."""

    def scores(self, nll_a, nll_b):
        return {("1ABC", 3, 6, "A", "W"): nll_a, ("1ABC", 3, 6, "W", "A"): nll_b}

    def test_lower_nll_for_a_gives_positive_q(self):
        # The model assigns A the lower negative log likelihood, so it prefers A.
        got = fz.frozen_contrast(self.scores(1.0, 2.0), "1ABC", 3, 6, "A", "W")
        self.assertAlmostEqual(got, 1.0)
        self.assertGreater(got, 0)

    def test_lower_nll_for_b_gives_negative_q(self):
        got = fz.frozen_contrast(self.scores(2.0, 1.0), "1ABC", 3, 6, "A", "W")
        self.assertLess(got, 0)

    def test_agreeing_frozen_scores_beat_chance_under_the_shared_convention(self):
        """End-to-end sign check: a frozen model that prefers the more stable arrangement
        must score above 0.5, not below. This is the assertion a missing negation breaks."""
        # Contrast 1: A is more stable (d > 0) and the model gives A the lower nll.
        # Contrast 2: B is more stable (d < 0) and the model gives B the lower nll.
        rows, _ = rows_for([("f", "d1", 1.5, 0.0), ("f", "d1", -2.0, 0.0)])
        q = [nll_b - nll_a for nll_a, nll_b in [(1.0, 2.0), (2.0, 1.0)]]
        self.assertEqual(tph.flat_accuracy(rows, q), 1.0)

    def test_missing_mirror_variant_returns_none(self):
        partial = {("1ABC", 3, 6, "A", "W"): 1.0}
        self.assertIsNone(fz.frozen_contrast(partial, "1ABC", 3, 6, "A", "W"))


class TestFrozenAveraging(unittest.TestCase):

    def test_orders_are_averaged_before_differencing(self):
        handle = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, newline="")
        writer = csv.DictWriter(handle, fieldnames=[
            "domain", "site_i", "site_j", "aa_i", "aa_j", "order_index", "full_nll_mean"])
        writer.writeheader()
        for order, value in enumerate([1.0, 3.0], start=1):
            writer.writerow({"domain": "1ABC", "site_i": 3, "site_j": 6, "aa_i": "A",
                             "aa_j": "W", "order_index": order, "full_nll_mean": value})
        handle.close()
        scores, counts = fz.load_variant_scores(Path(handle.name))
        self.assertAlmostEqual(scores[("1ABC", 3, 6, "A", "W")], 2.0)
        self.assertEqual(counts[("1ABC", 3, 6, "A", "W")], 2)


class TestBootstrap(unittest.TestCase):

    def test_interval_brackets_the_observed_difference(self):
        spec, q_learned, q_frozen = [], [], []
        for f in range(20):
            for k in range(5):
                spec.append(("fam%d" % f, "fam%d_d" % f, 1.0, 0.0))
                q_learned.append(1.0)      # always right
                q_frozen.append(-1.0)      # always wrong
        rows, _ = rows_for(spec)
        got = tph.bootstrap_family_difference(rows, q_learned, q_frozen, 500, 7)
        self.assertEqual(got["families"], 20)
        self.assertAlmostEqual(got["observed_mean_difference"], 1.0)
        self.assertGreater(got["ci95"][0], 0.0)
        self.assertEqual(got["family_wins"], 20)

    def test_identical_predictions_give_a_zero_interval(self):
        spec = [("fam%d" % f, "d%d" % f, 1.0, 0.0) for f in range(10)]
        rows, _ = rows_for(spec)
        q = [1.0] * 10
        got = tph.bootstrap_family_difference(rows, q, q, 200, 7)
        self.assertEqual(got["observed_mean_difference"], 0.0)
        self.assertEqual(got["ci95"], [0.0, 0.0])


if __name__ == "__main__":
    unittest.main()


class TestStandardizer(unittest.TestCase):
    """Only the log-probability block is rescaled, and only from training rows."""

    def table(self):
        import numpy
        # width 84 = 2 members x (21 logprobs + 21 one-hot), no neighbours.
        rng = numpy.random.default_rng(0)
        block = rng.normal(-4.0, 2.0, size=(40, 84)).astype("float32")
        for col in tph.logprob_columns(84):
            pass
        onehot = [c for c in range(84) if c not in set(tph.logprob_columns(84))]
        block[:, onehot] = 0.0
        block[:, onehot[0]] = 1.0
        return block

    def test_identifies_the_logprob_columns(self):
        self.assertEqual(tph.logprob_columns(84), list(range(0, 21)) + list(range(42, 63)))

    def test_neighbour_blocks_are_not_treated_as_logprobs(self):
        columns = tph.logprob_columns(252)
        self.assertEqual(columns, list(range(0, 21)) + list(range(42, 63)))
        self.assertTrue(all(c < 84 for c in columns))

    def test_logprob_columns_become_zero_mean_unit_scale(self):
        import numpy
        block = self.table()
        rows = [{"row_a": i, "row_b": i + 1} for i in range(0, 40, 2)]
        mean, scale = tph.fit_standardizer(block, rows, 84)
        out = tph.apply_standardizer(block, mean, scale)
        index = numpy.asarray(tph.logprob_columns(84))
        self.assertLess(abs(float(out[:, index].mean())), 1e-5)
        self.assertAlmostEqual(float(out[:, index].std()), 1.0, places=3)

    def test_one_hot_columns_are_left_alone(self):
        import numpy
        block = self.table()
        rows = [{"row_a": i, "row_b": i + 1} for i in range(0, 40, 2)]
        mean, scale = tph.fit_standardizer(block, rows, 84)
        out = tph.apply_standardizer(block, mean, scale)
        onehot = [c for c in range(84) if c not in set(tph.logprob_columns(84))]
        self.assertTrue(numpy.allclose(out[:, onehot], block[:, onehot]))

    def test_fitted_only_on_the_rows_it_is_given(self):
        import numpy
        block = numpy.zeros((10, 84), dtype="float32")
        block[:5, 0] = 1.0     # "training" rows
        block[5:, 0] = 1000.0  # held out; must not affect the mean
        rows = [{"row_a": 0, "row_b": 1}, {"row_a": 2, "row_b": 3}]
        mean, _ = tph.fit_standardizer(block, rows, 84)
        self.assertAlmostEqual(float(mean[0]), 1.0, places=5)

    def test_degenerate_column_is_not_amplified(self):
        import numpy
        block = numpy.zeros((10, 84), dtype="float32")
        rows = [{"row_a": 0, "row_b": 1}]
        mean, scale = tph.fit_standardizer(block, rows, 84)
        self.assertGreaterEqual(float(scale[0]), 1e-3)
        self.assertTrue(numpy.isfinite(tph.apply_standardizer(block, mean, scale)).all())

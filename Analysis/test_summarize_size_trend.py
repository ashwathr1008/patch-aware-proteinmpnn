"""Check the size-trend statistics, the exact permutation test and the non-result verdict."""
import unittest

import summarize_size_trend as analysis


def score_row(structure, length, joint, full, share, model_type="vanilla"):
    return {"structure_id": structure, "chain": "A", "model_type": model_type,
            "split": "development", "group": "g1", "length": str(length),
            "delta_joint_total": repr(joint), "delta_full_total": repr(full),
            "delta_fo_share": repr(share)}


class CorrelationTest(unittest.TestCase):
    def test_perfect_agreement_and_reversal(self):
        self.assertAlmostEqual(analysis.pearson([1, 2, 3], [2, 4, 6]), 1.0)
        self.assertAlmostEqual(analysis.pearson([1, 2, 3], [6, 4, 2]), -1.0)

    def test_a_constant_series_has_no_correlation(self):
        self.assertIsNone(analysis.pearson([1, 2, 3], [5, 5, 5]))

    def test_ranks_average_over_ties(self):
        self.assertEqual(analysis.ranks([3.0, 1.0, 3.0]), [1.5, 0.0, 1.5])

    def test_spearman_ignores_monotone_rescaling(self):
        self.assertAlmostEqual(analysis.spearman([1, 2, 3, 4], [1, 4, 9, 16]), 1.0)


class ExactPermutationTest(unittest.TestCase):
    def test_a_perfect_relation_has_the_smallest_attainable_p(self):
        """Only the identity and reversed orderings reach |r| = 1, so p = 2/n!."""
        observed, p_value = analysis.exact_permutation_p([1, 2, 3, 4], [1, 2, 3, 4],
                                                         analysis.pearson)
        self.assertAlmostEqual(observed, 1.0)
        self.assertAlmostEqual(p_value, 2 / 24)

    def test_the_p_value_is_two_sided(self):
        forward = analysis.exact_permutation_p([1, 2, 3, 4], [1, 2, 3, 4], analysis.pearson)[1]
        reverse = analysis.exact_permutation_p([1, 2, 3, 4], [4, 3, 2, 1], analysis.pearson)[1]
        self.assertAlmostEqual(forward, reverse)

    def test_every_permutation_is_enumerated(self):
        # A flat relation is matched or beaten by every ordering, so p is exactly 1.
        observed, p_value = analysis.exact_permutation_p([1, 2, 3], [1, 1, 2], analysis.pearson)
        self.assertIsNotNone(observed)
        self.assertLessEqual(p_value, 1.0)
        self.assertGreater(p_value, 0.0)

    def test_too_many_points_are_refused_rather_than_approximated(self):
        size = analysis.MAXIMUM_EXACT_N + 1
        with self.assertRaisesRegex(ValueError, "refuses"):
            analysis.exact_permutation_p(list(range(size)), list(range(size)), analysis.pearson)


class CollectTest(unittest.TestCase):
    def test_pairs_are_averaged_within_a_structure(self):
        rows = [score_row("s1", 10, 2.0, 10.0, 0.8),
                score_row("s1", 10, 4.0, 10.0, 0.9)]
        records = analysis.collect(rows)
        self.assertEqual(len(records), 1)
        self.assertAlmostEqual(records[0]["joint_total"], 3.0)
        self.assertAlmostEqual(records[0]["joint_per_residue"], 0.3)
        self.assertEqual(records[0]["pair_count"], 2)

    def test_checkpoints_are_kept_apart(self):
        rows = [score_row("s1", 10, 2.0, 10.0, 0.8),
                score_row("s1", 10, 9.0, 10.0, 0.5, model_type="soluble")]
        records = analysis.collect(rows)
        self.assertEqual({r["model_type"] for r in records}, {"vanilla", "soluble"})

    def test_per_residue_normalization_uses_chain_length(self):
        rows = [score_row("short", 10, 5.0, 10.0, 0.8),
                score_row("long", 100, 5.0, 10.0, 0.8)]
        records = {r["structure_id"]: r for r in analysis.collect(rows)}
        self.assertAlmostEqual(records["short"]["joint_per_residue"], 0.5)
        self.assertAlmostEqual(records["long"]["joint_per_residue"], 0.05)


class VerdictTest(unittest.TestCase):
    def rows(self, pairs, model_type="vanilla"):
        return [score_row("s{}".format(i), length, joint, 10.0, 0.8, model_type)
                for i, (length, joint) in enumerate(pairs)]

    def test_a_flat_relation_reports_no_significant_trend(self):
        records = analysis.collect(self.rows([(10, 1.0), (20, 1.0), (30, 1.0), (40, 1.0)]))
        result = analysis.verdict(analysis.analyze(records))
        self.assertFalse(result["any_significant"])
        self.assertIn("non-result", result["statement"])

    def test_a_perfect_trend_is_flagged_as_significant(self):
        # Five points give a smallest attainable p of 2/120, below 0.05.
        records = analysis.collect(
            self.rows([(10, 1.0), (20, 2.0), (30, 3.0), (40, 4.0), (50, 5.0)]))
        result = analysis.verdict(analysis.analyze(records))
        self.assertTrue(result["any_significant"])
        self.assertIn("joint_total", [r["measure"] for r in result["significant_relationships"]])

    def test_the_statement_reports_the_actual_structure_count(self):
        """The statement is quotable output, so it must not hardcode a sample size."""
        for count in (4, 5, 6):
            records = analysis.collect(
                self.rows([(10 * (i + 1), 1.0) for i in range(count)]))
            statement = analysis.verdict(analysis.analyze(records))["statement"]
            self.assertIn("{} structures".format(count), statement)

    def test_the_statement_only_claims_disagreement_when_it_exists(self):
        # All measures rise together, so nothing disagrees and the statement must not say so.
        records = analysis.collect(
            self.rows([(10, 1.0), (20, 4.0), (30, 9.0), (40, 16.0)]))
        result = analysis.verdict(analysis.analyze(records))
        self.assertEqual(result["normalizations_disagree"], [])
        self.assertNotIn("disagree in sign", result["statement"])
        self.assertIn("agree in sign", result["statement"])

    def test_disagreeing_normalizations_are_named_in_the_statement(self):
        records = analysis.collect(
            self.rows([(10, 5.0), (20, 5.2), (30, 5.4), (40, 5.6)]))
        result = analysis.verdict(analysis.analyze(records))
        # joint_total rises while joint_per_residue falls, so they must be reported apart.
        self.assertIn("joint_per_residue", result["normalizations_disagree"])
        self.assertIn("joint_per_residue", result["statement"])

    def test_an_absent_reference_measure_is_reported_rather_than_read_as_agreement(self):
        """With joint_total constant it lands in undefined_measures and cannot be compared."""
        records = analysis.collect(self.rows([(10, 3.0), (20, 3.0), (30, 3.0), (40, 3.0)]))
        block = analysis.analyze(records)["vanilla"]
        self.assertIn("joint_total", [u["measure"] for u in block["undefined_measures"]])
        result = analysis.verdict({"vanilla": block})
        self.assertFalse(result["disagreement_assessable"])
        self.assertIn("could not be compared", result["statement"])

    def test_four_points_cannot_reach_significance(self):
        """At n=4 the smallest attainable p is 2/24, so no result can be significant."""
        records = analysis.collect(self.rows([(10, 1.0), (20, 2.0), (30, 3.0), (40, 4.0)]))
        block = analysis.analyze(records)["vanilla"]
        self.assertAlmostEqual(block["measures"]["joint_total"]["pearson_exact_p"], 2 / 24)
        self.assertFalse(block["measures"]["joint_total"]["significant_at_0.05"])


if __name__ == "__main__":
    unittest.main()

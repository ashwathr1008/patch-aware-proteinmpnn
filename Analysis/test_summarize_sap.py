"""Check rank statistics, tie accounting and pool summaries for the SAP baseline."""
import math
import unittest

import summarize_sap as analysis


class RankStatisticsTest(unittest.TestCase):
    def test_ranks_average_over_tied_values(self):
        self.assertEqual(analysis.ranks([5.0, 1.0, 5.0, 3.0]), [2.5, 0.0, 2.5, 1.0])

    def test_a_monotone_relation_gives_unit_spearman(self):
        self.assertAlmostEqual(analysis.spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)

    def test_a_reversed_relation_gives_minus_one(self):
        self.assertAlmostEqual(analysis.spearman([1, 2, 3, 4], [40, 30, 20, 10]), -1.0)

    def test_spearman_ignores_monotone_rescaling(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        squared = [v ** 2 for v in values]
        self.assertAlmostEqual(analysis.spearman(values, squared), 1.0)

    def test_a_constant_series_has_no_correlation(self):
        self.assertIsNone(analysis.spearman([1, 2, 3, 4], [7, 7, 7, 7]))

    def test_too_few_points_return_nothing(self):
        self.assertIsNone(analysis.pearson([1.0], [2.0]))

    def test_pearson_recovers_a_known_value(self):
        self.assertAlmostEqual(analysis.pearson([1, 2, 3], [1, 2, 3]), 1.0)
        self.assertAlmostEqual(analysis.pearson([1, 2, 3], [3, 2, 1]), -1.0)


class SignTestTest(unittest.TestCase):
    def test_all_positive_matches_the_binomial_tail(self):
        self.assertAlmostEqual(analysis.sign_test([1.0] * 6)["p_value"], 2 / 64)

    def test_ties_are_excluded_from_the_trial_count(self):
        result = analysis.sign_test([1.0, -1.0, 0.0])
        self.assertEqual((result["positive"], result["negative"], result["ties"]), (1, 1, 1))
        self.assertEqual(result["nonties"], 2)

    def test_all_ties_give_no_p_value(self):
        self.assertIsNone(analysis.sign_test([0.0, 0.0])["p_value"])


class PairScoringTest(unittest.TestCase):
    def setUp(self):
        import numpy as np
        # Burial varies across positions, so a permutation must move the score.
        self.scaffolds = {("p1", "A"): {
            "burial": np.array([0.1, 0.9, 0.5, 0.5]), "window": np.eye(4, dtype=bool),
            "native_sequence": "FRLK", "length": 4, "split": "development", "group": "g1"}}
        self.dataset = {"pairs": [{
            "pair_id": "p1_1", "structure_id": "p1", "chain": "A", "patch_id": "p1_patch",
            "original_sequence": "FRLK", "permuted_sequence": "RFLK", "patch_indices": [1, 2],
            "controls": {"hydrophobic_fraction_delta": 0.0}}]}

    def test_a_tied_contact_descriptor_is_flagged_while_sap_separates(self):
        rows = analysis.score_pairs(self.dataset, self.scaffolds, {})
        self.assertTrue(rows[0]["contact_descriptor_tie"])
        self.assertFalse(rows[0]["sap_simple_tie"])
        self.assertNotAlmostEqual(rows[0]["delta_sap_simple"], 0.0)

    def test_moving_the_hydrophobe_to_the_exposed_position_raises_the_score(self):
        """F is hydrophobic and R is not; position 2 is the exposed one."""
        rows = analysis.score_pairs(self.dataset, self.scaffolds, {})
        self.assertGreater(rows[0]["delta_sap_simple"], 0.0)

    def test_changed_identity_count_is_derived_from_the_sequences(self):
        rows = analysis.score_pairs(self.dataset, self.scaffolds, {})
        self.assertEqual(rows[0]["changed_identity_count"], 2)

    def test_decomposition_columns_stay_empty_when_not_supplied(self):
        rows = analysis.score_pairs(self.dataset, self.scaffolds, {})
        self.assertIsNone(rows[0]["delta_fo_total"])

    def test_summary_counts_ties_for_both_descriptors(self):
        rows = analysis.score_pairs(self.dataset, self.scaffolds, {})
        summary = analysis.summarize_pairs(rows)
        self.assertEqual(summary["contact_descriptor_ties"], 1)
        self.assertEqual(summary["sap_simple_ties"], 0)
        self.assertEqual(summary["pair_count"], 1)


class CandidateSummaryTest(unittest.TestCase):
    def pool(self, structure, model_type, pairs):
        return [{"sequence_id": "{}_{}_{}".format(structure, model_type, index),
                 "structure_id": structure, "chain": "A", "model_type": model_type,
                 "record_type": "generated", "sample_index": index, "length": 4,
                 "generation_score": nll, "native_recovery": 0.5,
                 "sap_simple": sap_value, "sap_window": sap_value}
                for index, (nll, sap_value) in enumerate(pairs)]

    def test_pool_correlations_are_computed_within_each_pool(self):
        rows = (self.pool("s1", "vanilla", [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0), (4.0, 4.0)])
                + self.pool("s2", "vanilla", [(9.0, 4.0), (8.0, 3.0), (7.0, 2.0), (6.0, 1.0)]))
        summary = analysis.summarize_candidates(rows)
        self.assertEqual(summary["pool_count"], 2)
        for pool in summary["pools"]:
            self.assertAlmostEqual(pool["spearman_nll_vs_sap_simple"], 1.0)
        self.assertAlmostEqual(summary["mean_spearman_simple"], 1.0)

    def test_native_records_are_excluded_from_the_pools(self):
        rows = self.pool("s1", "vanilla", [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)])
        rows.append({**rows[0], "record_type": "native", "sequence_id": "native"})
        summary = analysis.summarize_candidates(rows)
        self.assertEqual(summary["generated_count"], 3)
        self.assertIsNotNone(summary["pools"][0]["native_sap_simple"])

    def test_opposing_pools_cancel_in_the_mean_and_show_in_the_sign_test(self):
        rows = (self.pool("s1", "vanilla", [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0), (4.0, 4.0)])
                + self.pool("s2", "vanilla", [(1.0, 4.0), (2.0, 3.0), (3.0, 2.0), (4.0, 1.0)]))
        summary = analysis.summarize_candidates(rows)
        self.assertAlmostEqual(summary["mean_spearman_simple"], 0.0)
        self.assertEqual(summary["pool_sign_test"]["positive"], 1)
        self.assertEqual(summary["pool_sign_test"]["negative"], 1)

    def test_the_significance_threshold_counts_only_strong_pools(self):
        rows = self.pool("s1", "vanilla", [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0), (4.0, 4.0)])
        summary = analysis.summarize_candidates(rows)
        self.assertEqual(summary["pools_individually_significant"], 1)


if __name__ == "__main__":
    unittest.main()

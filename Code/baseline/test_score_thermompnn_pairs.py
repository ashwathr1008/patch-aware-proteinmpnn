"""Check the position mapping and the double-mutant planning.

Both are places where a wrong answer would look right. A position mapping that assumed
`index == number - 1` returns a valid index for every site and silently scores the wrong
residues; the planner has to keep a contrast's two arrangements on the same ordered position
pair while upstream requires pos_i < pos_j, and getting that backwards would compare an
arrangement against itself. Neither needs the model, so both are tested directly.
"""
import unittest

import score_thermompnn_pairs as scorer


def contrast(site_i, site_j, aa_i, aa_j, difference="1.0"):
    return {"domain": "TEST", "site_i": str(site_i), "site_j": str(site_j),
            "aa_i": aa_i, "aa_j": aa_j, "difference": difference,
            "uncertainty": "0.1", "resolved": "True"}


class TestResidueIndex(unittest.TestCase):

    def test_contiguous_numbering(self):
        pdb = {"resn_list_A": ["1", "2", "3"], "seq": "ACD"}
        self.assertEqual(scorer.residue_index(pdb), {"1": 0, "2": 1, "3": 2})

    def test_offset_numbering_is_not_assumed_to_start_at_one(self):
        """1VII is deposited as residues 41-76; number - 1 would be wrong by 40."""
        pdb = {"resn_list_A": ["41", "42", "43"], "seq": "ACD"}
        self.assertEqual(scorer.residue_index(pdb), {"41": 0, "42": 1, "43": 2})

    def test_gapped_numbering(self):
        pdb = {"resn_list_A": ["5", "6", "9", "10"], "seq": "ACDE"}
        self.assertEqual(scorer.residue_index(pdb), {"5": 0, "6": 1, "9": 2, "10": 3})

    def test_whitespace_is_stripped(self):
        pdb = {"resn_list_A": [" 7", "8 "], "seq": "AC"}
        self.assertEqual(scorer.residue_index(pdb), {"7": 0, "8": 1})

    def test_multiple_chains_are_refused(self):
        pdb = {"resn_list_A": ["1"], "resn_list_B": ["1"], "seq": "AA"}
        with self.assertRaises(ValueError):
            scorer.residue_index(pdb)

    def test_duplicate_numbers_are_refused(self):
        pdb = {"resn_list_A": ["3", "3"], "seq": "AC"}
        with self.assertRaises(ValueError):
            scorer.residue_index(pdb)


class TestPlanDomain(unittest.TestCase):

    def setUp(self):
        # Sites 1..6 carry A C D E F G.
        self.index = {str(n): n - 1 for n in range(1, 7)}
        self.sequence = "ACDEFG"

    def test_both_arrangements_become_requests(self):
        requests, usable, rejected = scorer.plan_domain(
            [contrast(1, 2, "K", "W")], self.index, self.sequence)
        self.assertEqual(rejected, [])
        self.assertEqual(len(usable), 1)
        row, forward, reverse = usable[0]
        self.assertEqual(forward, (0, 1, "K", "W"))
        self.assertEqual(reverse, (0, 1, "W", "K"))
        self.assertEqual(set(requests), {forward, reverse})
        # Wild types recorded as alphabet indices for the two ordered positions: A then C.
        self.assertEqual(requests[forward],
                         (scorer.ALPHABET.index("A"), scorer.ALPHABET.index("C")))

    def test_arrangement_matching_wild_type_at_one_site_is_refused(self):
        """aa_i == wt at site 1, so the forward arrangement is a single mutant."""
        requests, usable, rejected = scorer.plan_domain(
            [contrast(1, 2, "A", "W")], self.index, self.sequence)
        self.assertEqual(usable, [])
        self.assertEqual(requests, {})
        self.assertEqual(len(rejected), 1)
        self.assertIn("double mutant", rejected[0][1])

    def test_arrangement_equal_to_wild_type_is_refused(self):
        requests, usable, rejected = scorer.plan_domain(
            [contrast(1, 2, "A", "C")], self.index, self.sequence)
        self.assertEqual(usable, [])
        self.assertIn("double mutant", rejected[0][1])

    def test_reverse_arrangement_is_checked_too(self):
        """Forward (K,W) is a clean double, but reverse puts W at site 1 and K at site 2;
        with wt C at site 2 that is still double, so this one must pass -- whereas a pair
        whose reverse hits a wild type must not."""
        _, usable, _ = scorer.plan_domain(
            [contrast(1, 2, "K", "W")], self.index, self.sequence)
        self.assertEqual(len(usable), 1)
        # Site 3 is D; identities {D, K} make the reverse arrangement D-at-3 = wild type.
        _, usable, rejected = scorer.plan_domain(
            [contrast(2, 3, "D", "K")], self.index, self.sequence)
        self.assertEqual(usable, [])
        self.assertIn("double mutant", rejected[0][1])

    def test_positions_are_ordered_and_identities_follow(self):
        """When site_i maps above site_j, positions swap and the identities must swap with
        them, or the arrangement described is no longer the one that was measured."""
        index = {"10": 5, "20": 2}          # site 10 -> index 5, site 20 -> index 2
        sequence = "ACDEFG"                  # index 2 is D, index 5 is G
        requests, usable, rejected = scorer.plan_domain(
            [contrast(10, 20, "K", "W")], index, sequence)
        self.assertEqual(rejected, [])
        row, forward, reverse = usable[0]
        # forward = K on site 10 (index 5) and W on site 20 (index 2), written low position
        # first: (2, 5, W, K).
        self.assertEqual(forward, (2, 5, "W", "K"))
        self.assertEqual(reverse, (2, 5, "K", "W"))
        # Wild types are those of the ordered positions: index 2 is D, index 5 is G.
        self.assertEqual(requests[forward],
                         (scorer.ALPHABET.index("D"), scorer.ALPHABET.index("G")))

    def test_requests_are_shared_between_contrasts_on_one_pair(self):
        requests, usable, _ = scorer.plan_domain(
            [contrast(1, 2, "K", "W"), contrast(1, 2, "K", "W")],
            self.index, self.sequence)
        self.assertEqual(len(usable), 2)
        self.assertEqual(len(requests), 2)

    def test_absent_site_is_refused(self):
        _, usable, rejected = scorer.plan_domain(
            [contrast(1, 99, "K", "W")], self.index, self.sequence)
        self.assertEqual(usable, [])
        self.assertIn("absent", rejected[0][1])

    def test_non_standard_residue_is_refused(self):
        _, usable, rejected = scorer.plan_domain(
            [contrast(1, 2, "X", "W")], self.index, self.sequence)
        self.assertEqual(usable, [])
        self.assertIn("non-standard", rejected[0][1])


if __name__ == "__main__":
    unittest.main()

"""Tests for scientific edge cases in the offline audit."""
import unittest
from pathlib import Path
import tempfile
from audit_esol import exact_mapping, numeric, pdb_ids
from run_baseline import stage_backbone


class AuditTests(unittest.TestCase):
    def test_missing_zero_and_unusual_labels_remain_distinct(self):
        self.assertIsNone(numeric(''))
        self.assertIsNone(numeric('NaN'))
        self.assertIsNone(numeric('unknown'))
        self.assertEqual(numeric('0'), 0)
        self.assertEqual(numeric('147'), 147)

    def test_candidates_not_arbitrary_four_character_words(self):
        self.assertEqual(pdb_ids('1onr; 1ONR / 2ABC:A'), ['1ONR', '2ABC'])
        self.assertEqual(pdb_ids('none; X1ABC'), [])

    def test_full_partial_mutated_and_ambiguous(self):
        self.assertEqual(exact_mapping('ACDE', 'ACDE')['mapping_status'], 'exact_full_length')
        partial = exact_mapping('MACDEK', 'ACDE')
        self.assertEqual(partial['reference_start'], 2)
        self.assertAlmostEqual(partial['reference_coverage'], 4 / 6)
        self.assertEqual(exact_mapping('ACDE', 'ACFE')['mapping_status'], 'no_exact_contiguous_match')
        self.assertEqual(exact_mapping('ACAC', 'AC')['mapping_status'], 'ambiguous_exact_match')

    def test_real_backbone_mapping(self):
        source = Path(__file__).resolve().parents[2] / 'Data/compatibility_pairs/inputs/1UBQ_A.pdb'
        with tempfile.TemporaryDirectory() as directory:
            chain = stage_backbone(source, 'A', Path(directory) / 'staged.pdb')
        self.assertEqual(chain['length'], 76)
        self.assertEqual(exact_mapping(chain['native_sequence'], chain['native_sequence'])['mapping_status'], 'exact_full_length')


if __name__ == '__main__':
    unittest.main()

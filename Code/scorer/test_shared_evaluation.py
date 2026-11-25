"""The shared swap benchmark must stay out of mixed train and validation."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scorer.run_v3_10k import shared_test_data


class SharedEvaluationTests(unittest.TestCase):
    def fixtures(self):
        training = (None, {"a": {"homology_group": "family_a", "split": "train"}},
                    [{"parent_id": "a", "split": "train", "reference_sequence": "ACDE", "variant_sequence": "CADE"}])
        heldout = (None, {"b": {"homology_group": "family_b", "split": "test"}},
                   [{"parent_id": "b", "split": "test", "reference_sequence": "FGHI", "variant_sequence": "GFHI", "is_swap": True}])
        return training, heldout

    def test_disjoint_common_test_is_accepted(self):
        training, heldout = self.fixtures()
        with patch("scorer.data.load_dataset", side_effect=[training, heldout]):
            parents, pairs = shared_test_data("mixed", "swaps")
        self.assertEqual(pairs, heldout[2])

    def test_family_and_exact_sequence_overlap_rejected(self):
        training, heldout = self.fixtures()
        heldout[1]["b"]["homology_group"] = "family_a"
        with patch("scorer.data.load_dataset", side_effect=[training, heldout]):
            with self.assertRaisesRegex(ValueError, "family"):
                shared_test_data("mixed", "swaps")
        training, heldout = self.fixtures()
        heldout[2][0]["reference_sequence"] = "CADE"
        with patch("scorer.data.load_dataset", side_effect=[training, heldout]):
            with self.assertRaisesRegex(ValueError, "sequence"):
                shared_test_data("mixed", "swaps")


if __name__ == "__main__":
    unittest.main()

"""Regression checks for filtering actual composition rather than pair_kind."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scorer.prepare_swaps_only import select_swaps


class SwapSelectionTests(unittest.TestCase):
    def test_preserves_both_swap_categories_labels_and_splits(self):
        pairs = [
            {"pair_id": "a", "parent_id": "p", "reference_sequence": "ACDE", "variant_sequence": "CADE", "pair_kind": "reference_variant", "ddg": 1.2},
            {"pair_id": "b", "parent_id": "p", "reference_sequence": "FGDE", "variant_sequence": "GFDE", "pair_kind": "measured_swap", "ddg": -0.8},
            {"pair_id": "c", "parent_id": "q", "reference_sequence": "ACDE", "variant_sequence": "VCDE", "pair_kind": "reference_variant", "ddg": 0.3},
            {"pair_id": "d", "parent_id": "p", "reference_sequence": "ACDE", "variant_sequence": "ACDE", "pair_kind": "reference_variant", "ddg": 0.},
        ]
        manifest = {"parents": [{"parent_id": "p", "split": "test", "homology_group": "g"},
                                {"parent_id": "q", "split": "train", "homology_group": "h"}], "pairs": pairs}
        selected = select_swaps(manifest)
        self.assertEqual(selected["pairs"], pairs[:2])
        self.assertEqual(selected["parents"], manifest["parents"][:1])
        self.assertEqual(len(manifest["pairs"]), 4)
        self.assertEqual(len(manifest["parents"]), 2)


if __name__ == "__main__":
    unittest.main()

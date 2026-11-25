"""Ensure subsampling never reshuffles existing parent/homology partitions."""
from collections import Counter
from itertools import permutations
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scorer.prepare_10k import select_subset


class SubsetTests(unittest.TestCase):
    def fixture(self):
        parents, pairs = [], []
        for split, sequence in (("train", "ACDE"), ("validation", "FGHI"), ("test", "KLMN")):
            parents.append({"parent_id": split, "homology_group": split + "_family", "split": split})
            for i, variant in enumerate(list(permutations(sequence))[1:11]):
                pairs.append({"pair_id": split + str(i), "parent_id": split,
                              "reference_sequence": sequence, "variant_sequence": "".join(variant), "ddg": i / 10})
        return {"parents": parents, "pairs": pairs}

    def test_exact_reproducible_counts_with_unchanged_records(self):
        source = self.fixture()
        counts = {"train": 6, "validation": 2, "test": 2}
        first = select_subset(source, counts)
        self.assertEqual(first, select_subset(source, counts))
        self.assertEqual(Counter(p["parent_id"] for p in first["pairs"]), counts)
        self.assertEqual(first["parents"], source["parents"])
        self.assertTrue(all(p in source["pairs"] for p in first["pairs"]))
        self.assertEqual(len({p["pair_id"] for p in first["pairs"]}), 10)
        self.assertEqual(len(source["pairs"]), 30)

    def test_group_crossing_and_insufficient_pools_rejected(self):
        source = self.fixture()
        source["parents"][1]["homology_group"] = source["parents"][0]["homology_group"]
        with self.assertRaisesRegex(ValueError, "homology group"):
            select_subset(source, {"train": 6, "validation": 2, "test": 2})
        with self.assertRaisesRegex(ValueError, "Insufficient"):
            select_subset(self.fixture(), {"train": 11, "validation": 2, "test": 2})

    def test_mixed_mode_retains_ordinary_mutations_without_changing_splits(self):
        source = self.fixture()
        for pair in source["pairs"]:
            pair["variant_sequence"] = "V" + pair["reference_sequence"][1:]
        counts = {"train": 6, "validation": 2, "test": 2}
        with self.assertRaises(ValueError):
            select_subset(source, counts)
        selected = select_subset(source, counts, require_swaps=False)
        self.assertEqual(len(selected["pairs"]), 10)
        self.assertEqual(selected["parents"], source["parents"])
        self.assertTrue(all(p in source["pairs"] for p in selected["pairs"]))


if __name__ == "__main__":
    unittest.main()

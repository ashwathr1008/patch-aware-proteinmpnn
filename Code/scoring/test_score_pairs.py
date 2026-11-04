"""Input, order-observation and score-accounting checks without model execution."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import torch
if __package__:
    from . import score_pairs as scorer
else:
    import score_pairs as scorer


def pair():
    return {"pair_id": "p1", "structure_id": "fixture", "chain": "A", "patch_id": "patch1",
            "original_sequence": "ACDE", "permuted_sequence": "ADCE", "patch_indices": [2, 3]}


class InputTests(unittest.TestCase):
    def test_valid_pair_and_native_mismatch(self):
        self.assertEqual(scorer.validate_pair(pair(), 4, "ACDE"), 2)
        with self.assertRaisesRegex(ValueError, "native"):
            scorer.validate_pair(pair(), 4, "ACDF")

    def test_composition_and_outside_changes_are_rejected(self):
        for sequence in ("ADFE", "CDCA"):
            changed = pair()
            changed["permuted_sequence"] = sequence
            with self.assertRaises(ValueError):
                scorer.validate_pair(changed)

    def test_bad_indices_identical_sequences_and_unknown_amino_acids(self):
        for change in ({"patch_indices": [1, 1]}, {"patch_indices": [0, 3]},
                       {"patch_indices": [2, 5]}, {"patch_indices": [True, 3]},
                       {"permuted_sequence": "ACDE"}, {"permuted_sequence": "AXDE"}):
            changed = pair()
            changed.update(change)
            with self.assertRaises(ValueError):
                scorer.validate_pair(changed)

    def test_load_verifies_source_hash_and_unique_pair_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "fixture.pdb"
            source.write_bytes(b"fixture")
            manifest = {"schema_version": 1, "purpose": "patch_compatibility_pilot",
                        "structures": [{"structure_id": "fixture", "chain": "A", "pdb_path": "fixture.pdb",
                                        "sha256": hashlib.sha256(b"fixture").hexdigest(), "group": "fixture", "split": "development"}],
                        "pairs": [pair()]}
            path = root / "pairs.json"
            path.write_text(json.dumps(manifest))
            self.assertEqual(len(scorer.load_pairs(path)[0]["pairs"]), 1)
            manifest["pairs"].append(pair())
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "Duplicate pair_id"):
                scorer.load_pairs(path)
            source.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "checksum"):
                scorer.load_pairs(path)

    def test_unique_orders_are_stable_and_structure_specific(self):
        first = scorer.create_orders(12, 8, 137, "fixture", "A")
        self.assertEqual(first, scorer.create_orders(12, 8, 137, "fixture", "A"))
        self.assertNotEqual(first, scorer.create_orders(12, 8, 137, "other", "A"))
        self.assertEqual(len({r["order_sha256"] for r in first}), 8)
        for row in first:
            self.assertEqual(sorted(row["decoding_order_one_based"]), list(range(1, 13)))
        with self.assertRaises(ValueError):
            scorer.create_orders(2, 3, 1, "x", "A")


class FakeModel:
    def __init__(self, mode="correct"):
        self.mode = mode

    def __call__(self, X, S, mask, chain_M, residue_idx, encoding, randn,
                 use_input_decoding_order=False, decoding_order=None):
        if not use_input_decoding_order:
            decoding_order = torch.argsort((chain_M + 0.0001) * torch.abs(randn))
        # The unrelated positional-feature encoding has three axes.
        torch.nn.functional.one_hot(torch.zeros((1, S.shape[1], 2), dtype=torch.long), num_classes=66)
        if self.mode != "missing":
            actual = decoding_order.flip(-1) if self.mode == "wrong" else decoding_order
            torch.nn.functional.one_hot(actual, num_classes=S.shape[1])
        logits = torch.arange(21, dtype=torch.float32).view(1, 1, 21).repeat(1, S.shape[1], 1)
        if self.mode == "nan":
            logits[0, 0, 0] = float("nan")
        return torch.log_softmax(logits, dim=-1)


class ScoringTests(unittest.TestCase):
    def setUp(self):
        self.features = {"X": torch.zeros((1, 4, 4, 3)), "mask": torch.ones((1, 4)),
                         "chain_M": torch.ones((1, 4)), "residue_idx": torch.arange(4)[None, :],
                         "chain_encoding_all": torch.ones((1, 4), dtype=torch.long)}
        self.order = {"decoding_order_one_based": [3, 1, 4, 2]}

    def test_actual_order_observed_and_original_torch_function_restored(self):
        original = torch.nn.functional.one_hot
        result = scorer.score_sequence(FakeModel(), self.features, "ACDE", self.order, torch)
        expected = -torch.log_softmax(torch.arange(21, dtype=torch.float32), dim=0)[:4].numpy()
        self.assertTrue((result == expected).all())
        self.assertIs(torch.nn.functional.one_hot, original)

    def test_wrong_missing_and_malformed_orders_fail(self):
        original = torch.nn.functional.one_hot
        for mode in ("wrong", "missing"):
            with self.assertRaises(ValueError):
                scorer.score_sequence(FakeModel(mode), self.features, "ACDE", self.order, torch)
            self.assertIs(torch.nn.functional.one_hot, original)
        with self.assertRaises(ValueError):
            scorer.score_sequence(FakeModel(), self.features, "ACDE", {"decoding_order_one_based": [1, 1, 2, 3]}, torch)

    def test_nonfinite_model_outputs_fail(self):
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            scorer.score_sequence(FakeModel("nan"), self.features, "ACDE", self.order, torch)

    def test_null_presentation_and_rank_derived_orders(self):
        first = scorer.score_sequence(FakeModel(), self.features, "ACDE", self.order, torch)
        scorer.score_sequence(FakeModel(), self.features, "ADCE", self.order, torch)
        repeated = scorer.score_sequence(FakeModel(), self.features, "ACDE", self.order, torch)
        implicit = scorer.score_sequence(FakeModel(), self.features, "ACDE", self.order, torch, explicit=False)
        self.assertTrue((first == repeated).all())
        self.assertTrue((first == implicit).all())

    def test_summary_keeps_signed_gaps_and_half_credit_ties(self):
        template = dict(zip(scorer.SUMMARY_FIELDS[:11], ["p1", "fixture", "A", "development", "g", "patch1", "vanilla", "v_48_020", 4, 2, 2]))
        rows = []
        for gap in (-0.2, 0.0, 0.4):
            row = copy.deepcopy(template)
            for region in ("full", "patch"):
                row["original_" + region + "_nll"] = 1.0
                row["permuted_" + region + "_nll"] = 1.0 + gap
                row[region + "_nll_gap"] = gap
            rows.append(row)
        result = scorer.summarize_rows(rows)[0]
        self.assertAlmostEqual(result["full_nll_gap_mean"], 0.2 / 3)
        self.assertEqual(result["fraction_native_preferred"], 0.5)
        self.assertEqual(result["full_nll_gap_min"], -0.2)


if __name__ == "__main__":
    unittest.main()

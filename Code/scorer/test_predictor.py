"""Contract tests using a tiny decoder, never synthetic experimental training data."""
import sys
import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch import nn
from scorer.predictor import FrozenDecoder, StabilityHead, pair_predictions, weight_digest, score_swapper_candidates, load_backbone
from scorer.data import fixed_orders, normalize_ddg, load_dataset
from baseline.run_baseline import sha256
from scorer.train import metrics
from scorer import train


class TinyDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(21, 8)
        self.W_out = nn.Linear(8, 21)
        self.dropout = nn.Dropout(0.9)
        self.orders = []
        self.hidden = None

    def forward(self, X, S, mask, chain_M, residue_idx, encoding, noise,
                use_input_decoding_order, decoding_order):
        assert not self.training and not torch.is_grad_enabled()
        assert use_input_decoding_order and not noise.any() and self.augment_eps == 0
        self.orders.append(decoding_order.clone())
        # Causal context makes a swapped identity's features depend on its placement.
        embedded = self.embedding(S)
        rank = torch.argsort(decoding_order, dim=1)
        preceding = rank[:, :, None] > rank[:, None, :]
        hidden = embedded + torch.bmm(preceding.float(), embedded) / S.shape[1]
        self.hidden = self.dropout(hidden)
        return self.W_out(self.hidden).log_softmax(-1)


def features():
    ones = torch.ones(1, 4)
    return {"X": torch.zeros(1, 4, 4, 3), "mask": ones, "chain_M": ones,
            "residue_idx": torch.arange(4)[None], "chain_encoding_all": ones.long()}


class PredictorTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.decoder = TinyDecoder()
        self.extractor = FrozenDecoder(self.decoder)
        self.head = StabilityHead(self.extractor.width, 12)
        self.orders = fixed_orders(4, 3)

    def test_exact_projection_input_identity_and_shared_order(self):
        vector, mask = self.extractor(features(), ["ACDE", "CADE"], self.orders[0])
        self.assertTrue(torch.equal(vector[..., :8], self.decoder.hidden))
        self.assertEqual(vector[0, 0, 8:].argmax().item(), 0)
        self.assertEqual(vector[1, 0, 8:].argmax().item(), 1)
        self.assertFalse(vector.requires_grad)
        self.assertTrue(torch.equal(self.decoder.orders[0][0], self.decoder.orders[0][1]))
        self.extractor.train(True)
        self.assertFalse(self.decoder.training)

    def test_training_updates_head_only_and_pair_antisymmetry(self):
        before = weight_digest(self.decoder)
        previous = weight_digest(self.head)
        optimizer = torch.optim.Adam(self.head.parameters(), lr=0.01)
        pred = pair_predictions(self.extractor, self.head, features(), "ACDE", ["CADE"], self.orders)
        (pred - 1).square().mean().backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in self.head.parameters()))
        self.assertTrue(all(p.grad is None for p in self.decoder.parameters()))
        optimizer.step()
        self.assertEqual(before, weight_digest(self.decoder))
        self.assertNotEqual(previous, weight_digest(self.head))
        forward = pair_predictions(self.extractor, self.head, features(), "ACDE", ["CADE"], self.orders)
        reverse = pair_predictions(self.extractor, self.head, features(), "CADE", ["ACDE"], self.orders)
        self.assertTrue(torch.allclose(forward, -reverse))
        zero = pair_predictions(self.extractor, self.head, features(), "ACDE", ["ACDE"], self.orders)
        self.assertAlmostEqual(zero.item(), 0, places=6)

    def test_masked_pool_and_order_average(self):
        vectors = torch.randn(1, 4, self.extractor.width)
        mask = torch.tensor([[1., 1., 0., 0.]])
        score = self.head(vectors, mask)
        vectors[:, 2:] = 1000
        self.assertTrue(torch.equal(score, self.head(vectors, mask)))
        values = [pair_predictions(self.extractor, self.head, features(), "ACDE", ["CADE"], [o]) for o in self.orders]
        self.assertTrue(torch.allclose(torch.stack(values).mean(0),
            pair_predictions(self.extractor, self.head, features(), "ACDE", ["CADE"], self.orders)))

    def test_swapper_and_invalid_sequences(self):
        candidate = {"pair_id": "x", "original_sequence": "ACDE", "permuted_sequence": "CADE", "patch_indices": [1, 2]}
        result = score_swapper_candidates(self.extractor, self.head, features(), "ACDE", [candidate], self.orders)
        self.assertEqual(result[0]["pair_id"], "x")
        with self.assertRaises(ValueError):
            self.extractor(features(), ["ACD"], self.orders[0])
        with self.assertRaises(ValueError):
            self.extractor(features(), ["ACDE"], [0, 0, 2, 3])

    def test_labels_and_metrics(self):
        self.assertAlmostEqual(normalize_ddg(4.184, "positive_stabilizing", "kJ/mol"), -1)
        with self.assertRaises(ValueError):
            normalize_ddg("-", "positive_stabilizing", "kcal/mol")
        report = metrics([{"parent_id": "p", "target": x, "prediction": x + 1} for x in (0, 1, 2)])
        self.assertEqual(report["mae"], 1)
        self.assertEqual(report["rmse"], 1)
        self.assertEqual(report["median_within_protein_spearman"], 1)

    def test_dataset_alignment_provenance_and_split_leakage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdb = root / "parent.pdb"
            lines = []
            for i, residue in enumerate(("ALA", "CYS", "ASP", "GLU"), 1):
                for a, atom in enumerate(("N", "CA", "C", "O")):
                    lines.append(f"ATOM  {len(lines)+1:5d} {atom:^4s} {residue} A{i:4d}    {float(i*4+a):8.3f}{0.:8.3f}{0.:8.3f}{1.:6.2f}{0.:6.2f}           C")
            pdb.write_text("\n".join(lines + ["END"]) + "\n")
            parent = {"parent_id": "p", "homology_group": "g", "split": "train",
                      "reference_sequence": "ACDE", "pdb_path": "parent.pdb", "pdb_sha256": sha256(pdb), "chain": "A"}
            raw = {"schema_version": 1, "homology": {"method": "audit", "parameters": {"identity": 0.3}, "audit_source": "fixture"},
                   "quality_policy": "fixture only", "label_source": "unit-test fixture, not biological evidence",
                   "units": "kcal/mol", "sign_convention": "positive_stabilizing", "parents": [parent],
                   "pairs": [{"pair_id": "x", "parent_id": "p", "reference_sequence": "ACDE", "variant_sequence": "CADE",
                              "label_kind": "experimental_ddg", "measurement_id": "fixture", "ddg": 1}]}
            manifest = root / "data.json"
            def load():
                manifest.write_text(json.dumps(raw))
                return load_dataset(manifest)
            _, parents, pairs = load()
            self.assertEqual(len(parents["p"]["coordinates"]), 4)
            self.assertTrue(pairs[0]["is_swap"])
            self.assertEqual(pairs[0]["target"], -1)
            raw["parents"].append(dict(parent, parent_id="q", split="test"))
            with self.assertRaisesRegex(ValueError, "Homology group"):
                load()
            raw["parents"].pop()
            parent["reference_sequence"] = "AAAA"
            with self.assertRaisesRegex(ValueError, "align"):
                load()

    def test_training_artifacts_and_restore(self):
        # Only pipeline plumbing is tested here; these fixture labels are not evidence.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("data.json", "checkpoint.pt", "protein_mpnn_utils.py"):
                (root / name).write_text("fixture")
            parents, pairs = {}, []
            for pid, split in (("p", "train"), ("v", "validation"), ("t", "test")):
                parents[pid] = {"parent_id": pid, "reference_sequence": "ACDE", "split": split,
                                "homology_group": pid, "coordinates": features()["X"][0].tolist(), "mapping": []}
                pairs.append({"pair_id": pid, "parent_id": pid, "reference_sequence": "ACDE",
                              "variant_sequence": "CADE", "target": 1., "split": split, "is_swap": True})
            raw = {"homology": {"method": "test"}, "parents": list(parents.values())}
            args = SimpleNamespace(data=root / "data.json", checkpoint=root / "checkpoint.pt", mpnn_dir=root,
                                   device="cpu", seed=7, hidden_width=12, lr=0.01, orders=2, epochs=2, output=root / "run")
            with patch.object(train, "load_dataset", return_value=(raw, parents, pairs)), patch.object(
                    train, "load_backbone", return_value=(self.extractor, None)):
                train.run(args)
                extractor, head, config = train.load_predictor(args.output / "head.pt", root, args.checkpoint)
            self.assertEqual(config["backbone_digest_before"], config["backbone_digest_after"])
            self.assertTrue(config["head_nonzero_gradient_verified"])
            self.assertTrue((args.output / "splits.json").exists())
            report = json.loads((args.output / "test_metrics.json").read_text())
            self.assertEqual(report["composition_preserving_swaps"]["count"], 1)


@unittest.skipUnless(os.environ.get("PROTEINMPNN_DIR") and os.environ.get("PROTEINMPNN_CHECKPOINT"),
                     "Set PROTEINMPNN_DIR and PROTEINMPNN_CHECKPOINT for real-checkpoint verification")
class CheckpointIntegrationTests(unittest.TestCase):
    def test_projection_capture_preserves_actual_forward_and_weights(self):
        extractor, _ = load_backbone(os.environ["PROTEINMPNN_DIR"], os.environ["PROTEINMPNN_CHECKPOINT"])
        length = 64
        fixture = features()
        ones = torch.ones(1, length)
        fixture = {"X": torch.randn(1, length, 4, 3), "mask": ones, "chain_M": ones,
                   "residue_idx": torch.arange(length)[None], "chain_encoding_all": ones.long()}
        sequence = ("ACDEFGHIKLMNPQRSTVWY" * 4)[:length]
        tokens = torch.tensor([[extractor.vocabulary.index(a) for a in sequence]])
        order = list(range(length))
        before = weight_digest(extractor.model)
        vectors, mask = extractor(fixture, [sequence], order)
        with torch.no_grad():
            reconstructed = extractor.model.W_out(vectors[..., :extractor.model.W_out.in_features]).log_softmax(-1)
            original = extractor.model(fixture["X"], tokens, ones, ones, fixture["residue_idx"],
                                       fixture["chain_encoding_all"], torch.zeros_like(ones),
                                       use_input_decoding_order=True, decoding_order=torch.tensor([order]))
        self.assertTrue(torch.allclose(reconstructed, original, atol=1e-6))
        head = StabilityHead(extractor.width)
        head(vectors, mask).square().mean().backward()
        self.assertTrue(any(p.grad is not None for p in head.parameters()))
        self.assertTrue(all(p.grad is None for p in extractor.parameters()))
        self.assertEqual(before, weight_digest(extractor.model))


if __name__ == "__main__":
    unittest.main()

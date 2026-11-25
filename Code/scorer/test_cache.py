"""Check cache reuse/invalidation and batched prediction/gradient equivalence."""
from pathlib import Path
import sys
import os
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from scorer.cache import FeatureCache, cached_predictions, precache
from scorer.data import backbone_features
from scorer.predictor import FrozenDecoder, StabilityHead, pair_predictions, weight_digest, load_backbone
from scorer.train import evaluate
from test_predictor import TinyDecoder, features


class CacheTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.extractor = FrozenDecoder(TinyDecoder())
        self.head = StabilityHead(self.extractor.width, 12)

    def test_resume_deduplicate_batch_and_invalidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            order = [0, 1, 2, 3]
            identity = {"weights": weight_digest(self.extractor.model)}
            cache = FeatureCache(tmp, self.extractor, identity, feature_batch_size=2, memory_entries=0)
            result = cache.get_many(features(), ["ACDE", "CADE", "ACDE", "DEAC"], order)
            self.assertEqual(cache.stats["misses"], 3)
            self.assertEqual(cache.stats["extractor_calls"], 2)
            self.assertTrue(torch.equal(result[0][0], result[2][0]))
            self.assertFalse(result[0][0].requires_grad)
            cache.close()
            cache = FeatureCache(tmp, self.extractor, identity, feature_batch_size=2)
            restored = cache.get_many(features(), ["ACDE", "CADE"], order)
            self.assertEqual(cache.stats["extractor_calls"], 0)
            self.assertTrue(torch.equal(result[1][0], restored[1][0]))
            cache.get_many(features(), ["ACDE"], list(reversed(order)))
            altered = features()
            altered["X"][0, 0, 0, 0] = 1
            cache.get_many(altered, ["ACDE"], order)
            self.assertEqual(cache.stats["misses"], 2)
            old_path = cache.path
            cache.close()
            different = FeatureCache(tmp, self.extractor, {"weights": "other_checkpoint"})
            self.assertNotEqual(different.path, old_path)
            different.close()

    def test_variable_length_batch_predictions_gradients_and_warm_evaluation(self):
        parents = {
            "p": {"reference_sequence": "ACDE", "coordinates": features()["X"][0].tolist()},
            "q": {"reference_sequence": "ACDEFG", "coordinates": torch.zeros(6, 4, 3).tolist()},
        }
        pairs = [
            {"pair_id": "a", "parent_id": "p", "reference_sequence": "ACDE", "variant_sequence": "CADE", "target": 1., "is_swap": True},
            {"pair_id": "b", "parent_id": "q", "reference_sequence": "ACDEFG", "variant_sequence": "CADEFG", "target": -1., "is_swap": True},
            {"pair_id": "c", "parent_id": "p", "reference_sequence": "ACDE", "variant_sequence": "DEAC", "target": .3, "is_swap": True},
        ]
        orders = {pid: [list(range(len(p["reference_sequence"]))), list(reversed(range(len(p["reference_sequence"]))))] for pid, p in parents.items()}
        indices = [0, 1, 0]
        direct = torch.cat([pair_predictions(self.extractor, self.head, backbone_features(parents[p["parent_id"]], "cpu"),
                           p["reference_sequence"], [p["variant_sequence"]], [orders[p["parent_id"]][i]]) for p, i in zip(pairs, indices)])
        direct.square().mean().backward()
        gradients = [p.grad.clone() for p in self.head.parameters()]
        self.head.zero_grad(set_to_none=True)
        before = weight_digest(self.extractor.model)
        with tempfile.TemporaryDirectory() as tmp:
            cache = FeatureCache(tmp, self.extractor, {"weights": before}, feature_batch_size=2)
            cached = cached_predictions(cache, self.head, parents, pairs, orders, indices, "cpu")
            self.assertTrue(torch.allclose(direct, cached, atol=1e-6))
            cached.square().mean().backward()
            for expected, parameter in zip(gradients, self.head.parameters()):
                self.assertTrue(torch.allclose(expected, parameter.grad, atol=1e-6))
            precache(cache, parents, pairs, orders, "cpu")
            calls = cache.stats["extractor_calls"]
            expected = evaluate(self.extractor, self.head, parents, pairs, orders, "cpu")
            actual = evaluate(self.extractor, self.head, parents, pairs, orders, "cpu", cache, batch_size=2)
            self.assertEqual(cache.stats["extractor_calls"], calls)
            for left, right in zip(expected["predictions"], actual["predictions"]):
                self.assertAlmostEqual(left["prediction"], right["prediction"], places=6)
            cache.close()
        self.assertEqual(before, weight_digest(self.extractor.model))
        self.assertTrue(all(p.grad is None for p in self.extractor.parameters()))

    def test_corruption_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = FeatureCache(tmp, self.extractor, {}, memory_entries=0)
            cache.get_many(features(), ["ACDE"], [0, 1, 2, 3])
            cache.db.execute("UPDATE features SET digest='corrupted'")
            cache.db.commit()
            with self.assertRaisesRegex(ValueError, "Corrupt"):
                cache.get_many(features(), ["ACDE"], [0, 1, 2, 3])
            cache.close()


@unittest.skipUnless(os.environ.get("PROTEINMPNN_DIR") and os.environ.get("PROTEINMPNN_CHECKPOINT"),
                     "Set checkpoint environment variables for real cached-feature integration")
class RealCacheTests(unittest.TestCase):
    def test_real_batched_cache_matches_direct_checkpoint_and_reuses_disk(self):
        extractor, _ = load_backbone(os.environ["PROTEINMPNN_DIR"], os.environ["PROTEINMPNN_CHECKPOINT"])
        before = weight_digest(extractor.model)
        sequence = ("ACDEFGHIKLMNPQRSTVWY" * 4)[:64]
        variant = sequence[1] + sequence[0] + sequence[2:]
        torch.manual_seed(17)
        parent = {"reference_sequence": sequence, "coordinates": torch.randn(64, 4, 3).tolist()}
        backbone = backbone_features(parent, "cpu")
        order = list(range(64))
        head = StabilityHead(extractor.width)
        direct = pair_predictions(extractor, head, backbone, sequence, [variant], [order])
        with tempfile.TemporaryDirectory() as tmp:
            cache = FeatureCache(tmp, extractor, {"weights": before}, feature_batch_size=2)
            vectors = cache.get_many(backbone, [sequence, variant], order)
            scores = head(torch.stack([v[0] for v in vectors]), torch.stack([v[1] for v in vectors]))
            self.assertTrue(torch.allclose(direct, scores[1:] - scores[:1], atol=1e-6))
            scores.square().mean().backward()
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in head.parameters()))
            cache.close()
            reopened = FeatureCache(tmp, extractor, {"weights": before})
            reopened.get_many(backbone, [sequence, variant], order)
            self.assertEqual(reopened.stats["extractor_calls"], 0)
            reopened.close()
        self.assertEqual(before, weight_digest(extractor.model))


if __name__ == "__main__":
    unittest.main()

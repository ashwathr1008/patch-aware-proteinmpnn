"""Check the patch feature layout, and that a target position is always decoded last.

The decoding-order invariant is the one that cannot be checked by inspection. A feature read
at a position only means "conditioned on the rest of the patch" if that position is decoded
after its partners; if it slips earlier, the module returns a plausible-looking number that
answers a different question. The stub model here records the orders it is handed, so the
invariant is asserted rather than assumed, including for noise chosen to break the weighted
argsort the module used previously.
"""
import unittest

import torch

import features


def make_features(length, device="cpu"):
    ones = torch.ones(1, length, device=device)
    return {"X": torch.zeros(1, length, 4, 3, device=device), "mask": ones,
            "chain_M": ones, "residue_idx": torch.arange(length, device=device)[None, :],
            "chain_encoding_all": ones}


class StubModel:
    """Returns log probabilities that encode (token, position, letter), and records its calls."""

    def __init__(self, nonfinite=False):
        self.orders = []
        self.batch_sizes = []
        self.nonfinite = nonfinite

    def __call__(self, X, tokens, mask, chain_M, residue_idx, encoding, bias,
                 use_input_decoding_order=False, decoding_order=None):
        self.orders.append(decoding_order[0].tolist())
        self.batch_sizes.append(int(tokens.shape[0]))
        batch, length = tokens.shape
        letters = torch.arange(len(features.ALPHABET), dtype=torch.float64)
        base = (tokens.to(torch.float64)[:, :, None] * 0.1
                + torch.arange(length, dtype=torch.float64)[None, :, None] * 0.01
                + letters[None, None, :] * 0.001)
        if self.nonfinite:
            base[0, 0, 0] = float("nan")
            return base
        return torch.log_softmax(base, dim=-1)

    def expected(self, sequence, position):
        token = features.ALPHABET.index(sequence[position])
        letters = torch.arange(len(features.ALPHABET), dtype=torch.float64)
        row = token * 0.1 + position * 0.01 + letters * 0.001
        return torch.log_softmax(row, dim=-1).tolist()


class TestWidth(unittest.TestCase):

    def test_width_counts_logprobs_identity_and_neighbours(self):
        alphabet = len(features.ALPHABET)
        self.assertEqual(features.feature_width(2), 2 * 2 * alphabet)
        self.assertEqual(features.feature_width(2, 3), 2 * 2 * alphabet + 3 * alphabet)
        self.assertEqual(features.feature_width(0), 0)


class TestFeatureVector(unittest.TestCase):

    def setUp(self):
        self.alphabet = len(features.ALPHABET)
        self.sequence = "ACDEFG"
        self.logprobs = {1: [0.5] * self.alphabet, 3: [-0.25] * self.alphabet}

    def test_layout_is_logprobs_then_identity_per_member(self):
        vector = features.patch_feature_vector(self.logprobs, self.sequence, [1, 3])
        self.assertEqual(len(vector), features.feature_width(2))
        self.assertEqual(vector[:self.alphabet], [0.5] * self.alphabet)
        identity = vector[self.alphabet:2 * self.alphabet]
        self.assertEqual(sum(identity), 1.0)
        self.assertEqual(identity.index(1.0), features.ALPHABET.index("C"))
        self.assertEqual(vector[2 * self.alphabet:3 * self.alphabet], [-0.25] * self.alphabet)
        second = vector[3 * self.alphabet:]
        self.assertEqual(second.index(1.0), features.ALPHABET.index("E"))

    def test_member_order_is_the_order_given(self):
        forward = features.patch_feature_vector(self.logprobs, self.sequence, [1, 3])
        reverse = features.patch_feature_vector(self.logprobs, self.sequence, [3, 1])
        self.assertNotEqual(forward, reverse)

    def test_neighbours_contribute_identity_only(self):
        vector = features.patch_feature_vector(self.logprobs, self.sequence, [1], neighbours=[4, 5])
        self.assertEqual(len(vector), features.feature_width(1, 2))
        tail = vector[2 * self.alphabet:]
        self.assertEqual(tail[:self.alphabet].index(1.0), features.ALPHABET.index("F"))
        self.assertEqual(tail[self.alphabet:].index(1.0), features.ALPHABET.index("G"))

    def test_missing_position_is_refused(self):
        with self.assertRaises(ValueError):
            features.patch_feature_vector(self.logprobs, self.sequence, [2])


class TestConditionalLogprobs(unittest.TestCase):

    def setUp(self):
        self.length = 12
        self.features = make_features(self.length)
        self.sequences = ["ACDEFGHIKLMN", "NMLKIHGFEDCA", "AAAAAAAAAAAA"]

    def test_target_position_is_decoded_last(self):
        model = StubModel()
        features.conditional_logprobs_at(model, self.features, self.sequences[:1], [0, 5, 11],
                                         torch)
        self.assertEqual([order[-1] for order in model.orders], [0, 5, 11])
        for order in model.orders:
            self.assertEqual(sorted(order), list(range(self.length)))

    def test_target_is_last_even_when_its_own_noise_is_tiny(self):
        """The weighted argsort this replaced would rank the target early for such noise."""
        model = StubModel()
        original = features._ordering_noise

        def tiny(torch_module, length, device, seed, tag):
            noise = torch.full((1, length), 50.0)
            noise[0, 4] = 1e-9
            return noise

        features._ordering_noise = tiny
        try:
            features.conditional_logprobs_at(model, self.features, self.sequences[:1], [4], torch)
        finally:
            features._ordering_noise = original
        self.assertEqual(model.orders[0][-1], 4)

    def test_values_match_the_model_at_each_position(self):
        model = StubModel()
        out = features.conditional_logprobs_at(model, self.features, self.sequences, [2, 7], torch)
        self.assertEqual(len(out), len(self.sequences))
        for sequence, row in zip(self.sequences, out):
            self.assertEqual(sorted(row), [2, 7])
            for position in (2, 7):
                for got, want in zip(row[position], model.expected(sequence, position)):
                    self.assertAlmostEqual(got, want, places=10)

    def test_batching_does_not_change_the_result(self):
        whole = features.conditional_logprobs_at(StubModel(), self.features, self.sequences,
                                                 [3], torch, batch_size=64)
        split = StubModel()
        pieces = features.conditional_logprobs_at(split, self.features, self.sequences,
                                                  [3], torch, batch_size=1)
        self.assertEqual(whole, pieces)
        self.assertEqual(split.batch_sizes, [1, 1, 1])

    def test_order_is_stable_across_calls_and_varies_with_seed(self):
        first, second, reseeded = StubModel(), StubModel(), StubModel()
        for model, seed in ((first, 0), (second, 0), (reseeded, 1)):
            features.conditional_logprobs_at(model, self.features, self.sequences[:1], [6],
                                             torch, seed=seed)
        self.assertEqual(first.orders, second.orders)
        self.assertNotEqual(first.orders, reseeded.orders)

    def test_empty_sequence_list_returns_nothing(self):
        self.assertEqual(features.conditional_logprobs_at(StubModel(), self.features, [],
                                                          [0], torch), [])

    def test_position_outside_the_chain_is_refused(self):
        for position in (-1, self.length):
            with self.assertRaises(ValueError):
                features.conditional_logprobs_at(StubModel(), self.features,
                                                 self.sequences[:1], [position], torch)

    def test_nonfinite_output_is_refused(self):
        with self.assertRaises(ValueError):
            features.conditional_logprobs_at(StubModel(nonfinite=True), self.features,
                                             self.sequences[:1], [0], torch)


if __name__ == "__main__":
    unittest.main()

"""Check the properties the spec names, especially the ones comparisons depend on.

The load-bearing test is `test_local_equals_whole_chain_when_the_mask_covers_everything`. The
spec's whole-chain-versus-local contrast only measures the pooling region if the two heads are
otherwise the same computation; if they had drifted apart, the comparison would silently be
about two architectures instead. Copied weights plus a full mask must give identical outputs.

The rest guard padding. A padded position that leaks into a mean or into attention makes a
short chain's score depend on the cohort's longest chain, which no downstream check would
catch.
"""
import unittest

import torch

import three_heads as th


WIDTH = 149


def states(batch, length, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(batch, length, WIDTH, generator=generator)


class TestMaskedMean(unittest.TestCase):

    def test_ignores_masked_positions(self):
        values = torch.tensor([[[1.0], [100.0]]])
        mask = torch.tensor([[1.0, 0.0]])
        self.assertAlmostEqual(float(th.masked_mean(values, mask)[0, 0]), 1.0)

    def test_averages_the_unmasked_positions(self):
        values = torch.tensor([[[1.0], [3.0], [100.0]]])
        mask = torch.tensor([[1.0, 1.0, 0.0]])
        self.assertAlmostEqual(float(th.masked_mean(values, mask)[0, 0]), 2.0)

    def test_empty_region_is_refused_not_divided_by_zero(self):
        with self.assertRaises(ValueError):
            th.masked_mean(torch.ones(1, 3, 1), torch.zeros(1, 3))


class TestPooledHead(unittest.TestCase):

    def test_output_has_no_bias(self):
        head = th.build("local_mean", WIDTH)
        self.assertIsNone(head.output.bias)

    def test_zeroed_input_gives_zero_through_a_bias_free_output(self):
        head = th.build("local_mean", WIDTH)
        with torch.no_grad():
            for layer in head.residue_mlp.net:
                if hasattr(layer, "bias") and layer.bias is not None:
                    layer.bias.zero_()
        mask = torch.ones(2, 5)
        out = head(torch.zeros(2, 5, WIDTH), mask)
        self.assertTrue(torch.allclose(out, torch.zeros(2), atol=1e-6))

    def test_padding_does_not_change_the_result(self):
        """A short chain scored alone and scored inside a padded batch must agree."""
        head = th.build("local_mean", WIDTH).eval()
        short = states(1, 4, seed=1)
        padded = torch.zeros(1, 10, WIDTH)
        padded[:, :4] = short
        with torch.no_grad():
            a = head(short, torch.ones(1, 4))
            b = head(padded, torch.cat([torch.ones(1, 4), torch.zeros(1, 6)], dim=1))
        self.assertAlmostEqual(float(a[0]), float(b[0]), places=5)

    def test_reordering_the_region_does_not_change_the_result(self):
        """A mean is order-free, so token order inside the region must not matter."""
        head = th.build("local_mean", WIDTH).eval()
        x = states(1, 6, seed=2)
        mask = torch.tensor([[1.0, 0.0, 1.0, 1.0, 0.0, 0.0]])
        shuffled = x.clone()
        shuffled[0, [0, 2, 3]] = x[0, [3, 0, 2]]
        with torch.no_grad():
            self.assertAlmostEqual(float(head(x, mask)[0]),
                                   float(head(shuffled, mask)[0]), places=5)


class TestWholeChainVersusLocal(unittest.TestCase):

    def test_local_equals_whole_chain_when_the_mask_covers_everything(self):
        whole = th.build("whole_chain_mean", WIDTH).eval()
        local = th.build("local_mean", WIDTH).eval()
        local.load_state_dict(whole.state_dict())
        x = states(3, 7, seed=3)
        mask = torch.ones(3, 7)
        with torch.no_grad():
            self.assertTrue(torch.allclose(whole(x, mask), local(x, mask), atol=1e-6))

    def test_they_differ_once_the_masks_differ(self):
        whole = th.build("whole_chain_mean", WIDTH).eval()
        local = th.build("local_mean", WIDTH).eval()
        local.load_state_dict(whole.state_dict())
        x = states(2, 8, seed=4)
        full = torch.ones(2, 8)
        narrow = torch.zeros(2, 8)
        narrow[:, :2] = 1.0
        with torch.no_grad():
            self.assertFalse(torch.allclose(whole(x, full), local(x, narrow), atol=1e-4))

    def test_parameter_counts_match(self):
        whole = th.build("whole_chain_mean", WIDTH)
        local = th.build("local_mean", WIDTH)
        self.assertEqual(sum(p.numel() for p in whole.parameters()),
                         sum(p.numel() for p in local.parameters()))


class TestAttentionHead(unittest.TestCase):

    def test_padding_does_not_change_the_result(self):
        head = th.build("local_attention", WIDTH).eval()
        short = states(1, 4, seed=5)
        padded = torch.zeros(1, 11, WIDTH)
        padded[:, :4] = short
        with torch.no_grad():
            a = head(short, torch.ones(1, 4))
            b = head(padded, torch.cat([torch.ones(1, 4), torch.zeros(1, 7)], dim=1))
        self.assertAlmostEqual(float(a[0]), float(b[0]), places=4)

    def test_attention_makes_it_differ_from_the_plain_local_mean(self):
        attention = th.build("local_attention", WIDTH).eval()
        mean = th.build("local_mean", WIDTH).eval()
        mean.residue_mlp.load_state_dict(attention.residue_mlp.state_dict())
        mean.output.load_state_dict(attention.output.state_dict())
        x = states(2, 6, seed=6)
        mask = torch.ones(2, 6)
        with torch.no_grad():
            self.assertFalse(torch.allclose(attention(x, mask), mean(x, mask), atol=1e-4))

    def test_has_more_parameters_than_the_mean_head(self):
        self.assertGreater(sum(p.numel() for p in th.build("local_attention", WIDTH).parameters()),
                           sum(p.numel() for p in th.build("local_mean", WIDTH).parameters()))

    def test_rejects_a_head_count_that_does_not_divide_the_width(self):
        with self.assertRaises(ValueError):
            th.build("local_attention", WIDTH, hidden=64, heads=5)

    def test_single_token_region_is_well_defined(self):
        head = th.build("local_attention", WIDTH).eval()
        x = states(1, 5, seed=7)
        mask = torch.zeros(1, 5)
        mask[0, 2] = 1.0
        with torch.no_grad():
            value = head(x, mask)
        self.assertTrue(torch.isfinite(value).all())


class TestRegionMask(unittest.TestCase):

    def test_whole_chain_covers_exactly_the_real_residues(self):
        mask = th.region_mask([[0, 1]], [5], 8, "whole_chain_mean")
        self.assertEqual(mask[0].tolist(), [1, 1, 1, 1, 1, 0, 0, 0])

    def test_local_covers_only_the_region(self):
        mask = th.region_mask([[0, 4, 2]], [6], 8, "local_mean")
        self.assertEqual(mask[0].tolist(), [1, 0, 1, 0, 1, 0, 0, 0])

    def test_positions_past_the_chain_are_dropped(self):
        mask = th.region_mask([[0, 9]], [4], 8, "local_mean")
        self.assertEqual(mask[0].tolist(), [1, 0, 0, 0, 0, 0, 0, 0])

    def test_attention_head_uses_the_same_local_region(self):
        a = th.region_mask([[1, 3]], [5], 6, "local_mean")
        b = th.region_mask([[1, 3]], [5], 6, "local_attention")
        self.assertEqual(a.tolist(), b.tolist())


if __name__ == "__main__":
    unittest.main()


class TestFrozenBaselineIsNested(unittest.TestCase):
    """The whole-chain head must be able to reproduce the frozen score exactly.

    This is the point of adding the log p(placed residue) channel. Before it, "the head loses
    to frozen" was not clean evidence about capacity or locality, because the head had to
    rediscover a log-softmax composed with a pretrained 128->21 projection from 46k contrasts.
    With the channel present the frozen score is a special case of the head, so any remaining
    gap is interpretable.

    The construction below is the ReLU identity trick: ReLU(x + B) - ReLU(-x + B) = 2x for B
    larger than |x|, which lets two ReLU layers carry a signed linear function through. The
    additive constants survive into each score and then cancel in the contrast, which is why
    the claim is about q rather than about an individual score.
    """

    WIDTH = 173
    CHANNEL = 170  # log p(the residue actually placed)

    def linear_readout_head(self, hidden=64, gain=0.5, offset=50.0):
        head = th.build("whole_chain_mean", self.WIDTH, hidden)
        with torch.no_grad():
            first, second = head.residue_mlp.net[0], head.residue_mlp.net[2]
            first.weight.zero_(); first.bias.zero_()
            first.weight[0, self.CHANNEL] = 1.0
            first.bias[0] = offset
            first.weight[1, self.CHANNEL] = -1.0
            first.bias[1] = offset
            second.weight.zero_(); second.bias.zero_()
            second.weight[0, 0] = 1.0
            second.weight[0, 1] = -1.0
            second.bias[0] = offset
            head.output.weight.zero_()
            head.output.weight[0, 0] = gain
        return head.eval()

    def test_contrast_equals_the_mean_logprob_difference(self):
        head = self.linear_readout_head(gain=0.5)
        generator = torch.Generator().manual_seed(11)
        a = torch.randn(4, 9, self.WIDTH, generator=generator)
        b = torch.randn(4, 9, self.WIDTH, generator=generator)
        mask = torch.ones(4, 9)
        with torch.no_grad():
            q = head(a, mask) - head(b, mask)
        # gain 0.5 against the trick's factor of two gives exactly the mean difference,
        # which is the frozen contrast nll(B) - nll(A).
        expected = a[:, :, self.CHANNEL].mean(dim=1) - b[:, :, self.CHANNEL].mean(dim=1)
        self.assertTrue(torch.allclose(q, expected, atol=1e-4),
                        msg="%s vs %s" % (q.tolist(), expected.tolist()))

    def test_it_respects_padding_like_the_baseline_does(self):
        head = self.linear_readout_head(gain=0.5)
        generator = torch.Generator().manual_seed(12)
        a = torch.randn(1, 12, self.WIDTH, generator=generator)
        b = torch.randn(1, 12, self.WIDTH, generator=generator)
        mask = torch.cat([torch.ones(1, 5), torch.zeros(1, 7)], dim=1)
        with torch.no_grad():
            q = head(a, mask) - head(b, mask)
        expected = (a[0, :5, self.CHANNEL].mean() - b[0, :5, self.CHANNEL].mean())
        self.assertAlmostEqual(float(q[0]), float(expected), places=4)

    def test_identical_arrangements_still_tie(self):
        head = self.linear_readout_head()
        generator = torch.Generator().manual_seed(13)
        x = torch.randn(2, 7, self.WIDTH, generator=generator)
        mask = torch.ones(2, 7)
        with torch.no_grad():
            self.assertTrue(torch.allclose(head(x, mask) - head(x, mask),
                                           torch.zeros(2), atol=1e-6))

    def test_a_local_head_cannot_nest_the_full_chain_baseline(self):
        """Nesting holds only for the whole-chain head: a ten-residue region can at best
        reproduce a patch-restricted mean, which is a different statistic."""
        head = self.linear_readout_head()
        generator = torch.Generator().manual_seed(14)
        a = torch.randn(1, 20, self.WIDTH, generator=generator)
        b = torch.randn(1, 20, self.WIDTH, generator=generator)
        full = torch.ones(1, 20)
        narrow = torch.zeros(1, 20)
        narrow[0, :10] = 1.0
        with torch.no_grad():
            whole = head(a, full) - head(b, full)
            local = head(a, narrow) - head(b, narrow)
        self.assertFalse(torch.allclose(whole, local, atol=1e-3))


class TestContinuousChannels(unittest.TestCase):

    def test_identity_block_is_excluded_from_standardisation(self):
        import train_three_heads as t3
        channels = set(t3.continuous_channels(173))
        self.assertTrue(all(c not in channels for c in range(128, 149)))
        self.assertTrue(all(c in channels for c in range(0, 128)))
        self.assertTrue(all(c in channels for c in range(149, 173)))

    def test_nesting_channel_is_standardised(self):
        import train_three_heads as t3
        self.assertIn(t3.NESTING_CHANNEL, set(t3.continuous_channels(173)))
        self.assertEqual(t3.NESTING_CHANNEL, 170)

    def test_old_width_without_logprob_channels_still_works(self):
        import train_three_heads as t3
        channels = set(t3.continuous_channels(149))
        self.assertEqual(channels, set(range(128)))

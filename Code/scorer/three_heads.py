"""The spec's three readouts over one shared frozen decoder state.

    whole_chain_mean   residue MLP -> mean over every valid position -> scalar
    local_mean         residue MLP -> mean over the patch region only -> scalar
    local_attention    residue MLP -> one self-attention block within the region
                                   -> mean over the region -> scalar

`whole_chain_mean` and `local_mean` are the *same* module with a different pooling mask, which
is what makes their comparison a test of the pooling region rather than of two architectures.
The spec requires this explicitly, and `test_three_heads.py` asserts it: given copied weights
and a region mask covering the whole chain, the two must produce identical outputs.

`local_attention` adds four 16-channel heads, a residual connection and layer normalisation on
top of the same MLP and the same region, so its comparison against `local_mean` isolates the
attention block -- though not capacity from mechanism, since it also adds parameters. That
caveat belongs in the write-up, not in the code.

Padding is masked everywhere: a padded position is excluded from the mean and from attention
keys, and a padded query never contributes. Pooling padding would make a short chain's score
depend on the cohort's longest chain.
"""

import torch
from torch import nn
from torch.nn import functional as F

NEG_INF = -1e4  # fp16-safe additive mask


class ResidueMLP(nn.Module):
    """Two layers, 64 hidden channels, ReLU after each -- the spec's shared trunk."""

    def __init__(self, width, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(width, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU())

    def forward(self, x):
        return self.net(x)


def masked_mean(values, mask):
    """Mean over positions where mask is 1; errors rather than dividing by zero."""
    weight = mask.unsqueeze(-1).to(values.dtype)
    total = mask.sum(dim=1, keepdim=True).to(values.dtype)
    if bool((total <= 0).any()):
        raise ValueError("Cannot pool an empty region")
    return (values * weight).sum(dim=1) / total


class PooledHead(nn.Module):
    """Shared trunk and bias-free scalar projection; the mask decides what is pooled.

    A bias on the output would cancel in every contrast `score(A) - score(B)` and so could
    never be learned; leaving it out keeps the parameter count honest.
    """

    def __init__(self, width, hidden=64):
        super().__init__()
        self.residue_mlp = ResidueMLP(width, hidden)
        self.output = nn.Linear(hidden, 1, bias=False)

    def forward(self, states, mask):
        return self.output(masked_mean(self.residue_mlp(states), mask)).squeeze(-1)


class AttentionHead(nn.Module):
    """Same trunk and output, with one masked self-attention block inside the region."""

    def __init__(self, width, hidden=64, heads=4):
        super().__init__()
        if hidden % heads:
            raise ValueError("hidden must divide evenly across attention heads")
        self.residue_mlp = ResidueMLP(width, hidden)
        self.heads = heads
        self.per_head = hidden // heads
        self.query = nn.Linear(hidden, hidden)
        self.key = nn.Linear(hidden, hidden)
        self.value = nn.Linear(hidden, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.output = nn.Linear(hidden, 1, bias=False)

    def forward(self, states, mask):
        values = self.residue_mlp(states)
        batch, length, hidden = values.shape
        split = lambda t: t.view(batch, length, self.heads, self.per_head).transpose(1, 2)
        q, k, v = split(self.query(values)), split(self.key(values)), split(self.value(values))
        scores = (q @ k.transpose(-2, -1)) / (self.per_head ** 0.5)
        # Padded keys must not receive attention from any query.
        key_mask = mask[:, None, None, :].to(torch.bool)
        scores = scores.masked_fill(~key_mask, NEG_INF)
        attended = (F.softmax(scores, dim=-1) @ v)
        attended = attended.transpose(1, 2).reshape(batch, length, hidden)
        # Residual then layer norm, and padded queries are zeroed so they cannot leak
        # through the residual into the pooled mean.
        combined = self.norm(values + attended) * mask.unsqueeze(-1).to(values.dtype)
        return self.output(masked_mean(combined, mask)).squeeze(-1)


def build(name, width, hidden=64, heads=4):
    if name in ("whole_chain_mean", "local_mean"):
        return PooledHead(width, hidden)
    if name == "local_attention":
        return AttentionHead(width, hidden, heads)
    raise ValueError("Unknown head: " + name)


HEADS = ("whole_chain_mean", "local_mean", "local_attention")


def region_mask(regions, lengths, max_length, name, torch_module=torch):
    """1 where the head should pool.

    `whole_chain_mean` pools every real residue; the local heads pool only the patch and its
    recorded neighbours. The region comes from the backbone, so it is identical for both
    arrangements of a contrast and was fixed before any label was read.
    """
    batch = len(regions)
    mask = torch_module.zeros(batch, max_length)
    if name == "whole_chain_mean":
        for row, length in enumerate(lengths):
            mask[row, :int(length)] = 1.0
        return mask
    for row, (region, length) in enumerate(zip(regions, lengths)):
        for position in region:
            if 0 <= position < int(length):
                mask[row, position] = 1.0
    return mask

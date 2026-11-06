"""Sequence-conditioned features for a patch, read out of a frozen ProteinMPNN.

The encoder cannot supply these. It initialises its node state to zeros and builds edge
features from backbone geometry alone, so its output is identical for every assignment on one
backbone: a head built on encoder embeddings would see the same input for an arrangement and
its reverse. Sequence enters at the decoder, so that is where features have to come from.

`ProteinMPNN.conditional_probs` gives what is needed — p(aa at a position | backbone, every
other residue) — but it loops over every position in the chain, which is L decoder passes per
sequence when a patch needs only its own members. This does the same computation for a named
set of positions and batches across sequences, which is the difference between a tractable
experiment and an intractable one.

**Patch members are conditioned on each other, deliberately.** Masking the whole patch would
produce features that describe what the context expects at those positions, which is the same
for every arrangement of the same residues and therefore useless for ranking them. Conditioning
each member on the rest of the chain, patch members included, makes the features depend both on
the residue at that position and on what its partners became. Third-residue dependence comes
along for free, since a third site in the same network is part of the conditioning set.

Determinism: the order among non-target positions comes from a seeded generator, and the
target position is always placed last, so the same structure, sequence and position return the
same features on every run.
"""

import hashlib

ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"


def _ordering_noise(torch, length, device, seed, tag):
    """A fixed pseudo-random vector, so an ordering never depends on global RNG state."""
    digest = hashlib.sha256(("patch_features_v1|%s|%d" % (tag, seed)).encode()).digest()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int.from_bytes(digest[:8], "big") % (2 ** 63))
    return torch.randn(1, length, generator=generator).to(device)


def conditional_logprobs_at(model, features, sequences, positions, torch,
                            seed=0, batch_size=64, tag="patch"):
    """Log p(residue | backbone, all other residues) at each named position.

    Returns a list, one entry per sequence, each a dict from position index to a list of 21
    log probabilities. One decoder pass per target position per batch, rather than one per
    chain position.
    """
    if not sequences:
        return []
    length = features["X"].shape[1]
    device = features["X"].device
    for position in positions:
        if not 0 <= position < length:
            raise ValueError("Position %d outside the chain" % position)

    out = [dict() for _ in sequences]
    noise = _ordering_noise(torch, length, device, seed, tag)

    base = torch.argsort(torch.abs(noise)[0]).tolist()
    for position in positions:
        # Decode this position last, so every other residue is already visible. Built
        # explicitly rather than by weighting the noise: a weighted argsort can misplace the
        # target when its own noise is near zero, which would silently return a probability
        # that is not conditioned on the rest of the patch.
        order = [p for p in (int(v) for v in base) if p != position] + [position]
        decoding_order = torch.tensor([order], dtype=torch.long, device=device)

        for start in range(0, len(sequences), batch_size):
            chunk = sequences[start:start + batch_size]
            size = len(chunk)
            tokens = torch.tensor([[ALPHABET.index(c) for c in s] for s in chunk],
                                  dtype=torch.long, device=device)
            expand = lambda t, d: t.expand(size, *([-1] * d)).contiguous()
            X = expand(features["X"], 3)
            mask = expand(features["mask"], 1)
            chain_M = expand(features["chain_M"], 1)
            residue_idx = expand(features["residue_idx"], 1)
            encoding = expand(features["chain_encoding_all"], 1)
            orders = decoding_order.expand(size, -1).contiguous()
            with torch.inference_mode():
                log_probs = model(X, tokens, mask, chain_M, residue_idx, encoding,
                                  torch.zeros_like(mask), use_input_decoding_order=True,
                                  decoding_order=orders)
            if not bool(torch.isfinite(log_probs).all().item()):
                raise ValueError("Nonfinite log probabilities")
            at_position = log_probs[:, position, :].detach().cpu().tolist()
            for offset, row in enumerate(at_position):
                out[start + offset][position] = row
    return out


def patch_feature_vector(logprobs, sequence, positions, neighbours=()):
    """Assemble one feature vector for a patch assignment.

    For each patch member, in a fixed order: the conditional log-probability vector at that
    position, and a one-hot of the residue actually placed there. The log probabilities say
    what the rest of the structure and sequence expect; the one-hot says what is there. A
    swap changes both, which is the signal the head is meant to read.

    Neighbouring positions, when supplied, contribute their own one-hot identities so that
    three-body terms can be represented explicitly rather than having to survive pooling.
    """
    vector = []
    for position in positions:
        if position not in logprobs:
            raise ValueError("No conditional probabilities at position %d" % position)
        vector.extend(logprobs[position])
        onehot = [0.0] * len(ALPHABET)
        onehot[ALPHABET.index(sequence[position])] = 1.0
        vector.extend(onehot)
    for position in neighbours:
        onehot = [0.0] * len(ALPHABET)
        onehot[ALPHABET.index(sequence[position])] = 1.0
        vector.extend(onehot)
    return vector


def feature_width(patch_size, neighbour_count=0):
    return patch_size * 2 * len(ALPHABET) + neighbour_count * len(ALPHABET)

"""Group related sequences before splitting, so patches cannot leak across the split.

Why not percent identity. The usual recipe clusters at a fixed identity threshold, around
30%. That recipe assumes sequences long enough for percent identity to carry information, and
it fails badly on the small domains this project studies. Among the seven pilot structures,
ubiquitin and the villin headpiece align at 39% identity and the 20-residue Trp-cage reaches
30% against three unrelated proteins. None of those pairs is homologous; the numbers are what
short sequences produce by chance. Clustering on them at 30% would chain nearly every protein
into one group and destroy the independence the split exists to protect.

What this uses instead. A Smith-Waterman score is compared against a null built by shuffling
one of the two sequences many times. Shuffling preserves length and amino-acid composition and
destroys only the order, so the null answers the right question: how good a local alignment do
these two sequences produce *by chance*, given how long they are and what they are made of. A
pair counts as related only when its score stands out from that null and its identity clears a
floor. Both conditions, because either alone misfires on short input.

Clustering is single-linkage: if A is related to B and B to C, all three land in one group even
when A and C look unrelated. That is deliberate. Single linkage over-merges rather than
under-merges, and for a leakage guard an over-merged group costs a little statistical power
while an under-merged one invalidates the evaluation.

Cost. Scoring every pair is cheap. Building a null for every pair is not, so the null is built
only for pairs whose raw score already places them among the candidates; everything else cannot
reach the threshold and is rejected without further work.
"""

import hashlib
import itertools
import random
import statistics

try:
    import parasail
except ImportError as error:  # pragma: no cover - environment-dependent
    raise ImportError("Sequence clustering needs parasail; install it first") from error

MATRIX = parasail.blosum62
GAP_OPEN = 11
GAP_EXTEND = 1
# A pair is related only when both hold. The Z threshold does the real work; the identity
# floor stops a long, weakly-similar alignment from passing on score alone.
DEFAULT_Z = 4.0
DEFAULT_IDENTITY = 0.25
DEFAULT_SHUFFLES = 300
# Pairs scoring below this quantile of all raw scores cannot clear the Z threshold in
# practice, so no null is built for them.
DEFAULT_CANDIDATE_QUANTILE = 0.80


def _align(a, b):
    """Local alignment, used for the score the significance test is built on."""
    return parasail.sw_stats_striped_16(a, b, GAP_OPEN, GAP_EXTEND, MATRIX)


def _align_global(a, b):
    """End-to-end alignment, used for the conventional identity figure."""
    return parasail.nw_stats_striped_16(a, b, GAP_OPEN, GAP_EXTEND, MATRIX)


def identity(a, b):
    """Matched residues over the shorter sequence, from an end-to-end alignment.

    This is the number the usual 30% recipe thresholds on, and it is reported here so the
    comparison against the significance test is visible. It is never used on its own: on
    short domains it reaches 39% between unrelated proteins.
    """
    if not a or not b:
        raise ValueError("Cannot align an empty sequence")
    return _align_global(a, b).matches / min(len(a), len(b))


def raw_score(a, b):
    return _align(a, b).score


def null_statistics(a, b, shuffles=DEFAULT_SHUFFLES, seed=0):
    """Mean and spread of the alignment score against shuffled copies of b.

    Shuffling holds length and composition fixed, so the null isolates the contribution of
    residue order. The seed is derived from the pair, so a pair's null never depends on the
    order sequences were handed in.

    The derivation goes through SHA-256 rather than `hash()`. Python randomises string hashing
    per process, so seeding from a tuple containing sequence text made every z-score change
    between runs: clusters, and any split built on them, could not be regenerated. That defect
    silently invalidated the promise this module exists to keep, and it is why the pair key is
    hashed explicitly here.
    """
    if shuffles < 2:
        raise ValueError("A null distribution needs at least two shuffles")
    letters = list(b)
    key = "homology_null_v1|%d|%d|%d|%s|%s" % (seed, len(a), len(b), a[:8], b[:8])
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    scores = []
    for _ in range(shuffles):
        rng.shuffle(letters)
        scores.append(_align(a, "".join(letters)).score)
    return statistics.fmean(scores), statistics.pstdev(scores)


def relatedness(a, b, shuffles=DEFAULT_SHUFFLES, seed=0):
    """Score a pair against its own shuffled null, returning the full record."""
    observed = raw_score(a, b)
    mean, spread = null_statistics(a, b, shuffles, seed)
    z = (observed - mean) / spread if spread > 0 else 0.0
    return {"score": observed, "null_mean": mean, "null_sd": spread,
            "z": z, "identity": identity(a, b)}


def _quantile(values, q):
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[index]


def related_pairs(sequences, z_threshold=DEFAULT_Z, identity_threshold=DEFAULT_IDENTITY,
                  shuffles=DEFAULT_SHUFFLES, seed=0,
                  candidate_quantile=DEFAULT_CANDIDATE_QUANTILE):
    """Every pair judged related, with the evidence for each.

    `sequences` maps an identifier to its sequence. Identifiers are sorted first, so the
    result never depends on insertion order.
    """
    names = sorted(sequences)
    if len(names) != len(set(names)):
        raise ValueError("Duplicate sequence identifier")
    for name in names:
        if not sequences[name]:
            raise ValueError("Empty sequence for " + name)
    pairs = list(itertools.combinations(names, 2))
    if not pairs:
        return [], []
    scored = [(a, b, raw_score(sequences[a], sequences[b])) for a, b in pairs]
    cutoff = _quantile([s for _, _, s in scored], candidate_quantile)
    evidence, related = [], []
    for a, b, score in scored:
        if score < cutoff:
            # Below the candidate cutoff; recorded as screened, never as related.
            evidence.append({"a": a, "b": b, "score": score, "screened": True,
                             "z": None, "identity": None, "related": False})
            continue
        record = relatedness(sequences[a], sequences[b], shuffles, seed)
        is_related = record["z"] >= z_threshold and record["identity"] >= identity_threshold
        evidence.append({"a": a, "b": b, "screened": False, "related": is_related, **record})
        if is_related:
            related.append((a, b))
    return related, evidence


def single_linkage(names, related):
    """Merge every connected group, so an indirect link still merges."""
    parent = {name: name for name in names}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in related:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    groups = {}
    for name in sorted(names):
        groups.setdefault(find(name), []).append(name)
    # Name each cluster after its first member, so names are stable across runs.
    return {root: sorted(members) for root, members in sorted(groups.items())}


def cluster(sequences, **kwargs):
    """Cluster sequences and return the groups together with the evidence behind them."""
    related, evidence = related_pairs(sequences, **kwargs)
    clusters = single_linkage(sorted(sequences), related)
    return {
        "clusters": clusters,
        "cluster_count": len(clusters),
        "sequence_count": len(sequences),
        "related_pairs": [{"a": a, "b": b} for a, b in related],
        "evidence": evidence,
        "settings": {
            "z_threshold": kwargs.get("z_threshold", DEFAULT_Z),
            "identity_threshold": kwargs.get("identity_threshold", DEFAULT_IDENTITY),
            "shuffles": kwargs.get("shuffles", DEFAULT_SHUFFLES),
            "candidate_quantile": kwargs.get("candidate_quantile", DEFAULT_CANDIDATE_QUANTILE),
            "matrix": "blosum62", "gap_open": GAP_OPEN, "gap_extend": GAP_EXTEND,
        },
    }


def assign_splits(clusters, fractions, seed=0):
    """Place whole clusters into named splits, largest first, filling the neediest split.

    Whole clusters move together, which is the entire point: a split that takes half a cluster
    leaks. Assignment is deterministic, so a split can be regenerated exactly.
    """
    if not fractions:
        raise ValueError("At least one split is required")
    total = sum(fractions.values())
    if total <= 0 or any(v < 0 for v in fractions.values()):
        raise ValueError("Split fractions must be nonnegative and sum above zero")
    targets = {name: len(
        [m for members in clusters.values() for m in members]) * value / total
        for name, value in fractions.items()}
    held = {name: [] for name in fractions}
    order = sorted(clusters.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    rng = random.Random(seed)
    for _, members in order:
        # Whichever split is furthest below its target takes the next cluster; ties break
        # on a seeded draw so the result is reproducible without favouring one name.
        gaps = {name: targets[name] - len(held[name]) for name in held}
        best = max(gaps.values())
        candidates = sorted(name for name, gap in gaps.items() if gap == best)
        held[rng.choice(candidates)].extend(members)
    return {name: sorted(members) for name, members in held.items()}


def verify_split(assignment, clusters):
    """Fail loudly if any cluster was divided between splits."""
    placement = {}
    for split, members in assignment.items():
        for member in members:
            if member in placement:
                raise ValueError("Sequence assigned to two splits: " + member)
            placement[member] = split
    for root, members in clusters.items():
        splits = {placement[m] for m in members if m in placement}
        if len(splits) > 1:
            raise ValueError("Cluster " + root + " is split across " + ", ".join(sorted(splits)))
    missing = {m for members in clusters.values() for m in members} - set(placement)
    if missing:
        raise ValueError("Unassigned sequences: " + ", ".join(sorted(missing)))
    return True

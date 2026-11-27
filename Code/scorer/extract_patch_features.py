"""Extract patch-local conditional-probability features for the measured arrangement contrasts.

One feature vector per arrangement, not per contrast. For a site pair (i, j) carrying
identities (a, b), arrangement A places a at i and b at j; arrangement B is the reverse.
Both sit on the same backbone, so one featurization and one batched forward pass serve every
arrangement of a site pair, exactly as `scoring/score_swaps_full.py` does for the frozen
likelihood.

**The site mapping is deliberately identical to the frozen baseline's.** `read_structure`,
the `{pdb_number: sequence_position}` index and the `letters[pi], letters[pj] = aa_i, aa_j`
substitution are reused from that module rather than reimplemented, so these features and the
committed -0.501 baseline describe the same contrasts at the same residues. A divergence here
would be invisible in the results and would invalidate every comparison.

Orientation follows the contrast table, which canonicalises A and B alphabetically without
reading the measurement: `aa_i` is the lexicographically smaller identity, `stability_forward`
is the arrangement with `aa_i` at `site_i`, and `difference = forward - reverse`. So
`d = difference` is already the spec's measured stability(A) - stability(B), higher meaning
more stable, and no sign is reversed anywhere in this module.

Output is one row per arrangement plus a manifest naming the two rows of each contrast. The
expensive part is therefore reused when the head, the loss or the eligibility rules change.
"""

import argparse
import copy
import csv
import importlib
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "baseline"))
sys.path.insert(0, str(HERE))
import run_baseline as baseline  # noqa: E402
import features as patch_features  # noqa: E402

sys.path.insert(0, str(HERE.parent / "scoring"))
from score_swaps_full import read_structure  # noqa: E402

ALPHABET = patch_features.ALPHABET
PATCH_SIZE = 2


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n",
                          encoding="utf-8")


def load_contrasts(path):
    """Group contrast rows by site pair, preserving every field for the manifest."""
    grouped, total = {}, 0
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            total += 1
            key = (row["domain"], int(row["site_i"]), int(row["site_j"]))
            grouped.setdefault(key, []).append({
                "domain": row["domain"],
                "site_i": int(row["site_i"]),
                "site_j": int(row["site_j"]),
                "aa_i": row["aa_i"],
                "aa_j": row["aa_j"],
                "stability_forward": float(row["stability_forward"]),
                "stability_reverse": float(row["stability_reverse"]),
                "difference": float(row["difference"]),
                "uncertainty": float(row["uncertainty"]),
                "resolved": row["resolved"].strip().lower() == "true",
            })
    return grouped, total


def load_homology(path):
    """Domain -> HSSP cluster and domain -> split, from the committed cluster file."""
    if path is None:
        return {}, {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    membership = dict(data["membership"])
    split_of = {}
    for split, domains in data["splits"]["assignment"].items():
        for domain in domains:
            split_of[domain] = split
    return membership, split_of


def arrangement(sequence, pos_i, pos_j, aa_i, aa_j):
    letters = list(sequence)
    letters[pos_i], letters[pos_j] = aa_i, aa_j
    return "".join(letters)


def nearest_neighbours(coordinates, pos_i, pos_j, count):
    """The `count` residues nearest the patch, by minimum Ca distance to either swapped site.

    Ties break by model sequence index, so the region is a deterministic function of the
    backbone alone. That matters twice over: the region is identical for arrangements A and B,
    which keeps the contrast honest, and it is fixed before any measurement is read, which
    keeps the region from being chosen to suit the labels.

    Squared distances are compared directly; the square root is monotonic and would only add
    floating-point noise to the tie-breaking.
    """
    if count <= 0:
        return []
    alpha = [row[1] for row in coordinates]
    patch = {pos_i, pos_j}
    scored = []
    for index, point in enumerate(alpha):
        if index in patch:
            continue
        scored.append((min(sum((a - b) ** 2 for a, b in zip(point, alpha[pos_i])),
                           sum((a - b) ** 2 for a, b in zip(point, alpha[pos_j]))), index))
    scored.sort()
    return [index for _, index in scored[:count]]


def check_pair(native, a, b, pos_i, pos_j):
    """The spec's eligibility rule, enforced rather than assumed.

    A and B must differ at exactly the two swapped positions, exchange two distinct
    identities, and agree everywhere else. Returns a reason string, or None when valid.
    """
    if len(a) != len(native) or len(b) != len(native):
        return "length mismatch against native chain"
    differing = [k for k, (x, y) in enumerate(zip(a, b)) if x != y]
    if differing != sorted((pos_i, pos_j)):
        return "arrangements differ at %d positions, not the two swapped sites" % len(differing)
    if a[pos_i] != b[pos_j] or a[pos_j] != b[pos_i]:
        return "the two identities are not exchanged"
    if a[pos_i] == a[pos_j]:
        return "identities are not distinct"
    if sorted(a) != sorted(b):
        return "composition is not preserved"
    return None


def load_model(mpnn_dir, model_type, device_request, require_cuda, torch):
    """Verify the pin, prove the GPU actually computes, then load frozen weights."""
    mpnn = Path(mpnn_dir).resolve()
    for relative, expected in baseline.UPSTREAM_HASHES.items():
        if baseline.sha256(mpnn / relative) != expected:
            raise ValueError("Pinned checksum mismatch: " + relative)
    sys.path.insert(0, str(mpnn))
    utils = importlib.import_module("protein_mpnn_utils")

    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA was required but is not available")
    device = device_request if (device_request != "cpu" and torch.cuda.is_available()) else "cpu"
    if device != "cpu":
        # A real computation, not a capability flag: the V100 failure showed that
        # torch.cuda.is_available() can be true on a GPU the binary cannot run on.
        probe = (torch.ones(64, 64, device=device) @ torch.ones(64, 64, device=device)).sum()
        if float(probe.item()) != 64 ** 3:
            raise RuntimeError("GPU arithmetic check failed")

    checkpoint_path = mpnn / (model_type + "_model_weights") / (baseline.MODEL + ".pt")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = utils.ProteinMPNN(num_letters=21, node_features=128, edge_features=128,
                              hidden_dim=128, num_encoder_layers=3, num_decoder_layers=3,
                              augment_eps=0.0, k_neighbors=checkpoint["num_edges"],
                              ca_only=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    model.requires_grad_(False)
    return model, utils, device, baseline.sha256(checkpoint_path)


def featurize_domain(utils, torch, device, domain, source):
    numbers, sequence, coordinates = read_structure(source)
    chain = {"name": domain, "seq_chain_A": sequence, "seq": sequence,
             "num_of_chains": 1, "coords_chain_A": {
                 "N_chain_A": [c[0] for c in coordinates],
                 "CA_chain_A": [c[1] for c in coordinates],
                 "C_chain_A": [c[2] for c in coordinates],
                 "O_chain_A": [c[3] for c in coordinates]}}
    packed = utils.tied_featurize([copy.deepcopy(chain)], device, {domain: (["A"], [])})
    return {
        "features": {"X": packed[0], "mask": packed[2], "chain_M": packed[4] * packed[10],
                     "chain_encoding_all": packed[5], "residue_idx": packed[12]},
        "sequence": sequence,
        "coordinates": coordinates,
        "index": {n: p for p, n in enumerate(numbers)},
    }


def mean_logprobs(model, entry, sequences, positions, torch, seeds, batch_size, tag):
    """Conditional log-probabilities at the patch, averaged over seeded decoding orders.

    Averaging in log space matches how the frozen baseline averages its per-order negative
    log likelihoods, so the two stay commensurable.
    """
    accumulated = None
    for seed in seeds:
        batch = patch_features.conditional_logprobs_at(
            model, entry["features"], sequences, positions, torch,
            seed=seed, batch_size=batch_size, tag=tag)
        if accumulated is None:
            accumulated = [{p: list(v) for p, v in row.items()} for row in batch]
            continue
        for target, row in zip(accumulated, batch):
            for position, values in row.items():
                target[position] = [a + b for a, b in zip(target[position], values)]
    count = len(seeds)
    if count > 1:
        for row in accumulated:
            for position in row:
                row[position] = [v / count for v in row[position]]
    return accumulated


def run(args):
    import numpy
    import torch

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    os.environ.setdefault("OMP_NUM_THREADS", "4")

    grouped, contrast_rows = load_contrasts(args.contrasts)
    membership, split_of = load_homology(args.homology)
    seeds = [int(s) for s in args.order_seeds.split(",") if s.strip()]
    if not seeds:
        raise ValueError("At least one order seed is required")

    model, utils, device, checkpoint_sha = load_model(
        args.mpnn_dir, args.model_type, args.device, args.require_cuda, torch)

    vectors, manifest, excluded = [], [], []
    cache = {}
    passes = 0
    width = patch_features.feature_width(PATCH_SIZE, args.neighbours)

    for (domain, site_i, site_j), rows in sorted(grouped.items()):
        if domain not in cache:
            source = Path(args.structures) / (domain + ".pdb")
            if not source.exists():
                cache[domain] = None
            else:
                cache[domain] = featurize_domain(utils, torch, device, domain, source)
        entry = cache[domain]
        if entry is None:
            excluded.append({"domain": domain, "site_i": site_i, "site_j": site_j,
                             "contrasts": len(rows), "reason": "no structure"})
            continue

        index, native = entry["index"], entry["sequence"]
        if site_i not in index or site_j not in index:
            excluded.append({"domain": domain, "site_i": site_i, "site_j": site_j,
                             "contrasts": len(rows), "reason": "site absent from structure"})
            continue
        pos_i, pos_j = index[site_i], index[site_j]

        # Build and validate every arrangement this site pair needs, then deduplicate.
        #
        # The cache is per site pair, deliberately. A feature vector is a function of the
        # sequence AND the two patch positions it was read at, so a cache keyed by sequence
        # alone would reuse rows across site pairs of the same domain -- the native sequence
        # belongs to every site pair whose identities happen to match the wild type, and one
        # domain had a single row claimed by seven site pairs -- and would silently return
        # log-probabilities read at the wrong residues. Scoping the cache to one site pair
        # makes that impossible by construction, and costs nothing: arrangements are never
        # shared between site pairs at the same positions.
        positions = [pos_i, pos_j]
        neighbours = nearest_neighbours(entry["coordinates"], pos_i, pos_j, args.neighbours)
        row_of, planned, needed = {}, [], []
        for row in rows:
            a = arrangement(native, pos_i, pos_j, row["aa_i"], row["aa_j"])
            b = arrangement(native, pos_i, pos_j, row["aa_j"], row["aa_i"])
            reason = check_pair(native, a, b, pos_i, pos_j)
            if reason is not None:
                excluded.append({**row, "reason": reason})
                continue
            planned.append((row, a, b))
            for sequence in (a, b):
                if sequence not in needed:
                    needed.append(sequence)
        if not planned:
            continue

        logprobs = mean_logprobs(model, entry, needed, positions, torch,
                                 seeds, args.batch_size, domain)
        passes += len(seeds) * PATCH_SIZE * -(-len(needed) // args.batch_size)
        for sequence, row_logprobs in zip(needed, logprobs):
            vector = patch_features.patch_feature_vector(row_logprobs, sequence, positions,
                                                         neighbours)
            if len(vector) < width:
                # A chain too short to supply the whole region uses all it has; the missing
                # blocks are all-zero, which no real residue identity produces, so "absent"
                # stays distinguishable from any amino acid rather than impersonating one.
                vector = vector + [0.0] * (width - len(vector))
            if len(vector) != width:
                raise ValueError("Unexpected feature width")
            row_of[sequence] = len(vectors)
            vectors.append(vector)

        for row, a, b in planned:
            manifest.append({
                "contrast_id": "%s:%d:%d:%s%s" % (domain, site_i, site_j,
                                                  row["aa_i"], row["aa_j"]),
                "domain": domain, "family": membership.get(domain),
                "split": split_of.get(domain),
                "site_i": site_i, "site_j": site_j,
                "position_i": pos_i, "position_j": pos_j,
                "aa_i": row["aa_i"], "aa_j": row["aa_j"],
                "row_a": row_of[a], "row_b": row_of[b],
                "d": row["difference"], "uncertainty": row["uncertainty"],
                "resolved": row["resolved"],
                "restores_wild_type": native[pos_i] in (row["aa_i"], row["aa_j"])
                                      or native[pos_j] in (row["aa_i"], row["aa_j"]),
                "neighbours": " ".join(str(n) for n in neighbours),
                "length": len(native),
            })

    if not manifest:
        raise ValueError("No eligible contrasts")

    # Within one site pair every contrast carries a distinct unordered identity pair, so its
    # two arrangements are distinct and are shared with no other contrast. Each eligible
    # contrast must therefore own exactly two private feature rows. This equality is what
    # catches a cache that reuses a row across site pairs, which is otherwise invisible.
    if len(vectors) != PATCH_SIZE * len(manifest):
        raise ValueError("Expected %d arrangement rows for %d contrasts, built %d; a feature "
                         "row is being shared between contrasts"
                         % (PATCH_SIZE * len(manifest), len(manifest), len(vectors)))

    table = numpy.asarray(vectors, dtype=numpy.float32)
    if not numpy.isfinite(table).all():
        raise ValueError("Nonfinite features")
    numpy.save(output / "patch_features.npy", table)
    with (output / "contrasts.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    write_json(output / "exclusions.json", excluded)

    resolved = sum(1 for row in manifest if row["resolved"])
    splits = {}
    for row in manifest:
        bucket = splits.setdefault(row["split"], {"contrasts": 0, "resolved": 0,
                                                  "domains": set(), "families": set()})
        bucket["contrasts"] += 1
        bucket["resolved"] += int(row["resolved"])
        bucket["domains"].add(row["domain"])
        bucket["families"].add(row["family"])
    record = {
        "schema_version": 1,
        "purpose": "patch_local_conditional_probability_features",
        "model_type": args.model_type, "model_name": baseline.MODEL,
        "upstream_commit": baseline.UPSTREAM_COMMIT,
        "checkpoint_sha256": checkpoint_sha,
        "contrasts_source_sha256": baseline.sha256(Path(args.contrasts)),
        "device": device, "batch_size": args.batch_size,
        "order_seeds": seeds, "patch_size": PATCH_SIZE, "feature_width": width,
        "neighbours_requested": args.neighbours,
        "contrasts_in_source": contrast_rows,
        "contrasts_featurized": len(manifest),
        "contrasts_resolved": resolved,
        "contrasts_excluded": sum(e.get("contrasts", 1) for e in excluded),
        "arrangements_featurized": len(vectors),
        "forward_passes": passes,
        "domains": len(set(row["domain"] for row in manifest)),
        "families": len(set(row["family"] for row in manifest)),
        "per_split": {k: {"contrasts": v["contrasts"], "resolved": v["resolved"],
                          "domains": len(v["domains"]), "families": len(v["families"])}
                      for k, v in sorted(splits.items(), key=lambda kv: str(kv[0]))},
        "restores_wild_type": sum(1 for row in manifest if row["restores_wild_type"]),
        "elapsed_seconds": time.monotonic() - started,
    }
    write_json(output / "run_record.json", record)
    print(json.dumps({k: v for k, v in record.items()
                      if k not in ("per_split",)}, indent=2), flush=True)
    print("per_split " + json.dumps(record["per_split"]), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--contrasts", required=True, type=Path)
    parser.add_argument("--structures", required=True, type=Path)
    parser.add_argument("--mpnn-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--homology", type=Path,
                        help="Committed HSSP cluster file, for family and split columns")
    parser.add_argument("--model-type", default="vanilla", choices=("vanilla", "soluble"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--neighbours", type=int, default=0,
                        help="Nearest same-chain residues around the patch contributing their "
                             "identity; the spec's local region is 8")
    parser.add_argument("--order-seeds", default="137",
                        help="Comma-separated decoding-order seeds, averaged in log space")
    parser.add_argument("--limit-domains", type=int, default=0,
                        help="Pilot mode: featurize only the first N domains")
    args = parser.parse_args()
    if args.limit_domains:
        original = load_contrasts
        globals()["load_contrasts"] = lambda path: _limited(original(path), args.limit_domains)
    run(args)


def _limited(loaded, limit):
    grouped, total = loaded
    keep = sorted(set(key[0] for key in grouped))[:limit]
    return ({k: v for k, v in grouped.items() if k[0] in keep}, total)


if __name__ == "__main__":
    main()

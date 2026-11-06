"""Score swap variants with the full autoregressive model, to get the real frozen baseline.

The first-order run established that ProteinMPNN's position-wise marginals predict measured
two-site arrangement effects at about 0.44. That is not the baseline a new scorer has to beat.
The baseline is what the *whole* frozen model achieves, and the difference between the two is
the joint term the pilot decomposition measured internally at roughly 11%. Here it is tested
against a laboratory measurement.

Why this is affordable. Every variant of one site pair sits on the same backbone, so one
featurization serves all of them and they can be scored in a single batched forward pass under
a shared decoding order. Scoring proceeds per site pair rather than per variant: 578 site pairs
times the number of orders, instead of hundreds of thousands of individual passes.

Decoding orders are shared and explicit, following the pilot: both members of a comparison are
scored under identical orders, and the order is supplied to the model rather than induced from
a random seed. The order spread measured on the pilot was 6-7% of the effect, so orders are
averaged and the spread is retained per variant rather than discarded.

Output is one row per variant per order. Swap differences are derived afterwards, so that the
expensive scoring need not be repeated when the contrast definition changes.
"""

import argparse
import copy
import csv
import hashlib
import importlib
import json
import os
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "baseline"))
import run_baseline as baseline

ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
BACKBONE = ("N", "CA", "C", "O")
FIELDS = ("domain site_i site_j aa_i aa_j order_index order_sha256 "
          "length full_nll_mean full_nll_total").split()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_structure(path):
    residues = {}
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        if line[:4] != "ATOM":
            continue
        atom = line[12:16].strip()
        name = line[17:20].strip()
        if atom not in BACKBONE or name not in baseline.AA3:
            continue
        number = int(line[22:26])
        entry = residues.setdefault(number, {"aa": baseline.AA3[name], "atoms": {}})
        entry["atoms"].setdefault(atom, tuple(
            float(line[30 + 8 * i:38 + 8 * i]) for i in range(3)))
    numbers = sorted(n for n in residues if len(residues[n]["atoms"]) == len(BACKBONE))
    if not numbers:
        raise ValueError("No complete backbone residues in " + str(path))
    sequence = "".join(residues[n]["aa"] for n in numbers)
    coordinates = [[residues[n]["atoms"][a] for a in BACKBONE] for n in numbers]
    return numbers, sequence, coordinates


def make_orders(length, count, seed, label):
    raw = hashlib.sha256(("swap_orders_v1|%s|%d" % (label, seed)).encode()).digest()
    rng = random.Random(int.from_bytes(raw[:8], "big"))
    seen, orders = set(), []
    while len(orders) < count:
        values = list(range(length))
        rng.shuffle(values)
        key = tuple(values)
        if key in seen:
            continue
        seen.add(key)
        encoded = json.dumps([v + 1 for v in values], separators=(",", ":")).encode()
        orders.append({"order_index": len(orders) + 1,
                       "order_sha256": hashlib.sha256(encoded).hexdigest(),
                       "order": values})
    return orders


def score_batch(model, features, sequences, order, torch, batch_size):
    """Mean negative log probability per sequence, all on one backbone and one order."""
    results = []
    length = features["X"].shape[1]
    device = features["X"].device
    decoding = torch.tensor([order], dtype=torch.long, device=device)
    for start in range(0, len(sequences), batch_size):
        chunk = sequences[start:start + batch_size]
        size = len(chunk)
        tokens = torch.tensor([[ALPHABET.index(c) for c in s] for s in chunk],
                              dtype=torch.long, device=device)
        X = features["X"].expand(size, -1, -1, -1).contiguous()
        mask = features["mask"].expand(size, -1).contiguous()
        chain_M = features["chain_M"].expand(size, -1).contiguous()
        residue_idx = features["residue_idx"].expand(size, -1).contiguous()
        encoding = features["chain_encoding_all"].expand(size, -1).contiguous()
        noise = torch.zeros_like(mask)
        orders = decoding.expand(size, -1).contiguous()
        with torch.inference_mode():
            log_probs = model(X, tokens, mask, chain_M, residue_idx, encoding, noise,
                              use_input_decoding_order=True, decoding_order=orders)
        if tuple(log_probs.shape) != (size, length, len(ALPHABET)):
            raise ValueError("Unexpected model output shape")
        nll = -torch.gather(log_probs, -1, tokens.unsqueeze(-1)).squeeze(-1)
        results.extend(nll.mean(dim=1).detach().cpu().tolist())
    return results


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    os.environ["OMP_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = "2"

    wanted = {}
    with Path(args.contrasts).open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            key = (row["domain"], int(row["site_i"]), int(row["site_j"]))
            pair = wanted.setdefault(key, set())
            pair.add((row["aa_i"], row["aa_j"]))
            pair.add((row["aa_j"], row["aa_i"]))

    import torch
    mpnn = args.mpnn_dir.resolve()
    for relative, expected in baseline.UPSTREAM_HASHES.items():
        if baseline.sha256(mpnn / relative) != expected:
            raise ValueError("Pinned checksum mismatch: " + relative)
    sys.path.insert(0, str(mpnn))
    utils = importlib.import_module("protein_mpnn_utils")
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA was required but is not available")
    device = "cuda" if (torch.cuda.is_available() and args.device != "cpu") else "cpu"
    # A real computation, not just a capability flag: the V100 failure showed availability
    # alone does not prove the installed binary can run on this GPU.
    if device == "cuda":
        probe = (torch.ones(64, 64, device=device) @ torch.ones(64, 64, device=device)).sum()
        if float(probe.item()) != 64 ** 3:
            raise RuntimeError("GPU arithmetic check failed")

    checkpoint = torch.load(mpnn / (args.model_type + "_model_weights") / (baseline.MODEL + ".pt"),
                            map_location=device, weights_only=True)
    model = utils.ProteinMPNN(num_letters=21, node_features=128, edge_features=128,
                              hidden_dim=128, num_encoder_layers=3, num_decoder_layers=3,
                              augment_eps=0.0, k_neighbors=checkpoint["num_edges"],
                              ca_only=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    model.requires_grad_(False)

    rows, done, skipped = [], 0, []
    cache = {}
    for (domain, site_i, site_j), identities in sorted(wanted.items()):
        if domain not in cache:
            source = Path(args.structures) / (domain + ".pdb")
            if not source.exists():
                skipped.append({"domain": domain, "reason": "no structure"})
                cache[domain] = None
            else:
                numbers, sequence, coordinates = read_structure(source)
                chain = {"name": domain, "seq_chain_A": sequence, "seq": sequence,
                         "num_of_chains": 1, "coords_chain_A": {
                             "N_chain_A": [c[0] for c in coordinates],
                             "CA_chain_A": [c[1] for c in coordinates],
                             "C_chain_A": [c[2] for c in coordinates],
                             "O_chain_A": [c[3] for c in coordinates]}}
                packed = utils.tied_featurize([copy.deepcopy(chain)], device,
                                              {domain: (["A"], [])})
                features = {"X": packed[0], "mask": packed[2],
                            "chain_M": packed[4] * packed[10],
                            "chain_encoding_all": packed[5], "residue_idx": packed[12]}
                cache[domain] = {"features": features, "sequence": sequence,
                                 "index": {n: p for p, n in enumerate(numbers)},
                                 "orders": make_orders(len(sequence), args.orders,
                                                       args.order_seed, domain)}
        entry = cache[domain]
        if entry is None:
            continue
        index = entry["index"]
        if site_i not in index or site_j not in index:
            skipped.append({"domain": domain, "sites": [site_i, site_j],
                            "reason": "site absent from structure"})
            continue
        pi, pj = index[site_i], index[site_j]
        combinations = sorted(identities)
        sequences = []
        for aa_i, aa_j in combinations:
            letters = list(entry["sequence"])
            letters[pi], letters[pj] = aa_i, aa_j
            sequences.append("".join(letters))
        for order in entry["orders"]:
            means = score_batch(model, entry["features"], sequences, order["order"],
                                torch, args.batch_size)
            for (aa_i, aa_j), mean in zip(combinations, means):
                rows.append({"domain": domain, "site_i": site_i, "site_j": site_j,
                             "aa_i": aa_i, "aa_j": aa_j,
                             "order_index": order["order_index"],
                             "order_sha256": order["order_sha256"],
                             "length": len(entry["sequence"]),
                             "full_nll_mean": mean,
                             "full_nll_total": mean * len(entry["sequence"])})
        done += 1
        if done % 25 == 0:
            print("  %d / %d site pairs (%.1f s)" % (done, len(wanted),
                                                     time.monotonic() - started), flush=True)

    with (output / "swap_variant_scores.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    write_json(output / "run_record.json", {
        "schema_version": 1, "purpose": "full_autoregressive_swap_scoring",
        "model_type": args.model_type, "model_name": baseline.MODEL,
        "upstream_commit": baseline.UPSTREAM_COMMIT, "device": device,
        "orders": args.orders, "order_seed": args.order_seed, "batch_size": args.batch_size,
        "site_pairs_requested": len(wanted), "site_pairs_scored": done,
        "variant_order_rows": len(rows), "skipped": skipped,
        "elapsed_seconds": time.monotonic() - started,
        "scores_sha256": baseline.sha256(output / "swap_variant_scores.csv"),
    })
    print("scored %d site pairs into %d rows in %.1f s" % (done, len(rows),
                                                           time.monotonic() - started),
          flush=True)
    print("Complete: " + str(output / "swap_variant_scores.csv"), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--contrasts", required=True)
    parser.add_argument("--structures", required=True)
    parser.add_argument("--mpnn-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-type", default="vanilla", choices=("vanilla", "soluble"))
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--orders", type=int, default=8)
    parser.add_argument("--order-seed", type=int, default=137)
    parser.add_argument("--batch-size", type=int, default=100)
    run(parser.parse_args())


if __name__ == "__main__":
    main()

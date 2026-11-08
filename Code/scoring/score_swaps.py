"""Score measured arrangement swaps with frozen ProteinMPNN and compare against the laboratory.

Each swap is two variants of one domain carrying the same two residue identities in opposite
arrangements, with a measured difference in folding stability. The question is whether the
model's preference between them tracks what was measured.

Why the first-order term is scored first, and separately. The decomposition on the pilot pairs
found that roughly 89% of ProteinMPNN's response to a rearrangement is a position-wise term
that takes no sequence context. That term comes from `unconditional_probs`, which needs **one
forward pass per structure**, not one per variant. Two hundred and eight passes therefore score
every one of the hundred thousand swaps, because each swap's first-order difference reduces to
four table lookups:

    delta_fo = [ -log p_i(Y) - log p_j(X) ] - [ -log p_i(X) - log p_j(Y) ]

Everything outside the two sites is identical between the variants and cancels exactly, which
is the same invariant the pilot decomposition asserts and verifies.

The remaining joint term needs full autoregressive scoring of both variants, two passes each,
and is left to a separate run on hardware that suits it. Splitting the work this way means the
cheap half of the experiment, which covers the dominant term, runs anywhere.

Residue numbering is taken from the structure. The site numbers in the dataset were already
checked against these structures geometrically: selected pairs sit a median 5.6 angstrom apart
against 14.0 for all pairs, which would be impossible under a numbering mismatch.
"""

import argparse
import csv
from collections import defaultdict
import importlib
import json
import math
import os
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "baseline"))
import run_baseline as baseline

ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
THREE_TO_ONE = baseline.AA3
BACKBONE = ("N", "CA", "C", "O")
FIELDS = ("domain site_i site_j aa_i aa_j measured_difference measured_uncertainty "
          "resolved model_first_order_difference").split()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_structure(path):
    """Ordered residue numbers, one-letter sequence and backbone coordinates from a PDB."""
    residues = {}
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        if line[:4] != "ATOM":
            continue
        atom = line[12:16].strip()
        if atom not in BACKBONE:
            continue
        name = line[17:20].strip()
        if name not in THREE_TO_ONE:
            continue
        number = int(line[22:26])
        entry = residues.setdefault(number, {"aa": THREE_TO_ONE[name], "atoms": {}})
        if entry["aa"] != THREE_TO_ONE[name]:
            raise ValueError("Conflicting residue identity at " + str(number))
        entry["atoms"].setdefault(atom, tuple(
            float(line[30 + 8 * i:38 + 8 * i]) for i in range(3)))
    numbers = sorted(n for n in residues if len(residues[n]["atoms"]) == len(BACKBONE))
    if not numbers:
        raise ValueError("No complete backbone residues in " + str(path))
    sequence = "".join(residues[n]["aa"] for n in numbers)
    coordinates = [[residues[n]["atoms"][a] for a in BACKBONE] for n in numbers]
    return numbers, sequence, coordinates


def first_order_log_probs(model, features, torch):
    with torch.inference_mode():
        log_probs = model.unconditional_probs(
            features["X"], features["mask"], features["residue_idx"],
            features["chain_encoding_all"])
    if not bool(torch.isfinite(log_probs).all().item()):
        raise ValueError("Nonfinite unconditional log probabilities")
    totals = torch.logsumexp(log_probs, dim=-1)
    if float(torch.max(torch.abs(totals)).item()) > 1e-4:
        raise ValueError("Unconditional distributions are not normalized")
    return log_probs[0].detach().cpu().numpy().astype("float64")


def spearman(xs, ys):
    def ranked(values):
        order = sorted(range(len(values)), key=lambda i: values[i])
        result = [0.0] * len(values)
        position = 0
        while position < len(order):
            stop = position
            while stop + 1 < len(order) and values[order[stop + 1]] == values[order[position]]:
                stop += 1
            shared = (position + stop) / 2.0
            for index in range(position, stop + 1):
                result[order[index]] = shared
            position = stop + 1
        return result
    a, b = ranked(xs), ranked(ys)
    mean_a, mean_b = statistics.fmean(a), statistics.fmean(b)
    da = [v - mean_a for v in a]
    db = [v - mean_b for v in b]
    denominator = math.sqrt(sum(v * v for v in da) * sum(v * v for v in db))
    return sum(x * y for x, y in zip(da, db)) / denominator if denominator else None


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.environ["OMP_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = "2"
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    contrasts = []
    with Path(args.contrasts).open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            contrasts.append(row)
    wanted = sorted({row["domain"] for row in contrasts})

    import copy
    import numpy as np
    import torch
    mpnn = args.mpnn_dir.resolve()
    for relative, expected in baseline.UPSTREAM_HASHES.items():
        if baseline.sha256(mpnn / relative) != expected:
            raise ValueError("Pinned checksum mismatch: " + relative)
    sys.path.insert(0, str(mpnn))
    utils = importlib.import_module("protein_mpnn_utils")

    checkpoint = torch.load(mpnn / (args.model_type + "_model_weights") / (baseline.MODEL + ".pt"),
                            map_location=args.device, weights_only=True)
    model = utils.ProteinMPNN(num_letters=21, node_features=128, edge_features=128,
                              hidden_dim=128, num_encoder_layers=3, num_decoder_layers=3,
                              augment_eps=0.0, k_neighbors=checkpoint["num_edges"],
                              ca_only=False).to(args.device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    model.requires_grad_(False)
    if model.training or model.features.augment_eps != 0:
        raise ValueError("Scoring requires evaluation mode and zero backbone noise")

    marginals, index_of, skipped = {}, {}, []
    staged = output / "inputs"
    staged.mkdir(exist_ok=True)
    for domain in wanted:
        source = Path(args.structures) / (domain + ".pdb")
        if not source.exists():
            skipped.append({"domain": domain, "reason": "no structure"})
            continue
        try:
            numbers, sequence, coordinates = read_structure(source)
        except ValueError as error:
            skipped.append({"domain": domain, "reason": str(error)})
            continue
        chain = {"name": domain, "seq_chain_A": sequence, "coords_chain_A": {
            "N_chain_A": [c[0] for c in coordinates], "CA_chain_A": [c[1] for c in coordinates],
            "C_chain_A": [c[2] for c in coordinates], "O_chain_A": [c[3] for c in coordinates]},
            "num_of_chains": 1, "seq": sequence}
        packed = utils.tied_featurize([copy.deepcopy(chain)], args.device,
                                      {domain: (["A"], [])})
        X, mask, encoding, residue_idx = packed[0], packed[2], packed[5], packed[12]
        features = {"X": X, "mask": mask, "chain_encoding_all": encoding,
                    "residue_idx": residue_idx}
        marginals[domain] = first_order_log_probs(model, features, torch)
        index_of[domain] = {number: position for position, number in enumerate(numbers)}
        print("  scored {} ({} residues)".format(domain, len(numbers)), flush=True)

    rows, dropped = [], []
    for row in contrasts:
        domain = row["domain"]
        if domain not in marginals:
            dropped.append({**row, "reason": "structure unavailable"})
            continue
        positions = index_of[domain]
        site_i, site_j = int(row["site_i"]), int(row["site_j"])
        if site_i not in positions or site_j not in positions:
            dropped.append({**row, "reason": "site absent from structure"})
            continue
        table = marginals[domain]
        a, b = row["aa_i"], row["aa_j"]
        if a not in ALPHABET or b not in ALPHABET:
            dropped.append({**row, "reason": "noncanonical identity"})
            continue
        pi, pj = positions[site_i], positions[site_j]
        ai, bi = ALPHABET.index(a), ALPHABET.index(b)
        # Forward puts a at site i and b at site j; reverse swaps them. Everything else
        # cancels, so only these four terms enter the difference.
        forward = -(table[pi, ai] + table[pj, bi])
        reverse = -(table[pi, bi] + table[pj, ai])
        rows.append({
            "domain": domain, "site_i": site_i, "site_j": site_j, "aa_i": a, "aa_j": b,
            "measured_difference": float(row["difference"]),
            "measured_uncertainty": float(row["uncertainty"]),
            "resolved": row["resolved"],
            "model_first_order_difference": forward - reverse,
        })

    with (output / "swap_first_order.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    def agreement(subset):
        if len(subset) < 3:
            return None
        measured = [r["measured_difference"] for r in subset]
        predicted = [r["model_first_order_difference"] for r in subset]
        # The model scores negative log probability, so a lower score means preferred; a
        # higher measured stability also means preferred. The sign convention is reported
        # rather than silently flipped.
        sign = sum(1 for r in subset
                   if (r["measured_difference"] > 0) == (r["model_first_order_difference"] < 0))
        return {"count": len(subset), "spearman": spearman(predicted, measured),
                "sign_agreement": sign / len(subset)}

    def per_domain(subset, minimum):
        """Per-domain correlations, so the first-order term is comparable to the full model.

        The full-model comparison reports a per-domain median because pooling tens of
        thousands of contrasts from two hundred domains overstates significance. Without the
        same block here, the two terms could only be compared pooled, and an earlier version
        of this script omitted it -- so the first-order per-domain median existed nowhere in
        the committed results and had to be recomputed by hand to be quoted at all.
        """
        grouped = defaultdict(list)
        for row in subset:
            grouped[row["domain"]].append(row)
        values = {}
        for domain, entries in grouped.items():
            if len(entries) < minimum:
                continue
            rho = spearman([e["model_first_order_difference"] for e in entries],
                           [e["measured_difference"] for e in entries])
            if rho is not None and rho == rho:
                values[domain] = rho
        if not values:
            return None
        ordered = sorted(values.values())
        return {"domains": len(values), "minimum_contrasts": minimum,
                "median": statistics.median(ordered),
                "p25": ordered[len(ordered) // 4], "p75": ordered[(3 * len(ordered)) // 4],
                "agreeing_in_sign": sum(1 for v in ordered if v < 0),
                "per_domain": [{"domain": d, "spearman": v}
                               for d, v in sorted(values.items())]}

    resolved = [r for r in rows if str(r["resolved"]).lower() == "true"]
    summary = {
        "schema_version": 2, "purpose": "model_versus_measured_arrangement_effect",
        "model_type": args.model_type, "model_name": baseline.MODEL,
        "term": "first-order only; the joint term requires full autoregressive scoring",
        "domains_scored": len(marginals), "domains_skipped": skipped,
        "contrasts_scored": len(rows), "contrasts_dropped": len(dropped),
        "all_contrasts": agreement(rows),
        "resolved_contrasts_only": agreement(resolved),
        "per_domain_resolved": per_domain(resolved, args.minimum_per_domain),
    }
    write_json(output / "swap_first_order_summary.json", summary)
    for label in ("all_contrasts", "resolved_contrasts_only"):
        block = summary[label]
        if block:
            print("{:<26} n={:<7} spearman={:+.3f}  sign agreement={:.1%}".format(
                label, block["count"], block["spearman"], block["sign_agreement"]), flush=True)
    block = summary["per_domain_resolved"]
    if block:
        print("{:<26} domains={:<5} median={:+.3f}  p25={:+.3f} p75={:+.3f}".format(
            "per_domain_resolved", block["domains"], block["median"],
            block["p25"], block["p75"]), flush=True)
    print("Complete: " + str(output / "swap_first_order_summary.json"), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--contrasts", required=True, help="swap_contrasts.csv")
    parser.add_argument("--structures", required=True, help="directory of domain PDB files")
    parser.add_argument("--mpnn-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-type", default="vanilla", choices=("vanilla", "soluble"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--minimum-per-domain", type=int, default=20,
                        help="contrasts a domain needs before its correlation is reported; "
                             "matches the full-model comparison so the two are comparable")
    run(parser.parse_args())


if __name__ == "__main__":
    main()

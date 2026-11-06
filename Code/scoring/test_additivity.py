"""Ask whether a two-site swap effect survives changing a third residue nearby.

This is the question that decides whether training a patch scorer on two-site contrasts can
reach patch scale. If the effect of swapping residues at sites i and j is the same whatever
occupies a third site k in the same network, then arrangement decomposes into pairs and
pairwise labels carry. If it shifts with k, patch effects are irreducibly many-body and no
amount of pairwise training reaches them.

It is the model counterpart of the measurement in the source paper, which reported that
substituting any member of a three-residue network for alanine removed the coupling between
the other two. The measured three-site variants are not in the double-mutant table used here,
so this asks the question of the model rather than the laboratory, which is the right target
anyway: the scorer is meant to improve on the model.

Design. For each three-residue network, and for each of its three pairs, the two residues of
the pair are placed in both arrangements while the third site is set to each of the twenty
identities in turn. The swap effect is the score difference between arrangements. A network
whose swap effects are flat across the third identity is additive at this scale; one whose
effects swing is not. Spread is reported against the swap effect's own size, because a swing
of a tenth of a nat means something different for a pair whose effect is three nats than for
one whose effect is a tenth.

All variants for one network and pair share a backbone and a decoding order, so they are
scored in a single batched pass.
"""

import argparse
import copy
import csv
import importlib
import json
import math
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "baseline"))
import run_baseline as baseline

ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
RESIDUES = "ACDEFGHIKLMNPQRSTVWY"
BACKBONE = ("N", "CA", "C", "O")
FIELDS = ("domain pair_i pair_j third_site aa_i aa_j third_aa swap_effect").split()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_structure(path):
    residues = {}
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        if line[:4] != "ATOM":
            continue
        atom, name = line[12:16].strip(), line[17:20].strip()
        if atom not in BACKBONE or name not in baseline.AA3:
            continue
        number = int(line[22:26])
        entry = residues.setdefault(number, {"aa": baseline.AA3[name], "atoms": {}})
        entry["atoms"].setdefault(atom, tuple(
            float(line[30 + 8 * i:38 + 8 * i]) for i in range(3)))
    numbers = sorted(n for n in residues if len(residues[n]["atoms"]) == len(BACKBONE))
    sequence = "".join(residues[n]["aa"] for n in numbers)
    coordinates = [[residues[n]["atoms"][a] for a in BACKBONE] for n in numbers]
    return numbers, sequence, coordinates


def score_batch(model, features, sequences, order, torch, batch_size):
    out = []
    device = features["X"].device
    decoding = torch.tensor([order], dtype=torch.long, device=device)
    for start in range(0, len(sequences), batch_size):
        chunk = sequences[start:start + batch_size]
        size = len(chunk)
        tokens = torch.tensor([[ALPHABET.index(c) for c in s] for s in chunk],
                              dtype=torch.long, device=device)
        expand = lambda t, d: t.expand(size, *([-1] * d)).contiguous()
        mask = expand(features["mask"], 1)
        with torch.inference_mode():
            log_probs = model(expand(features["X"], 3), tokens, mask,
                              expand(features["chain_M"], 1), expand(features["residue_idx"], 1),
                              expand(features["chain_encoding_all"], 1),
                              torch.zeros_like(mask), use_input_decoding_order=True,
                              decoding_order=decoding.expand(size, -1).contiguous())
        nll = -torch.gather(log_probs, -1, tokens.unsqueeze(-1)).squeeze(-1)
        out.extend((nll.sum(dim=1)).detach().cpu().tolist())
    return out


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    triples = []
    with Path(args.triples).open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            triples.append({"domain": row["pdb_name"].replace(".pdb", ""),
                            "sites": [int(row["pos1"]), int(row["pos2"]), int(row["pos3"])],
                            "wt": [row["aa1"], row["aa2"], row["aa3"]]})
    scored_pairs = set()
    if args.restrict_to:
        with Path(args.restrict_to).open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                scored_pairs.add((row["domain"], int(row["site_i"]), int(row["site_j"])))

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

    rows, networks, skipped = [], [], []
    cache = {}
    for triple in triples:
        domain, sites, wt = triple["domain"], triple["sites"], triple["wt"]
        pairs = [(0, 1, 2), (0, 2, 1), (1, 2, 0)]
        if scored_pairs:
            complete = all((domain, min(sites[a], sites[b]), max(sites[a], sites[b]))
                           in scored_pairs for a, b, _ in pairs)
            if not complete:
                skipped.append({"domain": domain, "reason": "not all pairs scored"})
                continue
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
                packed = utils.tied_featurize([copy.deepcopy(chain)], args.device,
                                              {domain: (["A"], [])})
                cache[domain] = {
                    "features": {"X": packed[0], "mask": packed[2],
                                 "chain_M": packed[4] * packed[10],
                                 "chain_encoding_all": packed[5], "residue_idx": packed[12]},
                    "sequence": sequence,
                    "index": {n: p for p, n in enumerate(numbers)},
                    "order": list(range(len(sequence)))}
        entry = cache[domain]
        if entry is None:
            continue
        if any(s not in entry["index"] for s in sites):
            skipped.append({"domain": domain, "reason": "site absent from structure"})
            continue
        for a, b, c in pairs:
            site_i, site_j, site_k = sites[a], sites[b], sites[c]
            aa_i, aa_j = wt[a], wt[b]
            if aa_i == aa_j:
                continue
            pi, pj, pk = (entry["index"][site_i], entry["index"][site_j],
                          entry["index"][site_k])
            sequences, labels = [], []
            for third in RESIDUES:
                for first, second in ((aa_i, aa_j), (aa_j, aa_i)):
                    letters = list(entry["sequence"])
                    letters[pi], letters[pj], letters[pk] = first, second, third
                    sequences.append("".join(letters))
                labels.append(third)
            totals = score_batch(model, entry["features"], sequences, entry["order"],
                                 torch, args.batch_size)
            effects = []
            for index, third in enumerate(labels):
                forward, reverse = totals[2 * index], totals[2 * index + 1]
                effect = forward - reverse
                effects.append(effect)
                rows.append({"domain": domain, "pair_i": site_i, "pair_j": site_j,
                             "third_site": site_k, "aa_i": aa_i, "aa_j": aa_j,
                             "third_aa": third, "swap_effect": effect})
            spread = statistics.pstdev(effects)
            mean = statistics.fmean(effects)
            networks.append({
                "domain": domain, "pair": [site_i, site_j], "third_site": site_k,
                "mean_swap_effect": mean, "spread_across_third": spread,
                "min": min(effects), "max": max(effects),
                "relative_spread": spread / abs(mean) if abs(mean) > 1e-9 else None,
                "sign_flips": sum(1 for e in effects if (e > 0) != (mean > 0)),
            })
        print("  %s done (%d networks)" % (domain, len(networks)), flush=True)

    with (output / "additivity_effects.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    relative = [n["relative_spread"] for n in networks if n["relative_spread"] is not None]
    summary = {
        "schema_version": 1, "purpose": "does_a_swap_effect_depend_on_a_third_residue",
        "model_type": args.model_type, "networks_tested": len(networks),
        "triples_skipped": skipped, "third_identities": len(RESIDUES),
        "mean_swap_effect_nats": {
            "median": statistics.median(abs(n["mean_swap_effect"]) for n in networks)
            if networks else None},
        "spread_across_third_nats": {
            "median": statistics.median(n["spread_across_third"] for n in networks)
            if networks else None},
        "relative_spread": {
            "median": statistics.median(relative) if relative else None,
            "p90": sorted(relative)[int(0.9 * len(relative))] if relative else None},
        "networks_with_a_sign_flip": sum(1 for n in networks if n["sign_flips"]),
        "networks": networks,
    }
    write_json(output / "additivity_summary.json", summary)
    if networks:
        print("\nnetworks tested          : %d" % len(networks))
        print("median |swap effect|     : %.3f nats"
              % summary["mean_swap_effect_nats"]["median"])
        print("median spread across k   : %.3f nats"
              % summary["spread_across_third_nats"]["median"])
        print("median relative spread   : %.1f%%" % (100 * summary["relative_spread"]["median"]))
        print("networks where the sign flips: %d of %d"
              % (summary["networks_with_a_sign_flip"], len(networks)))
    print("Complete: " + str(output / "additivity_summary.json"), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--triples", required=True)
    parser.add_argument("--structures", required=True)
    parser.add_argument("--mpnn-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--restrict-to", help="swap_contrasts.csv, to keep only scored triples")
    parser.add_argument("--model-type", default="vanilla", choices=("vanilla", "soluble"))
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--batch-size", type=int, default=40)
    run(parser.parse_args())


if __name__ == "__main__":
    main()

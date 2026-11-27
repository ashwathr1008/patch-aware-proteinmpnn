"""Cache the frozen decoder's final hidden state for every arrangement, whole chain.

This is the common input the spec's three heads all read: whole-chain mean, patch-local mean
and patch-local attention differ only in which positions they pool, so they must see exactly
the same tensor. Caching it once makes that identity structural rather than a thing to verify,
and keeps every fit cheap enough that cross-validation stays affordable.

The capture is a forward pre-hook on `W_out`, so the value stored is the actual input to the
checkpoint's output projection -- the real final decoder state under the real autoregressive
masks. Reimplementing the decoder to get it would risk a subtly different quantity that still
looked plausible.

Each arrangement is stored as `[L, hidden + 21]`: the decoder state concatenated with the
21-symbol identity one-hot of the residue actually placed there. Rows are padded to the
longest chain in the cohort and the true length is stored alongside, so a head can mask
padding rather than pool it.
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
sys.path.insert(0, str(HERE.parent / "scoring"))
import run_baseline as baseline  # noqa: E402
from score_swaps_full import read_structure, make_orders  # noqa: E402
from extract_patch_features import (  # noqa: E402
    arrangement, check_pair, load_contrasts, load_homology, nearest_neighbours, write_json)

ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
PATCH_SIZE = 2


def load_model(mpnn_dir, model_type, device_request, require_cuda, torch):
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
        probe = (torch.ones(64, 64, device=device) @ torch.ones(64, 64, device=device)).sum()
        if float(probe.item()) != 64 ** 3:
            raise RuntimeError("GPU arithmetic check failed")
    path = mpnn / (model_type + "_model_weights") / (baseline.MODEL + ".pt")
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = utils.ProteinMPNN(num_letters=21, node_features=128, edge_features=128,
                              hidden_dim=128, num_encoder_layers=3, num_decoder_layers=3,
                              augment_eps=0.0, k_neighbors=checkpoint["num_edges"],
                              ca_only=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    model.requires_grad_(False)
    return model, utils, device, baseline.sha256(path)


def arrangement_features(model, features, sequences, orders, torch, batch_size):
    """Per-position features for each arrangement, as [B, L, 128 + 21 + 24].

    Channel layout, in order:

        0:128    final decoder state, from the FIRST order only
        128:149  identity one-hot of the residue actually placed
        149:170  log-probability vector over the 21 symbols, averaged over orders
        170      log p(the residue actually placed)          <- the nesting channel
        171      entropy of the position's distribution
        172      log p(placed) - max_a log p(a)

    **Why channel 170 exists and how it is averaged.** The frozen baseline is, for one
    decoding order, the mean over positions of -log p(the placed residue), and those per-order
    means are then averaged across the eight orders. Averaging the *gathered* value per order
    and only then across orders reproduces that quantity exactly. Averaging the log-probability
    vector first and gathering afterwards would not, because the mean of a log-softmax is not
    the log-softmax of a mean.

    That exactness is the point: with this channel present, `whole_chain_mean` can reproduce
    the frozen score with a single weight, so the frozen baseline becomes a special case of
    the head rather than a rival architecture. Note this nesting holds only for the whole-chain
    head -- a local head pooling ten residues can at best reproduce a patch-restricted mean,
    which is a different statistic.

    Entropy and the margin are averaged per order as well. Both are derivable from the
    log-probability vector, but a 14k-parameter head should not have to learn the reduction.
    """
    out = []
    length = features["X"].shape[1]
    device = features["X"].device
    for start in range(0, len(sequences), batch_size):
        chunk = sequences[start:start + batch_size]
        size = len(chunk)
        tokens = torch.tensor([[ALPHABET.index(c) for c in s] for s in chunk],
                              dtype=torch.long, device=device)
        expand = lambda t, d: t.expand(size, *([-1] * d)).contiguous()
        state = None
        accumulated = None
        for position, order in enumerate(orders):
            decoding = torch.tensor([order], dtype=torch.long, device=device)
            captured = []
            hook = model.W_out.register_forward_pre_hook(
                lambda module, args: captured.append(args[0].detach()))
            try:
                with torch.inference_mode():
                    log_probs = model(
                        expand(features["X"], 3), tokens, expand(features["mask"], 1),
                        expand(features["chain_M"], 1), expand(features["residue_idx"], 1),
                        expand(features["chain_encoding_all"], 1),
                        torch.zeros(size, length, device=device),
                        use_input_decoding_order=True,
                        decoding_order=decoding.expand(size, -1).contiguous())
            finally:
                hook.remove()
            if len(captured) != 1:
                raise ValueError("Expected exactly one projection input per forward")
            if not bool(torch.isfinite(log_probs).all().item()):
                raise ValueError("Nonfinite log probabilities")
            if position == 0:
                state = captured[0]
                if not bool(torch.isfinite(state).all().item()):
                    raise ValueError("Nonfinite decoder state")
            gathered = torch.gather(log_probs, -1, tokens.unsqueeze(-1))
            entropy = -(log_probs.exp() * log_probs).sum(dim=-1, keepdim=True)
            margin = gathered - log_probs.max(dim=-1, keepdim=True).values
            block = torch.cat((log_probs, gathered, entropy, margin), dim=-1)
            accumulated = block if accumulated is None else accumulated + block
        accumulated = accumulated / len(orders)
        identity = torch.nn.functional.one_hot(tokens, len(ALPHABET)).to(state.dtype)
        combined = torch.cat((state, identity, accumulated.to(state.dtype)), dim=-1)
        out.append(combined.to(torch.float16).cpu().numpy())
    return out


def run(args):
    import numpy
    import torch

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    os.environ.setdefault("OMP_NUM_THREADS", "4")

    grouped, total_rows = load_contrasts(args.contrasts)
    if args.limit_domains:
        keep = sorted({key[0] for key in grouped})[:args.limit_domains]
        grouped = {k: v for k, v in grouped.items() if k[0] in keep}
        total_rows = sum(len(v) for v in grouped.values())
    membership, split_of = load_homology(args.homology)
    model, utils, device, checkpoint_sha = load_model(
        args.mpnn_dir, args.model_type, args.device, args.require_cuda, torch)

    manifest, excluded, cache = [], [], {}
    states = None
    lengths = numpy.zeros(2 * total_rows, dtype=numpy.int16)
    filled, passes, width = 0, 0, None

    for (domain, site_i, site_j), rows in sorted(grouped.items()):
        if domain not in cache:
            source = Path(args.structures) / (domain + ".pdb")
            if not source.exists():
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
                cache[domain] = {
                    "features": {"X": packed[0], "mask": packed[2],
                                 "chain_M": packed[4] * packed[10],
                                 "chain_encoding_all": packed[5], "residue_idx": packed[12]},
                    "sequence": sequence, "coordinates": coordinates,
                    "index": {n: p for p, n in enumerate(numbers)},
                    "orders": [o["order"] for o in make_orders(
                        len(sequence), args.orders, args.order_seed, domain)],
                }
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
        neighbours = nearest_neighbours(entry["coordinates"], pos_i, pos_j, args.neighbours)

        planned, needed, row_of = [], [], {}
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

        blocks = arrangement_features(model, entry["features"], needed, entry["orders"],
                                      torch, args.batch_size)
        passes += args.orders * -(-len(needed) // args.batch_size)
        stacked = numpy.concatenate(blocks, axis=0)
        if width is None:
            width = stacked.shape[2]
            states = numpy.zeros((2 * total_rows, args.max_length, width),
                                 dtype=numpy.float16)
        length = stacked.shape[1]
        if length > args.max_length:
            raise ValueError("Chain of %d exceeds --max-length %d" % (length, args.max_length))
        for offset, sequence in enumerate(needed):
            states[filled, :length, :] = stacked[offset]
            lengths[filled] = length
            row_of[sequence] = filled
            filled += 1

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
                "region": " ".join(str(p) for p in [pos_i, pos_j] + list(neighbours)),
                "length": len(native),
            })

    if filled != PATCH_SIZE * len(manifest):
        raise ValueError("Expected %d arrangement rows for %d contrasts, built %d"
                         % (PATCH_SIZE * len(manifest), len(manifest), filled))

    numpy.save(output / "decoder_states.npy", states[:filled])
    numpy.save(output / "lengths.npy", lengths[:filled])
    with (output / "contrasts.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    write_json(output / "exclusions.json", excluded)
    record = {
        "schema_version": 1, "purpose": "frozen_full_chain_decoder_states",
        "model_type": args.model_type, "model_name": baseline.MODEL,
        "upstream_commit": baseline.UPSTREAM_COMMIT,
        "checkpoint_sha256": checkpoint_sha,
        "contrasts_source_sha256": baseline.sha256(Path(args.contrasts)),
        "device": device, "batch_size": args.batch_size, "order_seed": args.order_seed,
        "orders_per_arrangement": args.orders, "state_width": width, "max_length": args.max_length,
        "neighbours": args.neighbours,
        "contrasts_in_source": total_rows, "contrasts_featurized": len(manifest),
        "contrasts_resolved": sum(1 for r in manifest if r["resolved"]),
        "arrangements_featurized": filled, "forward_passes": passes,
        "domains": len(set(r["domain"] for r in manifest)),
        "families": len(set(r["family"] for r in manifest)),
        "restores_wild_type": sum(1 for r in manifest if r["restores_wild_type"]),
        "cache_bytes": int(states[:filled].nbytes),
        "elapsed_seconds": time.monotonic() - started,
    }
    write_json(output / "run_record.json", record)
    print(json.dumps(record, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--contrasts", required=True, type=Path)
    parser.add_argument("--structures", required=True, type=Path)
    parser.add_argument("--mpnn-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--homology", type=Path)
    parser.add_argument("--model-type", default="vanilla", choices=("vanilla", "soluble"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--order-seed", type=int, default=137)
    parser.add_argument("--orders", type=int, default=8,
                        help="Decoding orders; the log-probability channels are\n                             averaged over them, matching the frozen baseline")
    parser.add_argument("--neighbours", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=72)
    parser.add_argument("--limit-domains", type=int, default=0,
                        help="Pilot mode: featurize only the first N domains")
    run(parser.parse_args())


if __name__ == "__main__":
    main()

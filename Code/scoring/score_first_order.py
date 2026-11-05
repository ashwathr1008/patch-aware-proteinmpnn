"""Decompose each arrangement gap into a first-order burial term and a joint-structure term.

The paired experiment reports a full-chain autoregressive gap, permuted minus original. Two
different models predict that gap:

  (a) ProteinMPNN represents which positions' identities depend on each other, so moving
      identities within a patch disrupts a joint it has learned.
  (b) ProteinMPNN holds position-specific marginal preferences driven by burial and nothing
      else. A permutation moves a hydrophobic residue to an exposed position whose marginal
      probability for it is low, and the total drops with no coupling involved.

Interpretation (b) alone predicts the published result, and (b) is burial-weighted composition,
which is the closed-form SAP statistic rather than a model of arrangement. Separating the two
is therefore necessary before the result can be described as evidence about arrangement.

The upstream model exposes the first-order distribution directly. ProteinMPNN.unconditional_probs
runs the encoder and decoder with an all-zero backward order mask, so no sequence information
reaches any position and the output is p(aa_i | backbone) at every position. It takes no decoding
order, so the first-order score is order-free by construction and one forward pass per structure
and checkpoint serves every pair on that structure.

  score_fo(assignment) = -sum_i log p_i_uncond(a_i)
  delta_fo    = score_fo(permuted) - score_fo(original)
  delta_joint = delta_full - delta_fo

delta_full is the order-averaged full-chain total from an existing pair_scores.csv, so the
decomposition adds forward passes for the first-order term only and reuses the scored run.

A structural invariant is asserted rather than assumed: because unconditional probabilities
ignore sequence context and these pairs change identities only inside the patch, every
outside-patch position must contribute exactly zero to delta_fo. The script verifies that the
whole-chain and patch-only first-order gaps agree to numerical tolerance for every pair, which
also confirms the patch indexing against the model's own featurization.

Scores are reported as totals in nats per chain, with per-residue means alongside. Totals are
the additive quantity; the per-residue means in pair_scores.csv are totals divided by length.
"""

import argparse
import copy
import csv
from collections import defaultdict
import importlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "baseline"))
import run_baseline as baseline
import score_pairs

ALPHABET = score_pairs.ALPHABET
TOLERANCE = 1e-6
# Agreement between the whole-chain and patch-only first-order gaps. Looser than TOLERANCE
# because it compares sums of many float32 model outputs, not a repeated identical computation.
INVARIANT_TOLERANCE = 1e-4
SCORE_FIELDS = (
    "pair_id structure_id chain split group patch_id model_type model_name length patch_size "
    "changed_identity_count original_fo_total permuted_fo_total delta_fo_total "
    "original_fo_patch_total permuted_fo_patch_total delta_fo_patch_total "
    "outside_patch_fo_gap delta_full_total delta_joint_total delta_fo_share "
    "delta_fo_per_residue delta_joint_per_residue order_count"
).split()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sign(value, tolerance=TOLERANCE):
    return 1 if value > tolerance else -1 if value < -tolerance else 0


def sign_test(values):
    """Two-sided exact binomial sign test against p=0.5, ties excluded."""
    positive = sum(1 for v in values if sign(v) > 0)
    negative = sum(1 for v in values if sign(v) < 0)
    trials = positive + negative
    result = {"positive": positive, "negative": negative,
              "ties": len(values) - trials, "nonties": trials}
    if not trials:
        return {**result, "p_value": None}
    tail = sum(math.comb(trials, k) for k in range(min(positive, negative) + 1)) / (2 ** trials)
    return {**result, "p_value": min(1.0, 2 * tail)}


def distribution(values):
    ordered = sorted(values)
    quartiles = statistics.quantiles(ordered, n=4) if len(ordered) > 1 else [ordered[0]] * 3
    return {"count": len(ordered), "mean": statistics.fmean(ordered),
            "median": statistics.median(ordered), "p25": quartiles[0], "p75": quartiles[2],
            "min": ordered[0], "max": ordered[-1]}


def correlation(xs, ys):
    """Pearson correlation and the least-squares slope of ys on xs."""
    if len(xs) < 3:
        return {"pearson_r": None, "r_squared": None, "slope": None}
    mean_x, mean_y = statistics.fmean(xs), statistics.fmean(ys)
    dx = [x - mean_x for x in xs]
    dy = [y - mean_y for y in ys]
    sxx = sum(v * v for v in dx)
    syy = sum(v * v for v in dy)
    sxy = sum(a * b for a, b in zip(dx, dy))
    if sxx <= 0 or syy <= 0:
        return {"pearson_r": None, "r_squared": None, "slope": None}
    r = sxy / math.sqrt(sxx * syy)
    return {"pearson_r": r, "r_squared": r * r, "slope": sxy / sxx}


def read_full_gaps(path):
    """Order-averaged full-chain gap totals per pair and checkpoint, in nats per chain."""
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"pair_id", "model_type", "order_index", "length", "full_nll_gap"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError("pair_scores.csv is missing columns: " + ", ".join(sorted(missing)))
        rows = list(reader)
    collected = defaultdict(dict)
    for row in rows:
        key = (row["pair_id"], row["model_type"])
        index = int(row["order_index"])
        if index in collected[key]:
            raise ValueError("Duplicate order_index for " + row["pair_id"])
        total = float(row["full_nll_gap"]) * int(row["length"])
        if not math.isfinite(total):
            raise ValueError("Nonfinite full_nll_gap for " + row["pair_id"])
        collected[key][index] = total
    gaps = {}
    for key, members in collected.items():
        if sorted(members) != list(range(1, len(members) + 1)):
            raise ValueError("Order indices are not a complete run for " + key[0])
        gaps[key] = {"delta_full_total": statistics.fmean(members.values()),
                     "order_count": len(members)}
    return gaps


def first_order_log_probs(model, features, torch, numpy):
    """Return an (L, 21) array of log p(aa_i | backbone) with no sequence context."""
    with torch.inference_mode():
        log_probs = model.unconditional_probs(
            features["X"], features["mask"], features["residue_idx"],
            features["chain_encoding_all"])
    length = features["X"].shape[1]
    if tuple(log_probs.shape) != (1, length, len(ALPHABET)):
        raise ValueError("Unconditional output has an unexpected shape")
    if not bool(torch.isfinite(log_probs).all().item()):
        raise ValueError("Unconditional output holds nonfinite log probabilities")
    totals = torch.logsumexp(log_probs, dim=-1)
    if float(torch.max(torch.abs(totals)).item()) > 1e-4:
        raise ValueError("Unconditional distributions are not normalized over the alphabet")
    return log_probs[0].detach().cpu().numpy().astype("float64")


def score_assignment(log_probs, sequence, positions=None):
    """Total first-order NLL over the chain, or over selected zero-based positions."""
    indices = range(len(sequence)) if positions is None else positions
    return float(sum(-log_probs[i, ALPHABET.index(sequence[i])] for i in indices))


def summarize(rows):
    by_model = defaultdict(list)
    for row in rows:
        by_model[row["model_type"]].append(row)
    result = {}
    for model_type in sorted(by_model):
        group = by_model[model_type]
        shares = [r["delta_fo_share"] for r in group if r["delta_fo_share"] is not None]
        result[model_type] = {
            "pair_count": len(group),
            "delta_full_total": distribution([r["delta_full_total"] for r in group]),
            "delta_fo_total": distribution([r["delta_fo_total"] for r in group]),
            "delta_joint_total": distribution([r["delta_joint_total"] for r in group]),
            "delta_fo_share_of_full": distribution(shares) if shares else None,
            "delta_full_sign_test": sign_test([r["delta_full_total"] for r in group]),
            "delta_fo_sign_test": sign_test([r["delta_fo_total"] for r in group]),
            "delta_joint_sign_test": sign_test([r["delta_joint_total"] for r in group]),
            "full_on_first_order_fit": correlation([r["delta_fo_total"] for r in group],
                                                   [r["delta_full_total"] for r in group]),
            "pairs_where_first_order_dominates": sum(1 for s in shares if s > 0.5),
            "pairs_where_joint_reverses_sign": sum(
                1 for r in group if sign(r["delta_full_total"]) and
                sign(r["delta_fo_total"]) == -sign(r["delta_full_total"])),
        }
    by_protein = defaultdict(list)
    for row in rows:
        by_protein[(row["structure_id"], row["chain"], row["model_type"])].append(row)
    proteins = []
    for key in sorted(by_protein):
        group = by_protein[key]
        shares = [r["delta_fo_share"] for r in group if r["delta_fo_share"] is not None]
        proteins.append({
            "structure_id": key[0], "chain": key[1], "model_type": key[2],
            "split": group[0]["split"], "group": group[0]["group"], "pair_count": len(group),
            "delta_full_total_mean": statistics.fmean([r["delta_full_total"] for r in group]),
            "delta_fo_total_mean": statistics.fmean([r["delta_fo_total"] for r in group]),
            "delta_joint_total_mean": statistics.fmean([r["delta_joint_total"] for r in group]),
            "delta_fo_share_median": statistics.median(shares) if shares else None,
            "joint_positive_count": sum(1 for r in group if sign(r["delta_joint_total"]) > 0),
        })
    return result, proteins


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    metadata = output / "metadata"
    metadata.mkdir()
    (output / "inputs").mkdir()
    started = time.monotonic()
    summary = {"schema_version": 1, "purpose": "first_order_burial_control", "status": "running",
               "started_utc": baseline.utc_now(), "unconditional_forward_calls": 0,
               "invariant_checks": []}
    write_json(metadata / "summary.json", summary)
    rows = []
    try:
        models = args.models.split(",")
        if not models or len(set(models)) != len(models) or any(
                m not in ("vanilla", "soluble") for m in models):
            raise ValueError("models must be vanilla, soluble, or vanilla,soluble")
        pairs_path, mpnn_dir = args.pairs.resolve(), args.mpnn_dir.resolve()
        scores_path = args.scores.resolve()
        manifest, structure_sources = score_pairs.load_pairs(pairs_path)
        full_gaps = read_full_gaps(scores_path)
        (metadata / "pairs.json").write_bytes(pairs_path.read_bytes())
        config = {"pairs_sha256": baseline.sha256(pairs_path),
                  "pair_scores_sha256": baseline.sha256(scores_path),
                  "upstream_commit": baseline.UPSTREAM_COMMIT, "model_name": baseline.MODEL,
                  "models": models, "device": args.device, "backbone_noise": 0,
                  "script_sha256": baseline.sha256(Path(__file__)),
                  "baseline_helper_sha256": baseline.sha256(Path(baseline.__file__)),
                  "score_pairs_sha256": baseline.sha256(Path(score_pairs.__file__)),
                  "upstream_files": {}, "alphabet": ALPHABET,
                  "first_order_definition":
                      "Total negative log probability in natural-log units under "
                      "ProteinMPNN.unconditional_probs, which applies an all-zero backward order "
                      "mask so no sequence context reaches any position",
                  "order_policy":
                      "The first-order score takes no decoding order and is order-free; "
                      "delta_full is the arithmetic mean of per-order full-chain totals from the "
                      "referenced scored run",
                  "decomposition": "delta_joint = delta_full - delta_fo, all as nats per chain",
                  "joint_term_meaning":
                      "Residual gap not explained by position-specific marginal preference; it "
                      "carries both genuine sequence coupling and any amortization error in the "
                      "autoregressive conditionals, so it is not a pure coupling estimate",
                  "invariant":
                      "Outside-patch positions must contribute exactly zero to delta_fo because "
                      "their identities are unchanged and unconditional probabilities ignore "
                      "sequence context"}
        for relative, expected in baseline.UPSTREAM_HASHES.items():
            actual = baseline.sha256(mpnn_dir / relative)
            if actual != expected:
                raise ValueError("Pinned model/code checksum mismatch: " + relative)
            config["upstream_files"][relative] = {
                "sha256": actual, "bytes": (mpnn_dir / relative).stat().st_size}
        write_json(metadata / "config.json", config)
        os.environ["OMP_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = "2"
        if args.device == "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        write_json(metadata / "environment.json", baseline.device_preflight(args.device))
        import numpy as np
        import torch
        sys.path.insert(0, str(mpnn_dir))
        utils = importlib.import_module("protein_mpnn_utils")
        prepared, features_by_key = [], {}
        for key, (item, source) in structure_sources.items():
            stem = "_".join(key)
            staged = output / "inputs" / (stem + ".pdb")
            info = baseline.stage_backbone(source, key[1], staged)
            parsed = utils.parse_PDB(staged.as_posix(), input_chain_list=[key[1]])
            if len(parsed) != 1 or parsed[0]["seq"] != info["native_sequence"]:
                raise ValueError("Upstream parser/native mapping mismatch: " + stem)
            packed = utils.tied_featurize([copy.deepcopy(parsed[0])], args.device,
                                          {parsed[0]["name"]: ([key[1]], [])})
            X, S, mask, chain_M = packed[0], packed[1], packed[2], packed[4]
            encoding, chain_M_pos, residue_idx = packed[5], packed[10], packed[12]
            length = info["length"]
            if X.shape != (1, length, 4, 3) or not bool(torch.isfinite(X).all().item()):
                raise ValueError("Incomplete or invalid backbone features: " + stem)
            if not torch.equal(mask, torch.ones_like(mask)) or not torch.equal(
                    chain_M * chain_M_pos, mask):
                raise ValueError("Expected all observed residues present and scored: " + stem)
            if "".join(ALPHABET[i] for i in S[0].tolist()) != info["native_sequence"]:
                raise ValueError("Featurization sequence mapping mismatch: " + stem)
            for pair in manifest["pairs"]:
                if (pair["structure_id"], pair["chain"]) == key:
                    score_pairs.validate_pair(pair, length, info["native_sequence"])
            info.update({"structure_id": key[0], "chain": key[1], "split": item["split"],
                         "group": item["group"], "raw_sha256": item["sha256"],
                         "staged_sha256": baseline.sha256(staged),
                         "staged_path": staged.relative_to(output).as_posix()})
            prepared.append(info)
            features_by_key[key] = {"X": X, "mask": mask, "residue_idx": residue_idx,
                                    "chain_encoding_all": encoding}
        write_json(metadata / "structures.json", prepared)
        by_key = {(r["structure_id"], r["chain"]): r for r in prepared}
        for model_type in models:
            checkpoint = torch.load(
                mpnn_dir / (model_type + "_model_weights") / (baseline.MODEL + ".pt"),
                map_location=args.device, weights_only=True)
            model = utils.ProteinMPNN(
                num_letters=21, node_features=128, edge_features=128, hidden_dim=128,
                num_encoder_layers=3, num_decoder_layers=3, augment_eps=0.0,
                k_neighbors=checkpoint["num_edges"], ca_only=False).to(args.device)
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            model.eval()
            model.requires_grad_(False)
            if model.training or model.features.augment_eps != 0:
                raise ValueError("Scoring requires evaluation mode and zero backbone noise")
            marginals = {}
            for key, features in features_by_key.items():
                marginals[key] = first_order_log_probs(model, features, torch, np)
                summary["unconditional_forward_calls"] += 1
                # A repeat pass must agree exactly; the first-order term has no sampling.
                repeat = first_order_log_probs(model, features, torch, np)
                summary["unconditional_forward_calls"] += 1
                drift = float(np.max(np.abs(marginals[key] - repeat)))
                if drift > TOLERANCE:
                    raise ValueError("Unconditional probabilities are not deterministic")
                summary["invariant_checks"].append({
                    "structure_id": key[0], "chain": key[1], "model_type": model_type,
                    "check": "deterministic_repeat", "max_abs_difference": drift,
                    "status": "passed"})
            for pair in manifest["pairs"]:
                key = (pair["structure_id"], pair["chain"])
                structure, log_probs = by_key[key], marginals[key]
                reference = full_gaps.get((pair["pair_id"], model_type))
                if reference is None:
                    raise ValueError("No scored full-chain gap for " + pair["pair_id"])
                positions = [i - 1 for i in pair["patch_indices"]]
                chain_totals, patch_totals = {}, {}
                for label in ("original", "permuted"):
                    sequence = pair[label + "_sequence"]
                    chain_totals[label] = score_assignment(log_probs, sequence)
                    patch_totals[label] = score_assignment(log_probs, sequence, positions)
                delta_fo = chain_totals["permuted"] - chain_totals["original"]
                delta_fo_patch = patch_totals["permuted"] - patch_totals["original"]
                outside = delta_fo - delta_fo_patch
                if abs(outside) > INVARIANT_TOLERANCE:
                    raise ValueError(
                        "Outside-patch first-order gap is nonzero for " + pair["pair_id"])
                delta_full = reference["delta_full_total"]
                delta_joint = delta_full - delta_fo
                length = structure["length"]
                rows.append({
                    "pair_id": pair["pair_id"], "structure_id": key[0], "chain": key[1],
                    "split": structure["split"], "group": structure["group"],
                    "patch_id": pair["patch_id"], "model_type": model_type,
                    "model_name": baseline.MODEL, "length": length,
                    "patch_size": len(positions),
                    "changed_identity_count": score_pairs.validate_pair(pair),
                    "original_fo_total": chain_totals["original"],
                    "permuted_fo_total": chain_totals["permuted"],
                    "delta_fo_total": delta_fo,
                    "original_fo_patch_total": patch_totals["original"],
                    "permuted_fo_patch_total": patch_totals["permuted"],
                    "delta_fo_patch_total": delta_fo_patch,
                    "outside_patch_fo_gap": outside,
                    "delta_full_total": delta_full, "delta_joint_total": delta_joint,
                    # Signed share, so a value above 1 means the first-order term overshoots
                    # and a negative value means it points the other way.
                    "delta_fo_share": delta_fo / delta_full
                    if abs(delta_full) > TOLERANCE else None,
                    "delta_fo_per_residue": delta_fo / length,
                    "delta_joint_per_residue": delta_joint / length,
                    "order_count": reference["order_count"],
                })
        if len(rows) != len(manifest["pairs"]) * len(models):
            raise ValueError("Completed row count differs from plan")
        checkpoints, proteins = summarize(rows)
        write_csv(output / "first_order_scores.csv", SCORE_FIELDS, rows)
        write_json(output / "first_order_summary.json", {
            "schema_version": 1, "purpose": "first_order_burial_control",
            "units": "nats per chain unless a field name says otherwise",
            "pair_count": len(manifest["pairs"]), "row_count": len(rows),
            "checkpoints": checkpoints, "proteins": proteins})
        summary.update({"status": "complete", "row_count": len(rows),
                        "scores_sha256": baseline.sha256(output / "first_order_scores.csv")})
    except BaseException as error:
        summary["status"] = "failed"
        summary["error"] = {"type": type(error).__name__, "message": str(error)}
        (metadata / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        summary["finished_utc"] = baseline.utc_now()
        summary["elapsed_seconds"] = time.monotonic() - started
        write_json(metadata / "summary.json", summary)
    for model_type, block in checkpoints.items():
        share = block["delta_fo_share_of_full"]
        print("{}: delta_full median {:.3f}, delta_fo median {:.3f}, delta_joint median {:.3f} "
              "nats".format(model_type, block["delta_full_total"]["median"],
                            block["delta_fo_total"]["median"],
                            block["delta_joint_total"]["median"]), flush=True)
        print("    first-order share median {:.3f}; R^2 of full on first-order {:.3f}".format(
            share["median"] if share else float("nan"),
            block["full_on_first_order_fit"]["r_squared"] or float("nan")), flush=True)
        joint = block["delta_joint_sign_test"]
        print("    joint term: {} positive, {} negative, {} ties, p = {}".format(
            joint["positive"], joint["negative"], joint["ties"],
            "n/a" if joint["p_value"] is None else "{:.3g}".format(joint["p_value"])), flush=True)
    print("Complete: " + str(output / "first_order_summary.json"), flush=True)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--mpnn-dir", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True,
                        help="pair_scores.csv from the scored paired run")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--models", default="vanilla,soluble")
    run(parser.parse_args())


if __name__ == "__main__":
    main()

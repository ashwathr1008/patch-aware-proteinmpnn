"""Score composition-matched pairs with directly verified shared decoding orders."""

import argparse
from collections import Counter, defaultdict
import copy
import csv
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import random
import re
import statistics
import sys
import time
import traceback
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "baseline"))
import run_baseline as baseline

ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
CANONICAL = frozenset(ALPHABET[:-1])
TOLERANCE = 1e-6
SCORE_FIELDS = [
    "pair_id", "structure_id", "chain", "split", "group", "patch_id",
    "model_type", "model_name", "order_index", "order_sha256", "length",
    "patch_size", "changed_identity_count", "presentation_first",
    "original_full_nll", "permuted_full_nll", "full_nll_gap",
    "original_patch_nll", "permuted_patch_nll", "patch_nll_gap",
]
SUMMARY_FIELDS = [
    "pair_id", "structure_id", "chain", "split", "group", "patch_id",
    "model_type", "model_name", "length", "patch_size", "changed_identity_count",
    "order_count", "original_full_nll_mean", "permuted_full_nll_mean",
    "full_nll_gap_mean", "full_nll_gap_min", "full_nll_gap_max", "full_nll_gap_std",
    "fraction_native_preferred", "original_patch_nll_mean", "permuted_patch_nll_mean",
    "patch_nll_gap_mean",
]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def stable_seed(*parts):
    raw = json.dumps(parts, separators=(",", ":")).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")


def validate_pair(pair, length=None, native=None):
    for field in ("pair_id", "structure_id", "patch_id"):
        if not isinstance(pair.get(field), str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", pair[field]):
            raise ValueError("Invalid or missing " + field)
    left, right = pair.get("original_sequence"), pair.get("permuted_sequence")
    if not isinstance(left, str) or not isinstance(right, str) or not left:
        raise ValueError("Pair requires nonempty sequence strings")
    if len(left) != len(right) or (length is not None and len(left) != length):
        raise ValueError("Pair sequence length mismatch")
    if not set(left + right) <= CANONICAL:
        raise ValueError("Pair sequences must be canonical amino acids")
    if native is not None and left != native:
        raise ValueError("Original sequence does not match parsed native backbone")
    indices = pair.get("patch_indices")
    if not isinstance(indices, list) or len(indices) < 2:
        raise ValueError("patch_indices must contain at least two one-based positions")
    if any(type(i) is not int or not 1 <= i <= len(left) for i in indices):
        raise ValueError("Invalid one-based patch position")
    if len(set(indices)) != len(indices):
        raise ValueError("Duplicate patch positions")
    selected = {i - 1 for i in indices}
    if any(a != b for i, (a, b) in enumerate(zip(left, right)) if i not in selected):
        raise ValueError("An outside-patch identity changed")
    if Counter(left[i] for i in selected) != Counter(right[i] for i in selected):
        raise ValueError("Patch amino acid composition differs")
    changed = sum(a != b for a, b in zip(left, right))
    if changed == 0:
        raise ValueError("Identity pairs are audit controls, not experimental permutations")
    return changed


def load_pairs(path):
    manifest = json.loads(path.read_text(encoding="utf-8-sig"))
    if manifest.get("schema_version") != 1 or manifest.get("purpose") != "patch_compatibility_pilot":
        raise ValueError("Expected schema_version=1, purpose='patch_compatibility_pilot'")
    structures, pairs = manifest.get("structures"), manifest.get("pairs")
    if not isinstance(structures, list) or not structures or not isinstance(pairs, list) or not pairs:
        raise ValueError("Nonempty structures and pairs lists are required")
    by_key = {}
    for item in structures:
        identifier, chain = item.get("structure_id"), item.get("chain")
        if not isinstance(identifier, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", identifier):
            raise ValueError("Invalid structure identifier")
        if not isinstance(chain, str) or not re.fullmatch(r"[A-Za-z0-9]", chain):
            raise ValueError("A single alphanumeric PDB chain is required")
        key = (identifier, chain)
        if key in by_key:
            raise ValueError("Duplicate structure/chain")
        if not all(isinstance(item.get(field), str) and item[field] for field in ("split", "group")):
            raise ValueError("Every structure needs explicit split/group labels")
        source = (path.parent / item["pdb_path"]).resolve()
        expected = item.get("sha256", "")
        if not re.fullmatch(r"[a-f0-9]{64}", expected) or baseline.sha256(source) != expected:
            raise ValueError("Structure input checksum mismatch: " + identifier)
        by_key[key] = (item, source)
    seen = set()
    for pair in pairs:
        validate_pair(pair)
        if pair["pair_id"] in seen:
            raise ValueError("Duplicate pair_id")
        seen.add(pair["pair_id"])
        if (pair["structure_id"], pair.get("chain")) not in by_key:
            raise ValueError("Pair refers to absent structure/chain")
    referenced = {(p["structure_id"], p["chain"]) for p in pairs}
    if referenced != set(by_key):
        raise ValueError("Every manifest structure must have at least one pair")
    return manifest, by_key


def create_orders(length, count, seed, identifier, chain):
    if length < 2 or count < 1 or count > math.factorial(length):
        raise ValueError("Invalid number of unique decoding orders")
    rng = random.Random(stable_seed("decoding_orders_v1", seed, identifier, chain))
    seen, orders = set(), []
    while len(orders) < count:
        values = list(range(length))
        rng.shuffle(values)
        key = tuple(values)
        if key in seen:
            continue
        seen.add(key)
        one_based = [x + 1 for x in values]
        encoded = json.dumps(one_based, separators=(",", ":")).encode("ascii")
        orders.append({"order_index": len(orders) + 1, "order_sha256": hashlib.sha256(encoded).hexdigest(),
                       "decoding_order_one_based": one_based})
    return orders


class OrderObserver:
    """Observe the actual order used to construct the pinned upstream causal mask."""
    def __init__(self, torch, expected):
        self.torch, self.expected = torch, expected
        self.calls = 0
        self.original = torch.nn.functional.one_hot

    def __call__(self, tensor, num_classes=-1):
        # Positional feature encodings also call one_hot, but have three axes.
        if tensor.ndim == 2 and num_classes == self.expected.shape[1]:
            if not self.torch.equal(tensor, self.expected):
                raise ValueError("Actual decoding-mask order differs from the recorded shared order")
            self.calls += 1
        return self.original(tensor, num_classes=num_classes)

    def require_observed(self):
        if self.calls != 1:
            raise ValueError("Expected exactly one observed decoding-mask construction")


def score_sequence(model, features, sequence, order, torch, explicit=True):
    length = len(sequence)
    token_ids = torch.tensor([[ALPHABET.index(a) for a in sequence]], dtype=torch.long,
                             device=features["X"].device)
    decoding = torch.tensor([[i - 1 for i in order["decoding_order_one_based"]]],
                            dtype=torch.long, device=token_ids.device)
    if tuple(decoding.shape) != (1, length) or sorted(decoding[0].tolist()) != list(range(length)):
        raise ValueError("Order is not a complete permutation of the sequence")
    observer = OrderObserver(torch, decoding)
    order_noise = torch.zeros_like(features["mask"])
    if not explicit:
        order_noise[0, decoding[0]] = torch.arange(1, length + 1, dtype=order_noise.dtype, device=order_noise.device)
    with torch.inference_mode(), patch.object(torch.nn.functional, "one_hot", observer):
        log_probs = model(features["X"], token_ids, features["mask"], features["chain_M"],
                          features["residue_idx"], features["chain_encoding_all"],
                          order_noise, use_input_decoding_order=explicit, decoding_order=decoding)
    observer.require_observed()
    if tuple(log_probs.shape) != (1, length, len(ALPHABET)) or not bool(torch.isfinite(log_probs).all().item()):
        raise ValueError("Model output has incorrect shape or nonfinite log probabilities")
    nll = -torch.gather(log_probs, -1, token_ids.unsqueeze(-1)).squeeze(-1)[0]
    return nll.detach().cpu().numpy().astype("float64")


def preference(gap):
    return 1.0 if gap > TOLERANCE else 0.0 if gap < -TOLERANCE else 0.5


def summarize_rows(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["pair_id"], row["model_type"])].append(row)
    result = []
    for key in sorted(groups):
        group = groups[key]
        first = group[0]
        record = {field: first[field] for field in SUMMARY_FIELDS[:11]}
        gaps = [r["full_nll_gap"] for r in group]
        record.update({"order_count": len(group), "full_nll_gap_min": min(gaps),
                       "full_nll_gap_max": max(gaps), "full_nll_gap_std": statistics.pstdev(gaps),
                       "fraction_native_preferred": statistics.mean(preference(g) for g in gaps)})
        for field in ("original_full_nll", "permuted_full_nll", "full_nll_gap",
                      "original_patch_nll", "permuted_patch_nll", "patch_nll_gap"):
            record[field + "_mean"] = statistics.mean(r[field] for r in group)
        result.append(record)
    return result


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    metadata = output / "metadata"
    metadata.mkdir()
    (output / "inputs").mkdir()
    started = time.monotonic()
    summary = {"schema_version": 1, "purpose": "patch_compatibility_pilot", "status": "running",
               "started_utc": baseline.utc_now(), "completed_score_rows": 0, "model_forward_calls": 0,
               "audits": [], "order_audit": {"method": "Observe actual tensor passed to one_hot for decoding mask; require exact shared order", "verified_forward_calls": 0}}
    write_json(metadata / "summary.json", summary)
    rows, residue_arrays = [], {}
    try:
        if args.orders < 1 or not 0 <= args.order_seed < 2 ** 32:
            raise ValueError("orders must be positive and seed must be uint32")
        models = args.models.split(",")
        if not models or len(set(models)) != len(models) or any(m not in ("vanilla", "soluble") for m in models):
            raise ValueError("models must be vanilla, soluble, or vanilla,soluble")
        pairs_path, mpnn_dir = args.pairs.resolve(), args.mpnn_dir.resolve()
        manifest, structure_sources = load_pairs(pairs_path)
        (metadata / "pairs.json").write_bytes(pairs_path.read_bytes())
        config = {"pairs_sha256": baseline.sha256(pairs_path), "upstream_commit": baseline.UPSTREAM_COMMIT,
                  "model_name": baseline.MODEL, "models": models, "orders": args.orders,
                  "order_seed": args.order_seed, "tie_tolerance": TOLERANCE, "backbone_noise": 0,
                  "device": args.device, "script_sha256": baseline.sha256(Path(__file__)),
                  "baseline_helper_sha256": baseline.sha256(Path(baseline.__file__)), "upstream_files": {},
                  "score_definition": "Mean negative log probability in natural-log units over the complete chain under each explicit autoregressive order; gap = permuted minus original",
                  "patch_score_definition": "Mean of full-chain autoregressive NLL terms at patch positions; not a standalone patch likelihood or fixed-outside-context conditional probability",
                  "order_average_definition": "Arithmetic mean of per-order NLLs, not negative log of an order-marginalized sequence probability",
                  "outside_patch_policy": "Identities are unchanged; all positions participate in each full-chain autoregressive order, so sequence-context effects can propagate outside the patch",
                  "alphabet": ALPHABET, "inference_temperature": "Unscaled model probabilities; no sampling or temperature transformation",
                  "presentation_policy": "Deterministically randomized member order independently of scoring; reverse-order audit on first pair per structure/checkpoint for every decoding order"}
        for relative, expected in baseline.UPSTREAM_HASHES.items():
            actual = baseline.sha256(mpnn_dir / relative)
            if actual != expected:
                raise ValueError("Pinned model/code checksum mismatch: " + relative)
            config["upstream_files"][relative] = {"sha256": actual, "bytes": (mpnn_dir / relative).stat().st_size}
        write_json(metadata / "config.json", config)
        os.environ["OMP_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = "2"
        if args.device == "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        environment = baseline.device_preflight(args.device)
        write_json(metadata / "environment.json", environment)
        import numpy as np
        import torch
        sys.path.insert(0, str(mpnn_dir))
        utils = importlib.import_module("protein_mpnn_utils")
        prepared, features_by_key, orders_by_key = [], {}, {}
        for key, (item, source) in structure_sources.items():
            stem = "_".join(key)
            staged = output / "inputs" / (stem + ".pdb")
            info = baseline.stage_backbone(source, key[1], staged)
            parsed = utils.parse_PDB(staged.as_posix(), input_chain_list=[key[1]])
            if len(parsed) != 1 or parsed[0]["seq"] != info["native_sequence"]:
                raise ValueError("Upstream parser/native mapping mismatch; gaps are unsupported: " + stem)
            packed = utils.tied_featurize([copy.deepcopy(parsed[0])], args.device, {parsed[0]["name"]: ([key[1]], [])})
            X, S, mask, lengths, chain_M, encoding = packed[:6]
            chain_M_pos, residue_idx = packed[10], packed[12]
            length = info["length"]
            if X.shape != (1, length, 4, 3) or not bool(torch.isfinite(X).all().item()):
                raise ValueError("Incomplete or invalid backbone features")
            if not torch.equal(mask, torch.ones_like(mask)) or not torch.equal(chain_M * chain_M_pos, mask):
                raise ValueError("Expected all observed residues present and scored")
            if "".join(ALPHABET[i] for i in S[0].tolist()) != info["native_sequence"]:
                raise ValueError("Featurization sequence mapping mismatch")
            for pair in manifest["pairs"]:
                if (pair["structure_id"], pair["chain"]) == key:
                    validate_pair(pair, length, info["native_sequence"])
            info.update({"structure_id": key[0], "chain": key[1], "split": item["split"], "group": item["group"],
                         "raw_sha256": item["sha256"], "staged_sha256": baseline.sha256(staged),
                         "staged_path": staged.relative_to(output).as_posix()})
            prepared.append(info)
            features_by_key[key] = {"X": X, "mask": mask, "chain_M": chain_M * chain_M_pos,
                                    "residue_idx": residue_idx, "chain_encoding_all": encoding}
            orders_by_key[key] = create_orders(length, args.orders, args.order_seed, *key)
        write_json(metadata / "structures.json", prepared)
        write_json(metadata / "orders.json", [{"structure_id": key[0], "chain": key[1], "orders": orders}
                                               for key, orders in orders_by_key.items()])
        summary.update({"pair_count": len(manifest["pairs"]), "structure_count": len(prepared),
                        "model_types": models, "order_count": args.orders,
                        "expected_score_rows": len(manifest["pairs"]) * len(models) * args.orders})
        by_key = {(r["structure_id"], r["chain"]): r for r in prepared}
        for model_type in models:
            checkpoint = torch.load(mpnn_dir / (model_type + "_model_weights") / (baseline.MODEL + ".pt"),
                                    map_location=args.device, weights_only=True)
            model = utils.ProteinMPNN(num_letters=21, node_features=128, edge_features=128, hidden_dim=128,
                                     num_encoder_layers=3, num_decoder_layers=3, augment_eps=0.0,
                                     k_neighbors=checkpoint["num_edges"], ca_only=False).to(args.device)
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            model.eval()
            model.requires_grad_(False)
            if model.training or model.features.augment_eps != 0:
                raise ValueError("Scoring requires evaluation mode and zero backbone noise")
            audited = set()
            for pair in manifest["pairs"]:
                key = (pair["structure_id"], pair["chain"])
                structure, features = by_key[key], features_by_key[key]
                patch_positions = np.asarray(pair["patch_indices"], dtype=int) - 1
                do_audit = key not in audited
                for order in orders_by_key[key]:
                    labels = ["original", "permuted"]
                    random.Random(stable_seed("presentation_v1", args.order_seed, pair["pair_id"], model_type, order["order_index"])).shuffle(labels)
                    values = {}
                    for label in labels:
                        values[label] = score_sequence(model, features, pair[label + "_sequence"], order, torch)
                        summary["model_forward_calls"] += 1
                        summary["order_audit"]["verified_forward_calls"] += 1
                    if do_audit:
                        reversed_values = {}
                        for label in reversed(labels):
                            reversed_values[label] = score_sequence(model, features, pair[label + "_sequence"], order, torch)
                            summary["model_forward_calls"] += 1
                            summary["order_audit"]["verified_forward_calls"] += 1
                        # The repeated original is an identical-sequence null comparison.
                        null_delta = float(np.max(np.abs(values["original"] - reversed_values["original"])))
                        reverse_delta = max(float(np.max(np.abs(values[label] - reversed_values[label]))) for label in labels)
                        derived_delta = None
                        if order["order_index"] == 1:
                            derived = score_sequence(model, features, pair["original_sequence"], order, torch, explicit=False)
                            summary["model_forward_calls"] += 1
                            summary["order_audit"]["verified_forward_calls"] += 1
                            derived_delta = float(np.max(np.abs(values["original"] - derived)))
                            if derived_delta > TOLERANCE:
                                raise ValueError("Explicit and rank-derived decoding orders give different scores")
                        if max(null_delta, reverse_delta) > TOLERANCE:
                            raise ValueError("Identical-sequence or reversed-presentation audit failed")
                        summary["audits"].append({"structure_id": key[0], "chain": key[1], "model_type": model_type,
                            "pair_id": pair["pair_id"], "order_index": order["order_index"],
                            "identical_sequence_max_abs_difference": null_delta,
                            "reversed_presentation_max_abs_difference": reverse_delta,
                            "explicit_vs_rank_derived_max_abs_difference": derived_delta, "status": "passed"})
                    row = {"pair_id": pair["pair_id"], "structure_id": key[0], "chain": key[1],
                           "split": structure["split"], "group": structure["group"], "patch_id": pair["patch_id"],
                           "model_type": model_type, "model_name": baseline.MODEL, "order_index": order["order_index"],
                           "order_sha256": order["order_sha256"], "length": structure["length"],
                           "patch_size": len(patch_positions), "changed_identity_count": validate_pair(pair),
                           "presentation_first": labels[0]}
                    for label in ("original", "permuted"):
                        row[label + "_full_nll"] = float(values[label].mean())
                        row[label + "_patch_nll"] = float(values[label][patch_positions].mean())
                        residue_arrays["{}_{}_{}_{}".format(pair["pair_id"], model_type, order["order_index"], label)] = values[label]
                    for region in ("full", "patch"):
                        row[region + "_nll_gap"] = row["permuted_" + region + "_nll"] - row["original_" + region + "_nll"]
                    rows.append(row)
                    summary["completed_score_rows"] += 1
                audited.add(key)
                write_csv(output / "pair_scores.csv", SCORE_FIELDS, rows)
                write_json(metadata / "summary.json", summary)
                print("Validated {} / {}: {} shared orders".format(pair["pair_id"], model_type, args.orders), flush=True)
            del model
        if summary["completed_score_rows"] != summary["expected_score_rows"]:
            raise ValueError("Completed score row count differs from plan")
        write_csv(output / "pair_summary.csv", SUMMARY_FIELDS, summarize_rows(rows))
        np.savez_compressed(output / "per_residue_nll.npz", **residue_arrays)
        summary.update({"status": "complete", "pair_scores_sha256": baseline.sha256(output / "pair_scores.csv"),
                        "pair_summary_sha256": baseline.sha256(output / "pair_summary.csv"),
                        "per_residue_nll_sha256": baseline.sha256(output / "per_residue_nll.npz")})
    except BaseException as error:
        summary["status"] = "failed"
        summary["error"] = {"type": type(error).__name__, "message": str(error)}
        (metadata / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        summary["finished_utc"] = baseline.utc_now()
        summary["elapsed_seconds"] = time.monotonic() - started
        write_json(metadata / "summary.json", summary)
    print("Complete: " + str(output / "pair_summary.csv"), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--mpnn-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--orders", type=int, default=8)
    parser.add_argument("--order-seed", type=int, default=137)
    parser.add_argument("--models", default="vanilla,soluble")
    run(parser.parse_args())


if __name__ == "__main__":
    main()

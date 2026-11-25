"""Train/evaluate a head on audited experimental pairs. Run with --help."""
import argparse
import copy
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from scorer.data import load_dataset, backbone_features, fixed_orders
from scorer.predictor import load_backbone, StabilityHead, pair_predictions, weight_digest
from scorer.cache import FeatureCache, cached_predictions, precache
from baseline.run_baseline import sha256
from scoring.score_swaps import spearman


def metrics(rows):
    if not rows:
        return {"count": 0, "mae": None, "rmse": None, "within_protein": []}
    errors = [r["prediction"] - r["target"] for r in rows]
    groups = defaultdict(list)
    for row in rows:
        groups[row["parent_id"]].append(row)
    ranking = []
    for parent, group in sorted(groups.items()):
        targets, predictions = [r["target"] for r in group], [r["prediction"] for r in group]
        rho = spearman(targets, predictions) if len(group) >= 3 and len(set(targets)) > 1 and len(set(predictions)) > 1 else None
        ranking.append({"parent_id": parent, "count": len(group), "spearman": rho})
    valid = sorted(r["spearman"] for r in ranking if r["spearman"] is not None)
    import statistics
    return {"count": len(rows), "mae": sum(abs(e) for e in errors) / len(errors),
            "rmse": math.sqrt(sum(e * e for e in errors) / len(errors)),
            "median_within_protein_spearman": statistics.median(valid) if valid else None,
            "within_protein": ranking}


def evaluate(extractor, head, parents, pairs, orders, device, cache=None, batch_size=64):
    head.eval()
    rows = []
    with torch.no_grad():
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start:start + batch_size]
            if cache is None:
                predictions = [pair_predictions(extractor, head, backbone_features(parents[p["parent_id"]], device),
                               p["reference_sequence"], [p["variant_sequence"]], orders[p["parent_id"]]).item() for p in batch]
            else:
                count = len(orders[batch[0]["parent_id"]])
                if any(len(orders[p["parent_id"]]) != count for p in batch):
                    raise ValueError("Evaluation requires the same fixed order count for each parent")
                predictions = torch.stack([cached_predictions(cache, head, parents, batch, orders,
                                          [i] * len(batch), device) for i in range(count)]).mean(0).cpu().tolist()
            for pair, prediction in zip(batch, predictions):
                rows.append({"pair_id": pair["pair_id"], "parent_id": pair["parent_id"], "target": pair["target"],
                             "prediction": prediction, "is_swap": pair["is_swap"]})
    return {"all": metrics(rows), "composition_preserving_swaps": metrics([r for r in rows if r["is_swap"]]),
            "predictions": rows}


def run(args):
    if Path(args.output).exists():
        raise ValueError("Use a fresh output directory")
    raw, parents, pairs = load_dataset(args.data)
    partitions = {s: [p for p in pairs if p["split"] == s] for s in ("train", "validation", "test")}
    if any(not rows for rows in partitions.values()):
        raise ValueError("Nonempty homology-separated train/validation/test sets required")
    if args.epochs < 1:
        raise ValueError("epochs must be positive")
    batch_size = getattr(args, "batch_size", 64)
    if batch_size < 1:
        raise ValueError("batch-size must be positive")
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    extractor, _ = load_backbone(args.mpnn_dir, args.checkpoint, args.device)
    before = weight_digest(extractor.model)
    head = StabilityHead(extractor.width, args.hidden_width).to(args.device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.01)
    orders = {pid: fixed_orders(len(p["reference_sequence"]), args.orders, args.seed, pid) for pid, p in parents.items()}
    identity = {"weights": before, "checkpoint": sha256(args.checkpoint),
                "source": sha256(args.mpnn_dir / "protein_mpnn_utils.py"), "device": str(args.device),
                "coordinate_noise": 0, "features": "final_decoder_plus_actual_identity_float32"}
    cache = FeatureCache(getattr(args, "cache_dir", Path("Data/cache/stability_features")), extractor, identity,
                         feature_batch_size=getattr(args, "feature_batch_size", 8))
    try:
        if getattr(args, "precache_only", False):
            precache(cache, parents, pairs, orders, args.device)
            if weight_digest(extractor.model) != before:
                raise RuntimeError("ProteinMPNN changed during cache extraction")
            print(json.dumps({"cache_only": True, "cache": str(cache.path), **cache.stats}), flush=True)
            return
        fit_head(args, raw, parents, pairs, partitions, extractor, head, optimizer, orders, before, rng, cache, batch_size)
    finally:
        cache.close()


def fit_head(args, raw, parents, pairs, partitions, extractor, head, optimizer, orders, before, rng, cache, batch_size):
    best, best_state, history, gradient_seen = float("inf"), None, [], False
    for epoch in range(args.epochs):
        head.train()
        training = list(partitions["train"])
        rng.shuffle(training)
        loss_total = 0.0
        for start in range(0, len(training), batch_size):
            batch = training[start:start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            indices = [rng.randrange(len(orders[p["parent_id"]])) for p in batch]
            prediction = cached_predictions(cache, head, parents, batch, orders, indices, args.device)
            target = torch.tensor([p["target"] for p in batch], dtype=prediction.dtype, device=args.device)
            loss = torch.nn.functional.mse_loss(prediction, target)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            if any(p.grad is not None for p in extractor.parameters()):
                raise RuntimeError("Frozen ProteinMPNN received gradients")
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in head.parameters()):
                raise RuntimeError("Nonfinite head gradient")
            gradient_seen |= any(p.grad is not None and bool((p.grad != 0).any()) for p in head.parameters())
            optimizer.step()
            loss_total += loss.item() * len(batch)
        validation = evaluate(extractor, head, parents, partitions["validation"], orders, args.device, cache, batch_size)
        value = validation["all"]["mae"]
        history.append({"epoch": epoch + 1, "train_mse": loss_total / len(training), "validation_mae": value,
                        "feature_cache": dict(cache.stats)})
        print(json.dumps(history[-1]), flush=True)
        if value < best:
            best, best_state = value, copy.deepcopy(head.state_dict())
    after = weight_digest(extractor.model)
    if before != after or not gradient_seen:
        raise RuntimeError("Frozen-weight/head-gradient verification failed")
    head.load_state_dict(best_state)
    test = evaluate(extractor, head, parents, partitions["test"], orders, args.device, cache, batch_size)
    if weight_digest(extractor.model) != before:
        raise RuntimeError("ProteinMPNN changed during evaluation")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    config = {"schema_version": 1, "input_width": extractor.width, "hidden_width": args.hidden_width,
              "vocabulary": {aa: i for i, aa in enumerate(extractor.vocabulary)},
              "units": "kcal/mol", "sign_convention": "positive_destabilizing",
              "prediction": "variant_score_minus_reference_score", "seed": args.seed,
              "orders": orders, "order_count": args.orders, "epochs": args.epochs, "lr": args.lr,
              "optimizer": "AdamW", "weight_decay": 0.01, "loss": "MSE",
              "batch_size": batch_size, "feature_batch_size": cache.feature_batch_size,
              "feature_cache": str(cache.path), "feature_cache_stats": dict(cache.stats),
              "checkpoint_sha256": sha256(args.checkpoint), "dataset_sha256": sha256(args.data),
              "proteinmpnn_source_sha256": sha256(args.mpnn_dir / "protein_mpnn_utils.py"),
              "backbone_digest_before": before, "backbone_digest_after": after,
              "head_nonzero_gradient_verified": gradient_seen}
    torch.save({"head_state_dict": head.state_dict(), "config": config}, output / "head.pt")
    artifacts = {"config.json": config, "splits.json": {"homology": raw["homology"],
                "parents": [{k: v for k, v in p.items() if k != "coordinates"} for p in parents.values()],
                "pairs": [{k: p[k] for k in ("pair_id", "parent_id", "split")} for p in pairs]},
                 "history.json": history, "test_metrics.json": test, "dataset.json": raw}
    for name, value in artifacts.items():
        (output / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_predictor(artifact, mpnn_dir, checkpoint, device="cpu"):
    """Restore an inference head; reject a different pretrained backbone checkpoint."""
    bundle = torch.load(artifact, map_location=device, weights_only=True)
    config = bundle["config"]
    if sha256(checkpoint) != config["checkpoint_sha256"]:
        raise ValueError("Inference checkpoint differs from training checkpoint")
    extractor, _ = load_backbone(mpnn_dir, checkpoint, device)
    if {aa: i for i, aa in enumerate(extractor.vocabulary)} != config["vocabulary"] or extractor.width != config["input_width"]:
        raise ValueError("Feature/vocabulary mismatch")
    head = StabilityHead(config["input_width"], config["hidden_width"]).to(device)
    head.load_state_dict(bundle["head_state_dict"], strict=True)
    head.eval()
    return extractor, head, config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "mpnn-dir", "checkpoint", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--hidden-width", type=int, default=64)
    parser.add_argument("--orders", type=int, default=8)
    parser.add_argument("--seed", type=int, default=137)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=64, help="Number of labelled pairs per head update")
    parser.add_argument("--feature-batch-size", type=int, default=8, help="Unique sequences per frozen ProteinMPNN pass")
    parser.add_argument("--cache-dir", type=Path, default=Path("Data/cache/stability_features"))
    parser.add_argument("--precache-only", action="store_true", help="Build/resume all fixed-order features without training")
    run(parser.parse_args())

"""Time real v3 CPU extraction and cached head steps; save timings, not a trained model."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from baseline.run_baseline import sha256
from scorer.cache import FeatureCache, cached_predictions, precache
from scorer.data import load_dataset, backbone_features, fixed_orders
from scorer.predictor import load_backbone, StabilityHead, weight_digest


def run(args):
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    _, parents, pairs = load_dataset(args.data)
    rng = random.Random(137)
    torch.manual_seed(137)
    default_threads = torch.get_num_threads()
    extractor, _ = load_backbone(args.mpnn_dir, args.checkpoint, "cpu")
    before = weight_digest(extractor.model)
    orders = {pid: fixed_orders(len(p["reference_sequence"]), args.orders, 137, pid) for pid, p in parents.items()}
    partitions = {s: [p for p in pairs if p["split"] == s] for s in ("train", "validation", "test")}
    target = {s: rng.sample(partitions[s], n) for s, n in (("train", 7000), ("validation", 1500), ("test", 1500))}
    # No repartitioning: target/sample pairs retain original parent/homology assignments.
    training = rng.sample(target["train"], args.batch_size)
    validation = rng.sample(target["validation"], args.batch_size)
    middle = sorted(training, key=lambda p: len(p["reference_sequence"]))[len(training) // 2]
    pid = middle["parent_id"]
    candidates = sorted({p[f] for p in partitions["train"] if p["parent_id"] == pid
                         for f in ("reference_sequence", "variant_sequence")})[:args.feature_batch_size]
    features = backbone_features(parents[pid], "cpu")
    thread_timings = []
    for threads in sorted({1, 4, 8, default_threads}):
        torch.set_num_threads(threads)
        extractor(features, candidates, orders[pid][0])  # warm-up excluded
        begin = time.perf_counter()
        for _ in range(3):
            extractor(features, candidates, orders[pid][0])
        seconds = (time.perf_counter() - begin) / 3
        result = {"threads": threads, "batch_seconds": seconds, "sequence_seconds": seconds / len(candidates)}
        thread_timings.append(result)
        print(json.dumps({"thread_probe": result}), flush=True)
    best_threads = min(thread_timings, key=lambda r: r["batch_seconds"])["threads"]
    torch.set_num_threads(best_threads)
    print(json.dumps({"selected_threads": best_threads, "phase": "cold_cache_for_training_and_validation_samples"}), flush=True)
    identity = {"weights": before, "checkpoint": sha256(args.checkpoint),
                "source": sha256(args.mpnn_dir / "protein_mpnn_utils.py"), "device": "cpu",
                "coordinate_noise": 0, "features": "final_decoder_plus_actual_identity_float32"}
    cache = FeatureCache(output / "cache", extractor, identity, feature_batch_size=args.feature_batch_size, memory_entries=0)
    try:
        begin = time.perf_counter()
        precache(cache, parents, training + validation, orders, "cpu")
        cold_seconds = time.perf_counter() - begin
        misses = cache.stats["misses"]
        cold_stats = dict(cache.stats)
        # Disk-backed measurements deliberately bypass the RAM LRU.
        head = StabilityHead(extractor.width, 64)
        optimizer = torch.optim.AdamW(head.parameters(), lr=.001, weight_decay=.01)
        def update():
            indices = [rng.randrange(args.orders) for _ in training]
            optimizer.zero_grad(set_to_none=True)
            predictions = cached_predictions(cache, head, parents, training, orders, indices, "cpu")
            labels = torch.tensor([p["target"] for p in training])
            torch.nn.functional.mse_loss(predictions, labels).backward()
            optimizer.step()
        update()  # optimizer warm-up excluded
        print(json.dumps({"phase": "cached_head_updates", "steps": args.steps}), flush=True)
        begin = time.perf_counter()
        for _ in range(args.steps):
            update()
        update_seconds = (time.perf_counter() - begin) / args.steps
        head.eval()
        with torch.no_grad():
            begin = time.perf_counter()
            for order_index in range(args.orders):
                cached_predictions(cache, head, parents, validation, orders,
                                   [order_index] * len(validation), "cpu")
            evaluation_order_seconds = (time.perf_counter() - begin) / args.orders
        if cache.stats["misses"] != misses:
            raise RuntimeError("Warm benchmark unexpectedly extracted features")
        if weight_digest(extractor.model) != before or any(p.grad is not None for p in extractor.parameters()):
            raise RuntimeError("Frozen ProteinMPNN verification failed")
        if not any(p.grad is not None and bool((p.grad != 0).any()) for p in head.parameters()):
            raise RuntimeError("No head gradients observed")
        needed = defaultdict(set)
        for group in target.values():
            for pair in group:
                needed[pair["parent_id"]].update((pair["reference_sequence"], pair["variant_sequence"]))
        sequence_orders = sum(map(len, needed.values())) * args.orders
        cache_seconds = cold_seconds / misses * sequence_orders
        training_seconds = math.ceil(7000 / args.batch_size) * args.epochs * update_seconds
        validation_seconds = math.ceil(1500 / args.batch_size) * args.epochs * args.orders * evaluation_order_seconds
        test_seconds = math.ceil(1500 / args.batch_size) * args.orders * evaluation_order_seconds
        cached_total = training_seconds + validation_seconds + test_seconds
        raw_feature_bytes = sum(len(parents[pid]["reference_sequence"]) * len(entries)
                                for pid, entries in needed.items()) * args.orders * (extractor.width + 1) * 4
        report = {"purpose": "timing_only_not_model_evaluation", "dataset_sha256": sha256(args.data),
                  "checkpoint_sha256": sha256(args.checkpoint), "torch_version": str(torch.__version__),
                  "default_threads": default_threads, "selected_threads": best_threads, "thread_probe": thread_timings,
                  "orders": args.orders, "epochs": args.epochs, "batch_size": args.batch_size,
                  "feature_batch_size": args.feature_batch_size, "sample_training_pairs": len(training),
                  "sample_validation_pairs": len(validation), "timed_optimizer_steps": args.steps,
                  "sample_pair_ids": {"train": [p["pair_id"] for p in training], "validation": [p["pair_id"] for p in validation]},
                  "cold_cache_seconds": cold_seconds, "cold_cache_stats": cold_stats,
                  "seconds_per_cached_training_batch": update_seconds,
                  "seconds_per_cached_evaluation_batch_order": evaluation_order_seconds,
                  "target_pair_counts": {s: len(v) for s, v in target.items()},
                  "target_unique_sequence_order_entries": sequence_orders,
                  "estimated_feature_cache_GiB_before_SQLite_overhead": raw_feature_bytes / 1024**3,
                  "estimated_seconds": {"feature_extraction_and_cache": cache_seconds, "head_training": training_seconds,
                                        "validation_all_epochs": validation_seconds, "test_once": test_seconds,
                                        "total_with_warm_cache": cached_total, "total_from_empty_cache": cache_seconds + cached_total},
                  "frozen_weights_verified": True, "head_gradients_verified": True,
                  "warm_cache_new_extractions": cache.stats["misses"] - misses,
                  "limitations": "Small-sample extrapolation, not a completed 10000-pair run. Original v3 splits retained. Disk/thermal/background-load variation and parent mix affect timing; thread choice tested on one representative parent.",
                  "benchmark_elapsed_seconds": time.perf_counter() - started}
        (output / "timings.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({k: v for k, v in report.items() if k != "sample_pair_ids"}, indent=2), flush=True)
    finally:
        cache.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("Data/processed/vani_training_v3/stability_pairs.json"))
    parser.add_argument("--mpnn-dir", type=Path, default=Path("Runs/vendor/ProteinMPNN"))
    parser.add_argument("--checkpoint", type=Path, default=Path("Runs/vendor/ProteinMPNN/vanilla_model_weights/v_48_020.pt"))
    parser.add_argument("--output", type=Path, default=Path("Results") / ("scorer-cpu-benchmark-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")))
    parser.add_argument("--orders", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--feature-batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()
    if min(args.orders, args.epochs, args.batch_size, args.feature_batch_size, args.steps) < 1:
        parser.error("All sizes/counts must be positive")
    run(args)

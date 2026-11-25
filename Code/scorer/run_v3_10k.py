"""Launch the prepared 10k CPU plan: cache all orders first, then train the head."""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Code"))


def shared_test_data(training_data, extra_data):
    """Verify the common swap test has no family or sequence in mixed training/validation."""
    from scorer.data import load_dataset
    _, training_parents, training_pairs = load_dataset(training_data)
    _, parents, pairs = load_dataset(extra_data)
    heldout = [p for p in pairs if p["split"] == "test"]
    if not heldout or not all(p["is_swap"] for p in heldout):
        raise ValueError("Shared evaluation must contain labelled held-out swaps")
    non_test_groups = {p["homology_group"] for p in training_parents.values() if p["split"] != "test"}
    if any(parents[p["parent_id"]]["homology_group"] in non_test_groups for p in heldout):
        raise ValueError("Shared swap test family appears in mixed train/validation")
    training_sequences = {p[field] for p in training_pairs if p["split"] != "test"
                          for field in ("reference_sequence", "variant_sequence")}
    if any(p[field] in training_sequences for p in heldout for field in ("reference_sequence", "variant_sequence")):
        raise ValueError("Shared swap test sequence appears in mixed train/validation")
    return parents, heldout


def evaluate_shared_swaps(config, extra_data):
    from baseline.run_baseline import sha256
    from scorer.cache import FeatureCache
    from scorer.data import fixed_orders
    from scorer.predictor import weight_digest
    from scorer.train import load_predictor, evaluate
    parents, heldout = shared_test_data(config["data"], extra_data)
    extractor, head, saved = load_predictor(config["output"] / "head.pt", config["mpnn_dir"], config["checkpoint"], config["device"])
    digest = weight_digest(extractor.model)
    identity = {"weights": digest, "checkpoint": sha256(config["checkpoint"]),
                "source": sha256(config["mpnn_dir"] / "protein_mpnn_utils.py"), "device": str(config["device"]),
                "coordinate_noise": 0, "features": "final_decoder_plus_actual_identity_float32"}
    orders = {pid: fixed_orders(len(p["reference_sequence"]), saved["order_count"], saved["seed"], pid) for pid, p in parents.items()}
    cache = FeatureCache(config["cache_dir"], extractor, identity, feature_batch_size=config["feature_batch_size"])
    try:
        report = evaluate(extractor, head, parents, heldout, orders, config["device"], cache, config["batch_size"])
        report["evaluation_data_sha256"] = sha256(extra_data)
        report["purpose"] = "same held-out v3 swap pairs used to evaluate the swap-only head; no model selection"
        if weight_digest(extractor.model) != digest:
            raise RuntimeError("ProteinMPNN changed during shared evaluation")
        (config["output"] / "shared_v3_swap_test_metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"shared_v3_swap_test": report["all"]}), flush=True)
    finally:
        cache.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_suffix(".json"))
    parser.add_argument("--dry-run", action="store_true", help="Check paths/data hash and print settings; no extraction/training")
    parser.add_argument("--precache-only", action="store_true", help="Build/resume cache only")
    options = parser.parse_args()
    config = json.loads(options.config.read_text())
    extra_data = config.pop("extra_evaluation_data", None)
    if extra_data:
        extra_data = (ROOT / extra_data).resolve()
        if not extra_data.is_file():
            raise ValueError("Missing shared evaluation data")
    for key in ("data", "mpnn_dir", "checkpoint", "output", "cache_dir"):
        config[key] = (ROOT / config[key]).resolve()
    from baseline.run_baseline import sha256
    summary = json.loads((config["data"].parent / "summary.json").read_text())
    if summary["pair_counts"] != {"train": 7000, "validation": 1500, "test": 1500}:
        raise ValueError("Expected prepared 7000/1500/1500 dataset")
    if sha256(config["data"]) != summary["manifest_sha256"]:
        raise ValueError("Prepared dataset differs from its validated summary")
    if not config["checkpoint"].is_file() or not (config["mpnn_dir"] / "protein_mpnn_utils.py").is_file():
        raise ValueError("Missing ProteinMPNN source/checkpoint")
    if config["output"].exists():
        raise ValueError("Output run already exists; choose a new output in the config")
    if extra_data:
        _, shared_pairs = shared_test_data(config["data"], extra_data)
        print(json.dumps({"shared_test_preflight_passed": True, "swap_test_pairs": len(shared_pairs)}), flush=True)
    print(json.dumps({"settings": config, "pair_counts": summary["pair_counts"], "extra_evaluation_data": extra_data,
                      "phases": ["build/resume all frozen features", "train head and evaluate"]}, default=str, indent=2), flush=True)
    if options.dry_run:
        return
    # Reuse the local matching-interpreter package cache if no PyTorch environment is active.
    if importlib.util.find_spec("torch") is None:
        cached_packages = ROOT / "Data/cache/scorer_test_packages"
        if cached_packages.exists():
            sys.path.insert(0, str(cached_packages))
    import torch
    from scorer.train import run
    torch.set_num_threads(config.pop("threads"))
    arguments = SimpleNamespace(**config, precache_only=True)
    status_path = config["output"].with_name(config["output"].name + ".status.json")
    status_path.parent.mkdir(parents=True, exist_ok=True)
    def status(phase, **details):
        status_path.write_text(json.dumps({"phase": phase, "updated_utc": datetime.now(timezone.utc).isoformat(),
                                          "output": str(config["output"]), **details}, indent=2) + "\n", encoding="utf-8")
    try:
        status("building_feature_cache")
        run(arguments)
        if not options.precache_only:
            status("training_head")
            arguments.precache_only = False
            run(arguments)
            if extra_data:
                status("evaluating_shared_v3_swaps")
                evaluate_shared_swaps(config, extra_data)
            status("complete")
        else:
            status("feature_cache_complete")
    except Exception as error:
        status("failed", error=str(error))
        raise


if __name__ == "__main__":
    main()

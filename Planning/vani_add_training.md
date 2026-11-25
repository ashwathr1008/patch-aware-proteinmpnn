## Stage 1: Prepare experimental labels

Filter MegaScale `_ML` measurements, verify mutation annotations, consolidate replicates, and normalize ΔΔG to kcal/mol with positive meaning destabilizing.  
Added: `Code/scorer/prepare_training.py`, `Code/scorer/requirements-preparation.txt`; produced: `Data/processed/vani_training_v2/measurements.sqlite` and `stage1_labels.json` (510,756 accepted rows; 497,198 consolidated measurements).

## Stage 2: Map structures and split proteins

Verify complete reference backbones and group parents before splitting, using ≥40% global edit similarity plus identical measured sequences; remote homology below that threshold is not ruled out.  
Added: mapping/clustering in `Code/scorer/prepare_training.py`; produced: `Data/processed/vani_training_v2/structures/` and `stage2_splits.json` (337 parents, 142 groups; train/validation/test: 236/50/51 parents).

## Stage 3: Build the training manifest

Build experimentally labelled reference–variant and measured reverse-swap pairs, preserve measurement provenance, and validate the manifest without training.  
Added: pair generation in `Code/scorer/prepare_training.py`, `Code/scorer/test_prepare_training.py`, and measured-swap support in `Code/scorer/data.py`; produced: `Data/processed/vani_training_v2/stability_pairs.json`, `stage3_manifest.json`, and `summary.json` (465,825 pairs, including 51,898 composition-preserving swaps).

## Stage 4: Train and evaluate the stability head

Training entry point: [Code/scorer/train.py](../Code/scorer/train.py); frozen ProteinMPNN extraction and trainable head: [Code/scorer/predictor.py](../Code/scorer/predictor.py); dataset loading: [Code/scorer/data.py](../Code/scorer/data.py); usage: [Code/scorer/README.md](../Code/scorer/README.md).

Added: training, evaluation, checkpoint-saving, and swapper-inference code in those files; planned output: `Results/stability-head/` containing `head.pt`, `config.json`, `splits.json`, `dataset.json`, `history.json`, and `test_metrics.json` (training has not started).

Frozen features are cached on disk and reused across epochs/runs, with batched extraction and padded MLP updates; `--precache-only` builds the cache without training.

Added: [Code/scorer/cache.py](../Code/scorer/cache.py), `Code/scorer/test_cache.py`, and batching/cache options in `Code/scorer/train.py`; planned cache: `Data/cache/stability_features/`; v3 CPU command: [Code/scorer/README.md](../Code/scorer/README.md#cached-cpu-training-and-batching).

## Swap-only v3 dataset

Keep only the 51,898 nonidentical, composition-preserving pairs from v2, retaining their experimental labels, pair IDs, and original parent/homology split assignments.

Added: [Code/scorer/prepare_swaps_only.py](../Code/scorer/prepare_swaps_only.py) and `Code/scorer/test_swaps_only.py`; produced: `Data/processed/vani_training_v3/stability_pairs.json`, checked `structures/`, subset `measurements.sqlite`, split audits, and `summary.json`; training uses `Code/scorer/train.py --data Data/processed/vani_training_v3/stability_pairs.json` with the usual checkpoint/output arguments.

## CPU timing benchmark

Measured 128 v3 sample pairs and 20 cached head updates on CPU; extrapolation for 10,000 pairs, 30 epochs, eight orders and batch size 64 is approximately four hours ten minutes from an empty cache, or six minutes twenty-three seconds with a complete cache (not a completed full training run).

Added: [Code/scorer/benchmark.py](../Code/scorer/benchmark.py); produced: [Analysis/scorer_training_data/cpu_benchmark_10000.json](../Analysis/scorer_training_data/cpu_benchmark_10000.json) and timing/cache artifacts in `Results/scorer-cpu-benchmark-20261125T053622Z/`; estimated full 10,000-pair cache payload is 5.3 GiB before database overhead.

## Prepared 10,000-pair CPU run

Select 7,000/1,500/1,500 swaps inside the original train/validation/test partitions, preserving all parent/homology assignments; launch cache preparation followed by 30 head-training epochs with `py -3.12 Code/scorer/run_v3_10k.py` (not started by setup).

Added: `Code/scorer/prepare_10k.py`, `Code/scorer/test_prepare_10k.py`, `Code/scorer/run_v3_10k.py`, and `Code/scorer/run_v3_10k.json`; produced: `Data/processed/vani_training_v3_10k/` and [Analysis/scorer_training_data/v3_10k_summary.json](../Analysis/scorer_training_data/v3_10k_summary.json) (10,000 swaps, 146 parents, 84 groups); planned output: `Results/stability-head-swaps-v3-10k/`.

## Mixed-data 10,000-pair run

Sample 7,000/1,500/1,500 pairs from mixed v2 without changing families or labels, then train with the same CPU settings and additionally evaluate on the original v3 swap test.

Added: mixed selection in `Code/scorer/prepare_10k.py`, `Code/scorer/run_mixed_10k.json`, shared-test/status support in `Code/scorer/run_v3_10k.py`, and `Code/scorer/test_shared_evaluation.py`; produced: `Data/processed/vani_training_v2_mixed_10k/` (10,000 pairs, 1,130 swaps); run output/logs: `Results/stability-head-mixed-v2-10k*`; common test output: `shared_v3_swap_test_metrics.json` inside the run directory after completion.

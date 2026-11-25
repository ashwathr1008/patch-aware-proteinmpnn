# Experimental stability predictor

This predictor is separate from `features.py` (conditional log probabilities) and
from frozen-model likelihood scoring. It captures the final decoder hidden state
at `W_out` using a temporary pre-hook, preserving the pinned upstream forward,
autoregressive masks and checkpoint weights. It concatenates the checkpoint's
21-symbol identity one-hot, applies a nonlinear residue MLP, masked mean pooling
over the complete chain and a shared scalar projection. Predictions are variant
minus reference scores, in **kcal/mol, positive = destabilizing**. Absolute scores
are not calibrated folding free energies. Only differences are supervised.

ProteinMPNN is frozen, always evaluated with coordinate noise zero, and run under
`torch.no_grad()`. Sequences are teacher-forced: no amino acids are sampled.
Training uses MSE and one fixed-set decoding order per pair per update; validation
and test average predicted differences over the entire shared fixed order set.
Best validation MAE selects the head; test is evaluated only after selection.
Metrics include MAE, RMSE and within-parent Spearman, both overall and for measured
composition-preserving pairs. Constant/undersized ranking groups return null.

## Data required before training

Stages 1–3 are implemented by `prepare_training.py`. Install
`requirements-preparation.txt`, then run `python Code/scorer/prepare_training.py`
from the repository root. The prepared dataset is
`Data/processed/vani_training_v2/stability_pairs.json`; summaries, a measurement
SQLite database with row provenance, checked PDBs, mapping exclusions and split
definitions are saved beside it. The output directory must be new. ProteinMPNN
is available at `Runs/vendor/ProteinMPNN`, at the pinned
revision, with `vanilla_model_weights/v_48_020.pt`; the real-checkpoint integration
test passes. Source verification normalizes Windows CRLF line endings to LF.
**No biological training or new accuracy claim has been made.** Existing raw
`deltaG` swap extraction is not automatically an eligible training cohort.
Preparation uses numeric `dG_ML` and `ddG_ML`, excludes censored/filtered values,
checks canonical sequences and mutation names (including dmutv and hnet scans),
and averages repeated measured `dG_ML` for each parent/sequence. Targets are
reference minus variant `dG_ML`, making positive values destabilizing. All source
rows and label/interval values are retained in `measurements.sqlite`. Never label
arbitrary swapper outputs or use ProteinMPNN likelihood as ground truth.

Preparation checks every mapped reference pair using global Levenshtein similarity
(`1 - edit_distance/max_length`), linking at >=0.40 by connected components. It
also merges parents sharing any identical accepted measured sequence, then assigns
entire groups approximately 70/15/15 with seed 7 **before** generating pairs. This
is a conservative sequence-similarity leakage guard, not proof that remote homology
is absent. The complete link audit and actual split sizes are saved in
`stage2_splits.json`. Custom manifests still require documented group provenance.

The input JSON schema is illustrated below (one parent/pair shown; actual training
requires nonempty train, validation and test partitions):

```json
{
  "schema_version": 1,
  "label_source": "dataset DOI/release and measured table",
  "quality_policy": "versioned filtering, replicate and censoring rules",
  "units": "kcal/mol",
  "sign_convention": "positive_stabilizing",
  "homology": {
    "method": "externally audited clustering method",
    "parameters": {"identity_threshold": 0.3, "coverage": 0.8},
    "audit_source": "path or identifier of clustering audit"
  },
  "parents": [{
    "parent_id": "parent1", "homology_group": "family1", "split": "train",
    "reference_sequence": "ACDE", "pdb_path": "parent1.pdb",
    "pdb_sha256": "REPLACE_WITH_SHA256", "chain": "A"
  }],
  "pairs": [{
    "pair_id": "measured1", "parent_id": "parent1",
    "reference_sequence": "ACDE", "variant_sequence": "CADE",
    "label_kind": "experimental_ddg", "measurement_id": "source row IDs",
    "ddg": 0.5
  }]
}
```

Here `ddg` is always the measured variant-minus-reference effect under the declared
source sign convention; positive-stabilizing input is negated. kJ/mol is divided
by 4.184. Parent sequence must exactly equal the selected complete canonical PDB
chain, with no omitted/unresolved residues. Indels and incomplete mappings fail.
Pairs can compare two experimentally measured reverse arrangements on the original
parent backbone: `pair_kind: measured_swap` requires measured row IDs for both
sequences. Such a pair need not have a mutant-specific structure. The full sequences
remain aligned to the verified parent chain. Homologous/identical parents must
stay together. Parent PDB paths are relative to
the manifest. Residue mappings, structure hashes and split assignments are saved.

## Run

For swap-only training, use
`--data Data/processed/vani_training_v3/stability_pairs.json` and a separate output
such as `Results/stability-head-swaps-v3`. Generate this subset reproducibly with
`python Code/scorer/prepare_swaps_only.py`. It retains all 51,898 actual
composition-preserving comparisons, including WT-to-swap pairs classified as
`reference_variant`; filtering only `pair_kind == measured_swap` would miss these.
Experimental labels and original v2 parent/homology splits are unchanged. Only
parents represented by retained pairs are included. The v2 mixed dataset remains
available for a separate comparison run.

Use a Python environment with the project's compatible PyTorch and NumPy builds:

```powershell
python Code/scorer/train.py --data Data/processed/vani_training_v2/stability_pairs.json --mpnn-dir Runs/vendor/ProteinMPNN --checkpoint Runs/vendor/ProteinMPNN/vanilla_model_weights/v_48_020.pt --output Results/stability-head --device cpu --orders 8
python -m unittest discover -s Code/scorer -p "test_*.py"
```

Contract tests use a tiny causal decoder and exercise one training/save/restore
cycle. To additionally verify capture against the real pretrained forward, set
`PROTEINMPNN_DIR` and `PROTEINMPNN_CHECKPOINT` before running tests. That integration
test otherwise explicitly skips. Identical-sequence differences can have numerical
roundoff at roughly floating-point precision; they should be zero within tolerance.

Use a fresh output directory. `head.pt` saves only head weights and configuration;
`config.json` records vocabulary mapping, checkpoint hash, fixed orders, dimensions,
seed, optimization settings and frozen-weight/head-gradient verification.
`splits.json`, `dataset.json`, `history.json` and `test_metrics.json` retain split,
source, mapping, selection and per-pair evaluation provenance. Backbone dimensions
and layer counts are inferred from the checkpoint; the upstream source must match
the repository pin. Only the standard full-backbone, 21-symbol checkpoint family
is supported; tensor shapes cannot establish a different vocabulary's semantics.

## Cached CPU training and batching

The prepared 10,000-pair run uses 7,000 training, 1,500 validation and 1,500 test
pairs from `Data/processed/vani_training_v3_10k/stability_pairs.json`. Pairs were
sampled separately inside the existing v3 partitions (seed 137): no parent or
homology group changes partition. Selecting some pairs from a protein does not
require keeping every other pair, provided all retained pairs preserve that
protein's original partition. Labels and pair IDs are unchanged. Reproduce the
selection with `python Code/scorer/prepare_10k.py` into a fresh output directory.

Run from the repository root:

```powershell
py -3.12 Code/scorer/run_v3_10k.py --dry-run
py -3.12 Code/scorer/run_v3_10k.py
```

The second command starts the complete run: build/resume the fixed eight-order
cache, then train the head and evaluate. Settings are recorded in
`Code/scorer/run_v3_10k.json`: CPU, 12 threads, 30 epochs, 64 pairs per head batch,
eight sequences per extraction batch. Output is `Results/stability-head-swaps-v3-10k/`.
`--precache-only` builds just the cache. Re-running resumes cache construction;
it does not resume interrupted head epochs or overwrite a completed run. The
launcher uses an active PyTorch installation, or the local matching-Python package
cache at `Data/cache/scorer_test_packages/` when available; that fallback is local
and ignored by Git. On another machine, install compatible PyTorch and NumPy first.

Training now uses a persistent SQLite feature cache by default. Only frozen decoder
features and masks are cached; experimental labels, head outputs and gradients are
never cached. Unique missing sequences on the same backbone and decoding order are
extracted together (`--feature-batch-size`, default 8). The head uses padded,
masked batches of labelled pairs (`--batch-size`, default 64). Each training pair
still draws one shared reference/variant order from its fixed set. Evaluation
averages scalar predictions over every shared order, not averaged feature vectors.

Start v3 CPU training with automatic, on-demand caching:

```powershell
python Code/scorer/train.py --data Data/processed/vani_training_v3/stability_pairs.json --mpnn-dir Runs/vendor/ProteinMPNN --checkpoint Runs/vendor/ProteinMPNN/vanilla_model_weights/v_48_020.pt --output Results/stability-head-swaps-v3 --device cpu --orders 8 --batch-size 64 --feature-batch-size 8 --cache-dir Data/cache/stability_features
```

To build/resume the complete feature cache separately, run that command with
`--precache-only` appended. It extracts every required sequence/order combination,
prints per-parent progress, and does not train or create the output run directory.
Then run the command without that flag. The first extraction is still expensive;
on-demand training may encounter new orders in later epochs. Once an entry exists,
subsequent epochs/runs reuse it without ProteinMPNN inference. Head training is not
resumable from interrupted epochs; only completed feature-cache batches resume.

Cache entries depend on the checkpoint weights/hash, upstream source, vocabulary,
PyTorch version, device, all backbone/mask/index tensors, full sequence and explicit
decoding order. Float32 values are saved without quantization. Checksums detect
damaged entries. A 256-entry RAM LRU bounds resident cache memory; remaining entries
stay on disk. The complete eight-order v3 cache can require tens of GB, so choose a
cache directory with sufficient free space. Smaller extraction batches reduce peak
ProteinMPNN memory without changing the selected pairs/orders. Use one writer per
cache directory at a time. `history.json` and `config.json` record cache hits/misses,
extraction calls and batch settings. Mini-batching changes the optimization schedule
from the old one-pair updates; numerical equivalence tests compare predictions and
gradients at fixed weights, not identical learning curves.

## Mixed-data 10k comparison

The mixed subset is `Data/processed/vani_training_v2_mixed_10k/stability_pairs.json`:
7,000 train, 1,500 validation and 1,500 test pairs, including 1,130 swaps overall
(737/198/195 respectively). Labels and source parent/group partitions are preserved.
It is a natural sample of the mixed data, not a swap-balanced dataset. Reproduce
it into a fresh directory with:

```powershell
py -3.12 Code/scorer/prepare_10k.py --mixed --source Data/processed/vani_training_v2 --output Data/processed/vani_training_v2_mixed_10k --report Analysis/scorer_training_data/mixed_10k_summary.json
```

Run the shared launcher with the separate mixed configuration:

```powershell
py -3.12 Code/scorer/run_v3_10k.py --config Code/scorer/run_mixed_10k.json --dry-run
py -3.12 Code/scorer/run_v3_10k.py --config Code/scorer/run_mixed_10k.json
```

Output is `Results/stability-head-mixed-v2-10k/`; the swap-only output is preserved.
The main checkpoint is selected on overall mixed validation MAE. After training,
an additional `shared_v3_swap_test_metrics.json` evaluates the mixed head on the
same 1,500 held-out swaps used for the previous swap-only run. Family and exact
sequence overlap with mixed training/validation are checked before launch and
evaluation. This extra test does not select or update the head. Compare that file
against the swap-only run's `test_metrics.json`; the two main test manifests differ.
The launcher's sibling `.status.json` records cache/training/shared-test phases
or failure. A background launch also has sibling `.stdout.log`, `.stderr.log`, and
`.process.json` files under `Results/`; do not launch a second writer concurrently.

## Score existing swapper candidates

From Python with `Code/` on `sys.path`:

```python
from scorer.train import load_predictor
from scorer.data import load_dataset, backbone_features, fixed_orders
from scorer.predictor import score_swapper_candidates

extractor, head, config = load_predictor("Results/stability-head/head.pt", mpnn_dir, checkpoint)
_, parents, _ = load_dataset("Data/processed/vani_training_v2/stability_pairs.json")
parent = parents[parent_id]  # already sequence/backbone verified
orders = config["orders"].get(parent_id) or fixed_orders(
    len(parent["reference_sequence"]), config["order_count"], config["seed"], parent_id)
predictions = score_swapper_candidates(
    extractor, head, backbone_features(parent, "cpu"),
    parent["reference_sequence"], swapper_pair_records, orders)
```

Candidates use the existing `original_sequence`, `permuted_sequence`, `pair_id`
and one-based `patch_indices` fields. Lower predicted ΔΔG is preferable under the
normalized convention; this is a prediction, not experimental validation.
Extraction batches the reference and candidates and is intended for sequential,
single-process use (temporary hooks must not share a model across concurrent calls).

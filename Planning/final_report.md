# Stability-head training: prepared datasets and CPU runs

## Complete Data/processed inventory

The local inventory contains **six dataset folders and 1,686 files**, including 1,642 reference-backbone PDB files. The [complete file index](data_processed_inventory.csv) lists every actual relative path, dataset, purpose, and size in bytes. Repeated backbones are counted separately in each dataset folder. No prepared data were removed or changed to make this inventory.

### Dataset versions: what was made and why

All pair counts below refer to labelled comparisons, not independent proteins. Experimental targets are in kcal/mol, positive meaning destabilization. The two 10k datasets each contain 7,000 training examples plus 3,000 held-out validation/test examples; they do not train on all 10,000 pairs.

| Folder under `Data/processed/` | Pairs | Composition-preserving pairs | Parents / groups | Why it was made | Training use |
| --- | ---: | ---: | --- | --- | --- |
| `vani_training/` | 306,501 | 0 | 337 / 142 | Initial preparation; the strict parser did not recognize scan prefixes and rejected double-mutant scan records. | Superseded diagnostic version; use v2 or later. |
| `vani_training_v1/` | 386,583 | 23,553 | 337 / 142 | Intermediate correction supporting `dmutv...` scan names; `hnet...` records were still excluded. | Superseded by the corrected v2 pool. |
| `vani_training_v2/` | 465,825 | 51,898 | 337 / 142 | Corrected full mixed pool with verified ordinary mutations and swaps, supporting both scan-name formats. | Full mixed training pool and source for derived datasets. |
| `vani_training_v3/` | 51,898 | 51,898 | 148 / 84 | Filter v2 to nonidentical sequences with exactly matching amino-acid counts, preserving experimental labels and splits. | Full swap-only training pool. |
| `vani_training_v3_10k/` | 10,000 | 10,000 | 146 / 84 | Smaller swap-only CPU experiment, sampled inside the existing partitions with seed 137. | Input to `Code/scorer/run_v3_10k.json`; the completed swap-only head used this dataset. |
| `vani_training_v2_mixed_10k/` | 10,000 | 1,130 | 337 / 142 | Equal-size mixed-data CPU comparison, naturally sampled from v2 with seed 137, without swap oversampling. | Input to `Code/scorer/run_mixed_10k.json`. |

The initial directory and v1 are historical outputs, not the current data-preparation defaults. Their counts differ because mutation-format handling was corrected; these are not alternative random seeds or trained-model versions.

### Pair split specifications

| Dataset folder | Train pairs | Validation pairs | Test pairs | Split rule |
| --- | ---: | ---: | ---: | --- |
| `vani_training/` | 206,738 | 49,596 | 50,167 | Parent/group assignment before pair generation; superseded cohort. |
| `vani_training_v1/` | 255,954 | 64,541 | 66,088 | Parent/group assignment before pair generation; intermediate cohort. |
| `vani_training_v2/` | 298,630 | 81,660 | 85,535 | Approximately 70/15/15 by parent counts, seed 7; not by pair counts. |
| `vani_training_v3/` | 29,631 | 10,548 | 11,719 | Filter v2 without reassigning any parent/group. |
| `vani_training_v3_10k/` | 7,000 | 1,500 | 1,500 | Sample separately inside the original partitions, seed 137. |
| `vani_training_v2_mixed_10k/` | 7,000 | 1,500 | 1,500 | Sample separately inside the original partitions, seed 137. |

For mixed 10k, the swap counts within train/validation/test are 737/198/195. Every pair in swap-only 10k is a swap. The mixed and swap-only main test sets contain different examples; use the common v3 test evaluation described below when comparing the heads' swap performance.

### What each prepared file does

Each path in the following table is relative to one dataset folder listed above. The exact folders containing each filename are listed in the next table and in the complete CSV index.

| File or pattern | Contents / reason it was saved | Read directly by head training? |
| --- | --- | --- |
| `stability_pairs.json` | Main training manifest: complete reference/variant sequences, experimental DDG labels, measurement IDs, parent backbone paths/hashes, and fixed parent/group/split definitions. Pass this file as `--data`. | **Yes.** |
| `structures/*.pdb` | Checked complete reference N/CA/C/O backbones. Reference and variant sequences use the same parent backbone; these are released AlphaFold predictions. | **Yes**, through paths in the manifest. |
| `measurements.sqlite` | Filtered stability measurements, replicate consolidation information, source names and CSV row provenance. Subsets keep measurements needed by their retained pairs. The full preparation databases also retain the filtered source `raw` table. | No; used for preparation and label provenance. |
| `source_checksums.json` | SHA256 values for the original MegaScale measurement and structure archives. | No; source verification audit. |
| `stage1_labels.json` | Filtering rules, accepted/excluded row counts, consolidated counts, and reference exclusions for a full preparation run. | No; filtering audit. |
| `mutation_audit.json` | Diagnostics explaining unsupported or rejected mutation annotations in the historical initial/v1 cohorts. | No; historical debugging audit. |
| `stage2_splits.json` | Sequence-similarity grouping, parent-to-structure mappings, partition definitions, and/or pointers to inherited full audits. | Parent/split fields are read from the manifest; this file is the separate audit. |
| `source_stage2_splits.json` | Copy of an inherited original parent/group audit, so derivation remains traceable. | No; ancestry audit. |
| `source_v3_splits.json` | Frozen source-v3 partition audit retained by the existing swap-only 10k dataset. | No; ancestry audit. |
| `source_partition_splits.json` | Frozen source partition audit retained by the mixed 10k dataset. | No; ancestry audit. |
| `stage3_manifest.json` | Pair-generation/filtering counts and final manifest checksum; some derived versions also include their validation summary here. | No; final pair audit. |
| `summary.json` | Dataset counts, validation status, source hashes and applicable selection settings. The launcher checks its manifest hash and partition counts. Its preparation status is not a live training status. | Read by the run launcher, not the head's feature path. |

### Exact file inventory by dataset folder

Each listed filename exists directly in that folder. `structures/*.pdb` represents all individual PDB files under its `structures/` subdirectory; every individual filename is enumerated in [data_processed_inventory.csv](data_processed_inventory.csv).

| Dataset folder | Files directly inside the folder | Backbone files | Total files |
| --- | --- | ---: | ---: |
| `vani_training/` | `measurements.sqlite`, `mutation_audit.json`, `source_checksums.json`, `stability_pairs.json`, `stage1_labels.json`, `stage2_splits.json`, `stage3_manifest.json`, `summary.json` | 337 | 345 |
| `vani_training_v1/` | `measurements.sqlite`, `mutation_audit.json`, `source_checksums.json`, `stability_pairs.json`, `stage1_labels.json`, `stage2_splits.json`, `stage3_manifest.json`, `summary.json` | 337 | 345 |
| `vani_training_v2/` | `measurements.sqlite`, `source_checksums.json`, `stability_pairs.json`, `stage1_labels.json`, `stage2_splits.json`, `stage3_manifest.json`, `summary.json` | 337 | 344 |
| `vani_training_v2_mixed_10k/` | `measurements.sqlite`, `source_checksums.json`, `source_partition_splits.json`, `source_stage2_splits.json`, `stability_pairs.json`, `stage2_splits.json`, `summary.json` | 337 | 344 |
| `vani_training_v3/` | `measurements.sqlite`, `source_checksums.json`, `source_stage2_splits.json`, `stability_pairs.json`, `stage2_splits.json`, `stage3_manifest.json`, `summary.json` | 148 | 155 |
| `vani_training_v3_10k/` | `measurements.sqlite`, `source_checksums.json`, `source_stage2_splits.json`, `source_v3_splits.json`, `stability_pairs.json`, `stage2_splits.json`, `summary.json` | 146 | 153 |

`Data/raw/megascale/` contains the source archives; `Data/cache/stability_features/` contains reusable frozen features; `Results/` contains trained `head.pt` files, histories, metrics and logs. Those are separate from the processed dataset inventory. Generated processed datasets are ignored by Git; this report and the CSV index document what exists locally without uploading the large artifacts.

## Dataset change: mixed v2 to swap-only v3

The v2 manifest combines ordinary reference-variant mutations and composition-preserving swaps. The v3 manifest retains only experimentally labelled pairs with different sequences and identical amino-acid counts. It preserves the source labels, pair IDs, parent assignments, and homology-group partitions; arbitrary generated swaps receive no labels.

| Dataset | Total pairs | Composition-preserving pairs | Manifest |
| --- | ---: | ---: | --- |
| Mixed v2 | 465,825 | 51,898 | `Data/processed/vani_training_v2/stability_pairs.json` |
| Swap-only v3 | 51,898 | 51,898 | `Data/processed/vani_training_v3/stability_pairs.json` |
| Prepared v3 10k subset | 10,000 | 10,000 | `Data/processed/vani_training_v3_10k/stability_pairs.json` |

V3 contains 51,606 records tagged `measured_swap` and 292 composition-preserving records tagged `reference_variant`. Selection checks the actual sequences, so it retains both categories. V3 includes 148 parents in 84 sequence-similarity groups, with 29,631 training, 10,548 validation, and 11,719 test pairs. Structures and a subset measurement database retain alignment and experimental provenance.

Preparation code: [prepare_training.py](../Code/scorer/prepare_training.py) for v2; [prepare_swaps_only.py](../Code/scorer/prepare_swaps_only.py) for v3. The [v3 summary](../Analysis/scorer_training_data/v3_summary.json) records the outputs.

## The 10,000-pair run and leakage controls

[prepare_10k.py](../Code/scorer/prepare_10k.py) samples without replacement separately inside the existing v3 partitions, with seed 137:

| Partition | Pairs | Represented parents |
| --- | ---: | ---: |
| Train | 7,000 | 95 |
| Validation | 1,500 | 28 |
| Test | 1,500 | 23 |
| Total | 10,000 | 146 |

All 84 original groups remain represented. A parent or family assigned to training stays in training; no retained pair from that family moves into validation or test. Subsampling does not require retaining every pair from a protein. The preparer verifies unchanged source records and labels, structure checksums, sequence alignment, and absence of cross-partition parent/group or identical-sequence leakage. See the [10k subset summary](../Analysis/scorer_training_data/v3_10k_summary.json).

The inherited grouping uses connected components at global edit similarity >=0.40, plus identical measured-sequence overlaps. This is a sequence-similarity leakage guard; it does not establish the absence of remote homology below the threshold.

## Model and training configuration

[predictor.py](../Code/scorer/predictor.py) extracts final decoder states immediately before the pretrained ProteinMPNN output projection and concatenates each residue's actual identity. ProteinMPNN remains frozen, in evaluation mode, with coordinate noise disabled and extraction under `torch.no_grad()`. A trainable nonlinear residue MLP, masked mean pooling, and scalar projection produce sequence scores. Predicted DDG is variant score minus reference score; targets are in kcal/mol, with positive meaning destabilization.

[train.py](../Code/scorer/train.py) trains only the head using MSE and AdamW. The [10k run configuration](../Code/scorer/run_v3_10k.json) specifies CPU, 12 threads, 30 epochs, learning rate 0.001, weight decay 0.01, hidden width 64, head batch size 64, feature-extraction batch size 8, and eight fixed decoding orders. Training samples one shared order per pair; validation and test average predictions over all eight orders. Best validation MAE selects the head; the held-out test reports MAE, RMSE, and within-protein ranking performance.

## Steps to run head training

Run all commands from the repository root, `D:\GenAI\patch-aware-proteinmpnn`.

1. **Check prerequisites.** Use Python 3.12 with compatible PyTorch and NumPy. The launcher can reuse the local `Data/cache/scorer_test_packages/` dependency cache if PyTorch is not installed in the active environment. That ignored cache is machine-specific. ProteinMPNN must be present at `Runs/vendor/ProteinMPNN`, at pinned revision `8907e6671bfbfc92303b5f79c4b5e6ce47cdef57`, with `vanilla_model_weights/v_48_020.pt`.

2. **Use the prepared dataset.** The 10k manifest and its checked structures/provenance already exist locally. Generated `Data/processed/` datasets and checkpoints are ignored by Git. On another checkout, restore them or regenerate from the verified MegaScale archives. Run the following preparation commands only when their output directories do not already exist:

   ```powershell
   py -3.12 Code/scorer/prepare_training.py
   py -3.12 Code/scorer/prepare_swaps_only.py
   py -3.12 Code/scorer/prepare_10k.py
   ```

   `prepare_training.py` additionally requires `Code/scorer/requirements-preparation.txt` installed in the active environment; the training launcher does not bootstrap this preparation dependency.

3. **Validate the run plan without starting computation.** This checks paths, the prepared manifest hash, and the 7,000/1,500/1,500 split counts:

   ```powershell
   py -3.12 Code/scorer/run_v3_10k.py --dry-run
   ```

4. **Start cache preparation followed by training and evaluation:**

   ```powershell
   py -3.12 Code/scorer/run_v3_10k.py
   ```

   The launcher first builds or resumes all required frozen features in `Data/cache/stability_features/`, then trains the head. To build the cache alone:

   ```powershell
   py -3.12 Code/scorer/run_v3_10k.py --precache-only
   ```

   Afterwards, run the command without that flag to reuse the cache and train. Completed cache batches survive interruption. Head epochs are not resumable; a completed output run is never overwritten. For a separate run, choose a fresh `output` path in `Code/scorer/run_v3_10k.json`.

5. **Inspect the outputs** in `Results/stability-head-swaps-v3-10k/`:

   | File | Contents |
   | --- | --- |
   | `head.pt` | Trained head weights and inference configuration; used alongside the frozen ProteinMPNN checkpoint |
   | `config.json` | Vocabulary, checkpoint/data hashes, orders, batch settings, and frozen-weight/head-gradient verification |
   | `splits.json`, `dataset.json` | Exact parent/group partitions and labelled input provenance |
   | `history.json` | Training MSE, validation MAE, and cache statistics per epoch |
   | `test_metrics.json` | Held-out metrics and per-pair predictions, including the composition-preserving subset |

## Timing and current status

The [CPU benchmark](../Analysis/scorer_training_data/cpu_benchmark_10000.json) measured 128 real v3 pairs and 20 cached head updates. Extrapolation for the configured 10k run is approximately four hours ten minutes from an empty cache, or six to ten minutes with a complete cache. Estimated cached feature payload is 5.3 GiB before database overhead. These are estimates; protein mix, disk performance, background load, and CPU throttling affect runtime.

The swap-only 10,000-pair run completed 30 epochs. The best validation MAE was 0.5744 kcal/mol at epoch 11; that selected head is saved at `Results/stability-head-swaps-v3-10k/head.pt`. On 1,500 held-out swaps across 23 proteins, MAE was 0.5368 kcal/mol, RMSE 0.6917 kcal/mol, and median within-protein Spearman 0.5070. Frozen-weight and head-gradient checks passed. Later epochs reduced training error but increased validation error; the checkpoint uses the best validation weights.

## Mixed-data 10,000-pair comparison

The separately prepared mixed subset is `Data/processed/vani_training_v2_mixed_10k/stability_pairs.json`. It contains 7,000 train, 1,500 validation and 1,500 test pairs across 337 parents and 142 groups, retaining original v2 family partitions and experimental labels. Its 1,130 composition-preserving swaps are distributed as 737 train, 198 validation and 195 test. The remaining 8,870 pairs change composition. See [mixed_10k_summary.json](../Analysis/scorer_training_data/mixed_10k_summary.json).

[run_mixed_10k.json](../Code/scorer/run_mixed_10k.json) uses the same CPU/head hyperparameters and shared feature cache as the swap-only run, but saves independently to `Results/stability-head-mixed-v2-10k/`. Launch with:

```powershell
py -3.12 Code/scorer/run_v3_10k.py --config Code/scorer/run_mixed_10k.json
```

The launcher builds/resumes missing features, trains for 30 epochs, and selects the checkpoint using overall mixed validation MAE. It reports the mixed test's overall and swap-subset metrics. It then evaluates the selected mixed head on the identical 1,500-pair v3 swap test, saving `shared_v3_swap_test_metrics.json`. That extra test is checked for family/sequence separation from mixed training/validation and is not used for model selection. Compare it with `Results/stability-head-swaps-v3-10k/test_metrics.json`; comparing the two runs' main overall test MAEs would mix different tasks and examples.

For a background run, progress is recorded in `Results/stability-head-mixed-v2-10k.stdout.log`, errors in `.stderr.log`, process details in `.process.json`, and pipeline phase in `.status.json`. The mixed checkpoint will be `Results/stability-head-mixed-v2-10k/head.pt` once training finishes.

# Analysis

The [controlled pair-scoring results](../Documentation/Midreport/controlled_scoring_results.md) extend the generation pilot. The first ubiquitin pair is scored in [paired_first/](paired_first/), with all per-order values, source-protein/group summaries, explicit decoding permutations, control audits and PNG/PDF figures. Both checkpoints prefer the original under all eight shared orders. This establishes sensitivity for one pair; it does not establish an added patch scorer's benefit.

The completed expansion is in [paired_expanded/](paired_expanded/). It contains 85 pairs from 17 patches/seven sources/six groups, scored by both checkpoints under eight shared orders: 1,360 paired order-level rows and 2,958 audited model forward calls. Both checkpoints prefer the original in 84/85 order-averaged comparisons; the same 1ENH pair is the exception and changes sign across orders. All composition/geometry controls tie. The simple hydrophobic-contact descriptor changes in 41/85 pairs, without a validated favorable direction. Read [per-protein results](paired_expanded/protein_summary.csv), [group/split summaries](paired_expanded/split_summary.csv), [paired gaps](paired_expanded/paired_gaps.pdf), [order ranges](paired_expanded/order_sensitivity.pdf), and [descriptor plot](paired_expanded/spatial_descriptor.pdf). Do not interpret pooled pair counts as 85 independent biological replicates.

The expanded gaps are decomposed in [decomposition/](decomposition/). ProteinMPNN's own unconditional probabilities supply a first-order burial score, `p(aa_i | backbone)` with no sequence context, and the order-averaged full-chain gap is split into that term and a joint residual. The first-order term accounts for a median **88.5% (vanilla) and 89.3% (soluble)** of the gap and predicts it with R² = 0.95, so most of the published effect is burial-weighted composition rather than arrangement. The joint residual is small but systematic, favouring the original in 74/85 and 71/85 pairs. The same directory measures decoding-order inconsistency from the existing 1,360 rows: the order spread is 6–7% of the effect, so the 84/85 result is not an artifact of decoding order, and the 1ENH exception is the one pair whose order spread exceeds its own gap. Read the [decomposition results](../Documentation/Midreport/decomposition_results.md), [per-pair terms](decomposition/first_order_scores.csv), [summary statistics](decomposition/first_order_summary.json), [decomposition figure](decomposition/decomposition.pdf) and [order-spread figure](decomposition/order_consistency.pdf). The joint residual absorbs amortization error as well as coupling, and is not a per-pair quantity.

Raw expanded run: `Results/paired-expanded-20261104`. Exact input: `Data/compatibility_pairs/pairs.json`, already shared with its small staged backbones and provenance. Replace `Data/first_pair/pairs.json` with this path in the scoring/export commands below to repeat the expansion. The initial single pair and expanded set are separate runs; the first pair is not an extra independent protein.

The November 4 **development baseline** generated 48 sequences across three structures and two checkpoints. These outputs establish pipeline feasibility; no patch benefit, folding, stability or solubility improvement has been tested.

| Artifact | Contents |
| --- | --- |
| [baseline_pilot/summary.csv](baseline_pilot/summary.csv) | Six groups; generated-only recovery, uniqueness, raw model scores and subprocess times |
| [baseline_pilot/candidates.csv](baseline_pilot/candidates.csv) | 48 samples and 6 explicitly marked native references |
| [Recovery figure](baseline_pilot/native_recovery.png), [PDF](baseline_pilot/native_recovery.pdf) | All eight samples and mean per group; recovery is sequence identity |
| [run_record.json](baseline_pilot/run_record.json) | Portable versions, input/checkpoint hashes, validations and timing |
| [Patch illustration](handoff_20261104/figures/patch_pair.png), [PDF](handoff_20261104/figures/patch_pair.pdf) | One fixed-backbone, composition-matched pair |
| [Patch inputs](handoff_20261104/patch_example/manifest.json) | Residue mappings, coordinates, original/permuted identities and seven invariant checks |
| [handoff_checks.json](handoff_20261104/handoff_checks.json) | All 76 residue mappings/304 backbone atoms match model input; same-machine repeat reproduced 16 generated sequences and scores |

Full primary run: `Results/baseline-local-20261104`; repeat: `Results/reproduce-1ubq-local-20261104`; patch: `Results/patch-demo-1ubq-seed7`. These directories are local and ignored. The two native references for each structure are the same native sequence scored by different checkpoints, not independent observations.

From the repository root, regenerate into new destinations:

```powershell
python Analysis/summarize_baseline.py --run-dir Results/baseline-local-20261104 --output Results/rebuilt-baseline-summary
python Analysis/check_handoff.py --baseline Results/baseline-local-20261104 --repeat Results/reproduce-1ubq-local-20261104 --patch Results/patch-demo-1ubq-seed7 --output Results/rebuilt-handoff
python Code/patches/visualize.py --manifest Results/rebuilt-handoff/patch_example/manifest.json --output Results/rebuilt-patch-figure
```

Use the project environment described in [Code/README.md](../Code/README.md). To reproduce the repeat run, make a temporary copy of the development manifest containing only 1UBQ, set its `pdb_path` relative to that temporary manifest (or absolute), then use the same baseline command/settings with a new output path. Different hardware/software may not yield identical floating-point results; this check establishes reproducibility only in the tested environment.

Raw generation scores use different decoding orders for native and generated sequences and different distributions for the checkpoints. Do not use their differences as a controlled matched-pair experiment or calibrated model-quality comparison. Report future patch comparisons by source protein rather than treating overlapping patches as independent replicates.

Full run artifacts and automatic downloads belong in the ignored `Runs/` and `Results/` directories. Their existence does not establish a successful experiment; inspect job state and the run summary.

To score a prepared pair dataset and curate its results, use new destinations:

```powershell
python Code/scoring/score_pairs.py --pairs Data/first_pair/pairs.json --mpnn-dir Runs/local-vendor/vendor/ProteinMPNN --output Results/my-paired-run --device cuda --orders 8 --order-seed 137 --models vanilla,soluble
python Analysis/summarize_pairs.py --run-dir Results/my-paired-run --pairs Data/first_pair/pairs.json --output Results/my-paired-summary
```

To decompose an existing scored run into its first-order and joint terms, and to measure its decoding-order spread:

```powershell
python Code/scoring/score_first_order.py --pairs Data/compatibility_pairs/pairs.json --mpnn-dir Runs/local-vendor/vendor/ProteinMPNN --scores Analysis/paired_expanded/pair_scores.csv --output Results/my-first-order-run --device cpu
python Analysis/summarize_order_consistency.py --scores Analysis/paired_expanded/pair_scores.csv --output Results/my-order-consistency
python Analysis/plot_decomposition.py --scores Results/my-first-order-run/first_order_scores.csv --output Results/my-first-order-run/decomposition
```

The first-order control runs one unconditional forward pass per structure and checkpoint, takes no decoding order, and reuses the scored run for the full-chain term, so it adds 14 passes rather than a new scoring campaign. It asserts per pair that outside-patch positions contribute exactly zero to the first-order gap; all 85 pairs returned `0.00e+00`, which also checks patch indices against the model's featurization. The order-consistency script needs no model or GPU.

The pair runner loads models once and scores with explicit shared orders, without generating new sequences. Use the project environment and its hardware-compatible torch installation. [Scoring definitions and audits](../Code/scoring/README.md) explain why these outputs differ from raw generation scores.

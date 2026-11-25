# Can a learned scorer rank composition-matched residue arrangements better than frozen ProteinMPNN?

**Experiment specification · November 25, 2026**

For two experimentally measured arrangements A and B on the same protein backbone, predict which arrangement is more stable. Evaluate on held-out protein families. Train three heads on identical two-site swaps and compare them with the complete frozen ProteinMPNN score. The unit of prediction is one swap. The unit of inference is a family.

## 1. Four scorers, three comparisons

| Scorer                | How one arrangement receives a score                                                                                             | Question answered                                                       |
|-----------------------|----------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------|
| Frozen ProteinMPNN    | Mean full-chain autoregressive log likelihood over valid positions and shared decoding orders. Higher means more compatible.     | How accurately does the existing model rank swaps?                      |
| Whole-chain mean      | Vani’s residue MLP followed by a mean over every valid residue and a shared linear scalar output.                                | Does fitting a stability readout help?                                  |
| Patch-local mean      | Same residue MLP and output layer. Take the mean only over the two swapped sites and their eight closest structural neighbors.   | Does focusing the readout near the swap help?                           |
| Patch-local attention | Same local region and residue MLP. Apply one self-attention block within the region, then the same local mean and scalar output. | Does allowing local residue features to interact again improve ranking? |

Each learned head faces the frozen baseline. Whole-chain mean versus local mean isolates the pooling region. Local attention versus local mean measures the incremental value of the attention block, including its additional parameters. A favorable result alone cannot distinguish extra capacity from a particular physical interaction mechanism.

**Design choices for this run.** Preserve Vani’s full-chain decoder features and whole-chain head architecture. Refit all three heads with the same ranking objective. Vani’s existing mixed-mutation, ΔΔG-regression checkpoint, if available, is a separate reference. The earlier architecture draft used target-last probabilities, geometry-biased attention, and a patch-members-only final average. Here both local heads use the same hidden states and average over exactly the same residues. Explicit pair-distance bias is reserved for a later ablation. eSOL is outside this experiment.

## 2. Freeze the labeled cohort and split

1.  Start from the measured_swap rows in Vani’s prepared stability_pairs.json. The preparation summary reports 29,467 training, 10,486 validation and 11,653 test swaps (51,898 total) before this experiment’s eligibility checks. Load them with Code/scorer/data.py. Retain both measurement IDs, replicate handling, normalized ΔΔG, checked backbone and residue mapping. The checked-in summary records preparation and loader validation only. Fit the whole-chain head in this experiment.

Match each pair to the revised extraction by parent and unordered full sequences. It reports 98,505 contrasts and 68,273 resolved before downstream checks. Align A/B orientation, units and sign, then compare the measured values and source records. Vani’s labels use dG_ML. Carry over a resolved flag only when its uncertainty applies to the same reconciled measurements. Count unmatched, discordant, unresolved and zero-effect pairs separately. Exclude these from the primary experiment pending reconciliation.

2.  Reconstruct both measured full sequences and verify each against the same parent chain and residue mapping. A and B must differ at exactly two positions, exchanging two distinct identities, with identical sequence elsewhere. Check four finite backbone coordinates at each accepted residue. Deduplicate unordered sequence pairs within parent. Keep an exclusion table with reason and count.
3.  Save one manifest row per contrast with contrast ID, parent, family, split, chain, backbone path/hash, A/B sequences, two model indices, original residue identifiers, measured difference, uncertainty, resolved flag, and whether either arrangement restores a wild-type residue. Define d = measured stability(A) − measured stability(B), in kcal/mol, with higher measured stability preferred. Check the source sign against two original assay records. Canonicalize A/B without reading the outcome; reversing orientation negates d.
4.  Audit the existing split on CPU before training. Vani grouped 337 parents into 142 groups using a 40% global sequence-similarity threshold and links between identical measured sequences. Cross-tabulate HSSP family against existing split among eligible parents. Exclude unmapped parents with a recorded reason. Keep Vani’s assignments when every HSSP family stays in one partition and all leakage checks pass. Otherwise join parents linked by HSSP family, Vani group or identical sequence and assign intact components approximately 70/15/15 by parent count, seed 7. Use HSSP family as the reporting unit.

CPU audit deliverables. Export the frozen pair manifest, the HSSP-by-split table, label agreement and exclusions by reason, and pre/post counts of families, parents, swaps and class balance per split. Verify no shared parent, family, Vani group or identical sequence crosses partitions. Use one frozen set of eligible pairs and splits for all four scorers and all seeds.

## 3. Implement the heads on one common input

For C contrasts there are 2C sequence assignments. Load backbone coordinates in Å as \[2C,L,4,3\] and residue identities as \[2C,L\], padding to the batch maximum L. Use Vani’s frozen vanilla v_48_020 extractor, evaluation mode and zero coordinate noise. For each assignment and decoding order, concatenate its final decoder state and its 21-symbol identity one-hot. The resulting tensor is \[2C,L,H+21\], where H is read from the checkpoint. Every scorer sees the full sequence and backbone during feature extraction.

Define the local region from the shared backbone before reading either label. For each non-patch residue compute its minimum Cα distance to the two swapped sites. Select the nearest eight valid same-chain residues, breaking distance ties by model sequence index. Include both swapped positions exactly once. A shorter chain uses all available neighbors. The region mask is identical for A and B. Save the selected indices.

Use Vani’s two-layer residue MLP (64 hidden channels, ReLU after each layer) to obtain \[2C,L,64\]. Use a bias-free scalar projection after the relevant masked mean. For the attention head, gather the local tokens \[2C,T,64\], T ≤ 10. Apply four 16-channel heads with scaled dot-product self-attention, a residual connection and layer normalization. Mask padded keys and exclude padded queries from the mean. For this first comparison use one attention layer, no dropout, no extra feed-forward block and no positional embedding. Sequence and structural context already enter through the frozen decoder. Whole-chain and local mean must share identical architecture apart from the mask; train their weights independently.

Use FrozenDecoder, StabilityHead and pair_predictions in Code/scorer/predictor.py as reference implementations for feature extraction, whole-chain pooling and order-averaged pair differences. For A = variant and B = reference, d equals the negative of Vani’s normalized positive-destabilizing target. The new higher-is-more-stable head returns q directly in that orientation.

Define the learned contrast as q = score(A) − score(B), with q \> 0 predicting A is more stable. Scores are dimensionless ranking outputs. If reusing Vani’s pair function, which returns variant minus reference, pass A as variant and B as reference or reverse its sign exactly once. For the frozen model use log likelihood(A) − log likelihood(B); negate existing negative-log-likelihood differences when needed. Verify all four signs with a hand-worked example.

## 4. Train, select and evaluate

- Optimize mean paired hinge loss: max(0, m − sign(d) × q), with m = clip(\|d\| / (1 kcal/mol), 0.1, 2). Train on resolved training swaps. Uniformly sample training parents, then swaps within parent. Define an epoch as the number of resolved training contrasts sampled with replacement. Share sampled pair IDs and orders across heads for a given seed.
- Start with AdamW, learning rate 0.001, weight decay 0.0001, batch size 64 contrasts, maximum 100 epochs, and seeds 7, 17 and 27. Use the same eight deterministic decoding orders per parent from fixed_orders (order seed 137, independent of training seeds) for all models and seeds. Use one shared A/B order per update. At evaluation average q over all eight orders before ranking. Cache frozen features by full sequence, backbone, order, checkpoint and extractor version. Benchmark time and cache size on a training-only subset before scaling.
- Select each seed’s checkpoint using validation family-macro swap accuracy, with lower validation ranking loss as the tie-breaker. Early-stop after ten epochs without improvement. Select the primary learned architecture using its mean validation accuracy over seeds; a tie favors whole-chain mean, then local mean, then attention. Freeze all choices before opening test predictions. Report every prespecified head and seed.
- **Primary metric:** for each resolved held-out swap, award 1 if q and d have the same sign, 0 if opposite, 0.5 for a predicted tie (\|q\| ≤ 1e−8). Average first within parent, then within family, then equally across families. Primary effect is the validation-selected learned scorer’s test accuracy minus frozen accuracy, in percentage points. Average learned-model accuracies across seeds; seeds are repeated fits.
- **Uncertainty:** form a paired difference per test family. Resample whole families with replacement 10,000 times, retaining all models and seeds together; report the percentile 95% interval, family count and family wins/ties/losses. An interval above zero supports improvement on this endpoint; an interval spanning zero leaves the difference unresolved. Report local mean minus whole-chain mean and attention minus local mean as prespecified secondary contrasts.
- **Continuity with the midterm:** also compute within-parent Spearman between q and d, then the median across eligible held-out parents. Use the benchmark’s minimum of 20 contrasts and flag constant-vector exclusions for every model. Higher is better after sign alignment. Report how well scores rank effect magnitudes separately from how often they choose the correct arrangement. Report the genuine double-mutant subset separately from swaps that restore one wild-type residue. ThermoMPNN-D can be shown as an external reference on identical eligible contrasts outside its training and validation domains.

Extend the existing per-pair prediction export and reuse the tie-aware Spearman calculation in the evaluator. Apply the 20-pair eligibility rule consistently across scorers and report excluded parents. Keep MAE and RMSE for optional calibrated regression outputs only. Save validation checkpoints during fitting and defer test evaluation until the architecture selection across seeds is final.

## 5. Verification and deliverables

Before fitting, verify that A/B reversal negates q, identical assignments tie, composition counts tie, local membership is deterministic, padding and local token reordering preserve outputs, and frozen weights receive no gradients. With copied weights and a mask covering the whole chain, the local and whole-chain mean outputs must agree. Run a shuffled-training-label control on validation, shuffling signs within parents while keeping inputs and split fixed. Investigate performance materially above the corresponding chance control before interpreting learned results.

Retain the existing scorer tests for pair reversal, identical inputs, masked pooling, order averaging, split leakage, frozen gradients and weights, and checkpoint save/restore. Extend them to both local heads and cached features. Save the dataset, checkpoint and source hashes, decoding orders, split assignments, run configuration, training history and pair-level predictions. Preserve the regression mode and its units in saved artifacts. Label ranking outputs dimensionless.

| Output                           | Required content                                                                                                                                                            |
|----------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Frozen manifest and audit        | Split assignments, source hashes, sign checks, exclusions, overlap and actual counts                                                                                        |
| Reproducible run record          | Configuration, model parameter counts, seeds, shared orders, selection history, wall time and checkpoint IDs                                                                |
| Prediction table                 | One row per contrast, model and seed with parent, family, region, d, q, correctness and subgroup                                                                            |
| Primary figure and results table | Accuracy for all four scorers, paired differences and intervals, per-family results, Spearman and subgroup counts                                                           |
| Interpretation                   | State separately whether supervision helped, whether locality helped and whether attention added value. If intervals are wide, report the observed effects and uncertainty. |

The original 85 larger-patch permutations may be scored after the measured two-site experiment as an exploratory consistency check. They lack experimental stability labels and contribute no test-accuracy observations.

## 6. Coding and GPU time

Budget **24–40 hours of hands-on work** for one developer familiar with the repository (about 3–5 eight-hour days), plus **6–12 GPU-hours on one NVIDIA L40S** for a first complete experiment. These are planning estimates. They assume access to the original assay files, verified backbones, pretrained weights and fast local scratch storage. GPU jobs can run unattended while analysis and reporting code are completed. Queue wait is additional.

### Coding and verification

| Task                                      | Completion check                                                                                                         | Hands-on hours |
|-------------------------------------------|--------------------------------------------------------------------------------------------------------------------------|----------------|
| Reconcile data and freeze splits          | Reproduce cohort counts, verify sequence mapping and sign, export one manifest, and audit family overlap.                | 5–8            |
| Implement batched extraction and readouts | Cache full-chain features, implement both local heads and match the whole-chain outputs when the pooling masks agree.    | 7–12           |
| Adapt training and evaluation             | Implement the ranking objective, grouped sampling, checkpoint selection, paired metrics, bootstrap and subgroup outputs. | 6–10           |
| Integrate, test and review                | Run invariance and shuffled-label checks, profile throughput, verify saved predictions and produce the primary figure.   | 6–10           |
| **Total**                                 | One developer, including integration and review.                                                                         | **24–40**      |

### GPU allocation

| Stage                               | Basis for the estimate                                                                                                                                                                                                                  | L40S GPU-hours |
|-------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------|
| Extract and write reusable features | At most 51,898 swaps × 2 arrangements × 8 orders = 830,368 sequence–order evaluations before exclusions and cache reuse. The earlier throughput implies about 13 minutes for inference alone. Allow additional time for feature writes. | 0.5–1.5        |
| Train and validate all heads        | 3 heads × 3 seeds = 9 fits. At an assumed 30 epochs per fit and 1–2 minutes per epoch including validation, this is 270–540 minutes.                                                                                                    | 4.5–9          |
| Final evaluation and controls       | Score held-out swaps and run a scoped shuffled-label check. Bootstrap and figure generation can run on CPU.                                                                                                                             | 0.5–1.5        |
| **Planning total**                  | Stage totals span 5.5–12 hours, rounded for scheduling.                                                                                                                                                                                 | **6–12**       |

**Measured reference.** The earlier frozen scoring job processed 1,609,152 sequence–order evaluations in about 25 minutes per checkpoint on an L40S. It wrote scores. The proposed job also writes decoder features, so storage throughput matters. The 1–2 minute epoch time and 30-epoch stopping point are assumptions for budgeting. [Recorded L40S benchmark](https://github.com/ashwathr1008/patch-aware-proteinmpnn/commit/cc8cd51a983d14f4edf9e24fb193dc6c2971a53e)

**First-hour pilot.** Time batched extraction including writes on representative training families. Record sequences per second, bytes per cached sequence–order, projected disk space, and peak GPU memory. For each head, time training batches and eight-order validation on development data, then scale to the actual partition sizes. Count reusable pilot outputs toward the production allocation; add up to one GPU-hour if the pilot is discarded.

**Revised budget.** Multiply each head’s measured minutes per training-plus-validation epoch by its expected epoch count and three seeds, sum across heads, and divide by 60. Add feature extraction and final evaluation once. At 5 minutes per epoch and 30 epochs, training alone takes **22.5 GPU-hours**. Recalculate after the first completed seed. The 6–12 hour estimate assumes batched, cached features throughout; Vani’s original per-pair regression loop needs separate profiling if used directly.

## Repository references

- [Vani’s predictor and whole-chain readout](https://github.com/ashwathr1008/patch-aware-proteinmpnn/blob/vani/scorer/Code/scorer/predictor.py)
- [Existing regression trainer](https://github.com/ashwathr1008/patch-aware-proteinmpnn/blob/vani/scorer/Code/scorer/train.py)
- [Measured swap extraction](https://github.com/ashwathr1008/patch-aware-proteinmpnn/blob/kirti/esol-dataset-plan/Analysis/extract_swap_contrasts.py)
- [Homology grouping rationale](https://github.com/ashwathr1008/patch-aware-proteinmpnn/blob/kirti/esol-dataset-plan/Planning/Homology_Split_Baseline.md)

Vani’s preparation summary, split definitions and tests are documented on vani/scorer in Analysis/scorer_training_data/summary.json, Code/scorer/README.md, Code/scorer/test_predictor.py and Code/scorer/test_prepare_training.py.

- [Earlier patch architecture](https://github.com/ashwathr1008/patch-aware-proteinmpnn/blob/main/Planning/Patch_Scorer_Architecture.md)

# Can a learned scorer rank composition-matched residue arrangements better than frozen ProteinMPNN?

**Experiment specification · November 25, 2026**

For two experimentally measured arrangements A and B on the same protein backbone, predict which arrangement is more stable. Evaluate on held-out protein families. Train three heads on identical two-site swaps and compare them with the complete frozen ProteinMPNN score. The unit of prediction is one swap. The unit of inference is a family.

## 1. Four scorers, three comparisons

| Scorer | How one arrangement receives a score | Question answered |
|---|---|---|
| Frozen ProteinMPNN | Mean full-chain autoregressive log likelihood over valid positions and shared decoding orders. Higher means more compatible. | How accurately does the existing model rank swaps? |
| Whole-chain mean | Vani’s residue MLP followed by a mean over every valid residue and a shared linear scalar output. | Does fitting a stability readout help? |
| Patch-local mean | Same residue MLP and output layer. Take the mean only over the two swapped sites and their eight closest structural neighbors. | Does focusing the readout near the swap help? |
| Patch-local attention | Same local region and residue MLP. Apply one self-attention block within the region, then the same local mean and scalar output. | Does allowing local residue features to interact again improve ranking? |

Each learned head faces the frozen baseline. Whole-chain mean versus local mean isolates the pooling region. Local attention versus local mean measures the incremental value of the attention block, including its additional parameters. A favorable result alone cannot distinguish extra capacity from a particular physical interaction mechanism.

**Design choices for this run.** Preserve Vani’s full-chain decoder features and whole-chain head architecture. Refit all three heads with the same ranking objective. Vani’s existing mixed-mutation, ΔΔG-regression checkpoint, if available, is a separate reference. The earlier architecture draft used target-last probabilities, geometry-biased attention, and a patch-members-only final average. Here both local heads use the same hidden states and average over exactly the same residues. Explicit pair-distance bias is reserved for a later ablation. eSOL is outside this experiment.

## 2. Freeze the labeled cohort and split

1. Use the revised extraction on `kirti/esol-dataset-plan` as the cohort of record. Its reported 98,505 measured contrasts include 68,273 marked resolved before downstream eligibility checks. Apply the extractor’s replicate reconciliation and existing `resolved` flag. Record the revision and input hashes. Treat its uncertainty cutoff as a screening rule. Vani’s 51,898 measured swaps were prepared under different filters and splits. Reuse her loader and backbone checks where applicable, and report any differences in coverage.
2. Reconstruct both measured full sequences and verify each against the same parent chain and residue mapping. A and B must differ at exactly two positions, exchanging two distinct identities, with identical sequence elsewhere. Check four finite backbone coordinates at each accepted residue. Deduplicate unordered sequence pairs within parent. Keep an exclusion table with reason and count.
3. Save one manifest row per contrast with contrast ID, parent, family, split, chain, backbone path/hash, A/B sequences, two model indices, original residue identifiers, measured difference, uncertainty, resolved flag, and whether either arrangement restores a wild-type residue. Define d = measured stability(A) − measured stability(B), in kcal/mol, with higher measured stability preferred. Check the source sign against two original assay records. Canonicalize A/B without reading the outcome; reversing orientation negates d.
4. Use the project’s declared HSSP family assignments for the benchmark parents. The earlier 208-domain cohort produced 84 clusters. Assign intact clusters to approximately 70/15/15 train/validation/test by parent count with fixed seed 7; save the manifest before fitting. Audit for shared parents and identical full sequences across partitions. If identical sequences connect clusters, merge those groups before splitting and record the change. Exclude unmapped parents with a recorded reason. Report actual families, parents, resolved swaps and class balance per split. Keep this same split for every model and training seed.

## 3. Implement the heads on one common input

For C contrasts there are 2C sequence assignments. Load backbone coordinates in Å as [2C,L,4,3] and residue identities as [2C,L], padding to the batch maximum L. Use Vani’s frozen vanilla v_48_020 extractor, evaluation mode and zero coordinate noise. For each assignment and decoding order, concatenate its final decoder state and its 21-symbol identity one-hot. The resulting tensor is [2C,L,H+21], where H is read from the checkpoint. Every scorer sees the full sequence and backbone during feature extraction.

Define the local region from the shared backbone before reading either label. For each non-patch residue compute its minimum Cα distance to the two swapped sites. Select the nearest eight valid same-chain residues, breaking distance ties by model sequence index. Include both swapped positions exactly once. A shorter chain uses all available neighbors. The region mask is identical for A and B. Save the selected indices.

Use Vani’s two-layer residue MLP (64 hidden channels, ReLU after each layer) to obtain [2C,L,64]. Use a bias-free scalar projection after the relevant masked mean. For the attention head, gather the local tokens [2C,T,64], T ≤ 10. Apply four 16-channel heads with scaled dot-product self-attention, a residual connection and layer normalization. Mask padded keys and exclude padded queries from the mean. For this first comparison use one attention layer, no dropout, no extra feed-forward block and no positional embedding. Sequence and structural context already enter through the frozen decoder. Whole-chain and local mean must share identical architecture apart from the mask; train their weights independently.

Define the learned contrast as q = score(A) − score(B), with q > 0 predicting A is more stable. Scores are dimensionless ranking outputs. If reusing Vani’s pair function, which returns variant minus reference, pass A as variant and B as reference or reverse its sign exactly once. For the frozen model use log likelihood(A) − log likelihood(B); negate existing negative-log-likelihood differences when needed. Verify all four signs with a hand-worked example.

## 4. Train, select and evaluate

- Optimize mean paired hinge loss: max(0, m − sign(d) × q), with m = clip(|d| / (1 kcal/mol), 0.1, 2). Train on resolved training swaps. Uniformly sample training parents, then swaps within parent. Define an epoch as the number of resolved training contrasts sampled with replacement. Share sampled pair IDs and orders across heads for a given seed.
- Start with AdamW, learning rate 0.001, weight decay 0.0001, batch size 64 contrasts, maximum 100 epochs, and seeds 7, 17 and 27. Use the same eight deterministic decoding orders per parent for all models and seeds. Use one shared A/B order per update. At evaluation average q over all eight orders before ranking. Cache frozen features by full sequence, backbone, order, checkpoint and extractor version. Benchmark time and cache size on a training-only subset before scaling.
- Select each seed’s checkpoint using validation family-macro swap accuracy, with lower validation ranking loss as the tie-breaker. Early-stop after ten epochs without improvement. Select the primary learned architecture using its mean validation accuracy over seeds; a tie favors whole-chain mean, then local mean, then attention. Freeze all choices before opening test predictions. Report every prespecified head and seed.
- **Primary metric:** for each resolved held-out swap, award 1 if q and d have the same sign, 0 if opposite, 0.5 for a predicted tie (|q| ≤ 1e−8). Average first within parent, then within family, then equally across families. Primary effect is the validation-selected learned scorer’s test accuracy minus frozen accuracy, in percentage points. Average learned-model accuracies across seeds; seeds are repeated fits.
- **Uncertainty:** form a paired difference per test family. Resample whole families with replacement 10,000 times, retaining all models and seeds together; report the percentile 95% interval, family count and family wins/ties/losses. An interval above zero supports improvement on this endpoint; an interval spanning zero leaves the difference unresolved. Report local mean minus whole-chain mean and attention minus local mean as prespecified secondary contrasts.
- **Continuity with the midterm:** also compute within-parent Spearman between q and d, then the median across eligible held-out parents. Use the benchmark’s minimum of 20 contrasts and flag constant-vector exclusions for every model. Higher is better after sign alignment. Report how well scores rank effect magnitudes separately from how often they choose the correct arrangement. Report the genuine double-mutant subset separately from swaps that restore one wild-type residue. ThermoMPNN-D can be shown as an external reference on identical eligible contrasts outside its training and validation domains.

## 5. Verification and deliverables

Before fitting, verify that A/B reversal negates q, identical assignments tie, composition counts tie, local membership is deterministic, padding and local token reordering preserve outputs, and frozen weights receive no gradients. With copied weights and a mask covering the whole chain, the local and whole-chain mean outputs must agree. Run a shuffled-training-label control on validation, shuffling signs within parents while keeping inputs and split fixed. Investigate performance materially above the corresponding chance control before interpreting learned results.

Use the scorer branch’s predictor, data loader and training loop as starting points. Add region masks and a choice of readout to the predictor, a swap-ranking training mode, family-aware evaluation and the tests above. Preserve the existing regression mode. Record branch and commit hashes for code and benchmark data. Local genotype/assay files and the pinned ProteinMPNN weights are prerequisites.

| Output | Required content |
|---|---|
| Frozen manifest and audit | Split assignments, source hashes, sign checks, exclusions, overlap and actual counts |
| Reproducible run record | Configuration, model parameter counts, seeds, shared orders, selection history, wall time and checkpoint IDs |
| Prediction table | One row per contrast, model and seed with parent, family, region, d, q, correctness and subgroup |
| Primary figure and results table | Accuracy for all four scorers, paired differences and intervals, per-family results, Spearman and subgroup counts |
| Interpretation | State separately whether supervision helped, whether locality helped and whether attention added value. If intervals are wide, report the observed effects and uncertainty. |

The original 85 larger-patch permutations may be scored after the measured two-site experiment as an exploratory consistency check. They lack experimental stability labels and contribute no test-accuracy observations.

## Repository references

- [Vani’s predictor and whole-chain readout](https://github.com/ashwathr1008/patch-aware-proteinmpnn/blob/vani/scorer/Code/scorer/predictor.py)
- [Existing regression trainer](https://github.com/ashwathr1008/patch-aware-proteinmpnn/blob/vani/scorer/Code/scorer/train.py)
- [Measured swap extraction](https://github.com/ashwathr1008/patch-aware-proteinmpnn/blob/kirti/esol-dataset-plan/Analysis/extract_swap_contrasts.py)
- [Homology grouping rationale](https://github.com/ashwathr1008/patch-aware-proteinmpnn/blob/kirti/esol-dataset-plan/Planning/Homology_Split_Baseline.md)
- [Earlier patch architecture](https://github.com/ashwathr1008/patch-aware-proteinmpnn/blob/main/Planning/Patch_Scorer_Architecture.md)

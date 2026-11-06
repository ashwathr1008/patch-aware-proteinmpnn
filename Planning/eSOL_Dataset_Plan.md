# eSOL-based dataset plan

Draft of November 6, 2026, for the final project. Supersedes nothing yet; the 20–100 residue
range in [the scope statement](../Documentation/Midreport/scope_statement.md) must be amended
if this plan is adopted, and the reason for amending it is recorded below rather than left
implicit.

This plan exists because the seven pilot structures carry **no measured outcome of any kind**.
They were frozen for tractability before the eSOL audit existed, and their overlap with eSOL is
zero. Nothing about them was wrong; the sequencing simply meant nobody checked label
availability when choosing them.

## What this dataset can and cannot do

**It cannot label the existing experiment.** eSOL measured real native *E. coli* proteins. The
native-versus-permuted comparison scores a permuted sequence that has never been synthesised,
expressed or measured by anyone, so no label set can exist for it. That comparison stays a
compatibility experiment regardless of which proteins it runs on.

**It enables a different experiment**: across many native proteins, does a patch-level
arrangement score predict measured solubility, and does it add anything beyond composition and
frozen ProteinMPNN? That is a genuine prediction task with real labels and a real baseline, and
it is what the [scope note](Midreport_vs_Final_Scope.md) says the final project needs.

Keeping these two apart matters. Reporting them as one study would imply the permuted pairs
carry measured labels, which is exactly the claim the instructor's feedback warns against.

## Cohort: 20–150 residues, 185 proteins

The scope statement says 20–100 residues. **That range cannot support this experiment**, and the
reason is measurable rather than a matter of taste.

| Cohort | Proteins | Label SD | Observed r needed | A true r = 0.5 appears as |
| --- | ---: | ---: | ---: | ---: |
| 20–100 residues | 54 | 18.5 | 0.373 | **0.339 — undetectable** |
| 20–150 residues | 185 | 24.1 | 0.205 | 0.425 |
| 20–300 residues | 632 | 29.0 | 0.111 | 0.492 |

Detection thresholds are for 80% power at α = 0.05 two-sided; the attenuation column applies the
standard range-restriction correction against the full usable set's SD of 29.6.

At 20–100 residues, 69% of proteins are above 80% soluble and none is below 23%. Two biases
stack there: small proteins are intrinsically more soluble, and solved structures skew soluble
(median 56% against 33% for proteins without one). The result is that **even a strong true
relationship would be compressed below the detection threshold**. This is not a sample-size
problem that more patches on the same 54 proteins would fix; patches are not independent
replicates of the protein-level label.

Extending to 150 residues recovers usable spread — 44 proteins below 60% solubility rather than
8 — while remaining single-domain territory at a median length of 113.

**The honest limit of this cohort.** n = 185 powers a test of whether patch scores predict
solubility at all. It may not power the test that matters, which is whether they add anything
*after* composition and frozen ProteinMPNN. The decomposition found the arrangement term is
roughly 11% of the model's own signal; if a similar proportion carries into prediction, the
incremental correlation could sit near 0.15–0.20, under this cohort's 0.206 threshold. Plan for
this rather than discovering it at the end:

- Pre-declare the raw-correlation test as primary, since it is powered.
- Declare the incremental test secondary and exploratory at n = 185.
- If the incremental effect is the claim the project wants to make, the cohort has to reach
  roughly 600 proteins (20–300 residues), and the small-systems framing goes with it.

That trade should be made deliberately and written into the report, not left for a reviewer.

## Build steps

### 1. Filter structures properly

The 1,328 usable count is an **upper bound**. UniProt cross-references were counted per protein,
never checked for quality. Each candidate needs its PDB entries filtered on:

- coverage of the full mature chain, not a fragment or a single domain of a longer construct
- resolution, with a fixed cutoff chosen before looking at outcomes
- absence of engineered mutations and non-native tags
- a complete canonical N, CA, C, O backbone, which the existing `stage_backbone` already enforces

Where several entries pass, pick one by a fixed rule — best resolution, then lowest PDB ID — and
record the rule. Do not pick per protein by inspection. Expect this filter to remove a
substantial share; re-run the power table on whatever survives before committing.

### 2. Cluster before splitting

Cluster the surviving sequences at a fixed identity threshold and assign **whole clusters** to
development and held-out groups. Two cautions carried from the audit:

- Ordered-locus proximity and gene-name similarity do not imply homology. Neighbouring
  b-numbers are frequently an operon, not a family.
- All patches and variants descended from one protein stay in that protein's group.

Fix the split before any scoring, and never re-draw it after seeing results.

### 3. Select patches with the existing tooling

`Code/patches` is organism-agnostic and needs no change. Keep the current definition — surface
neighbourhoods on a fixed backbone — and keep the existing seven invariants. Record the
exposure-proxy caveat as before: the neighbour count is not solvent-accessible surface area.

Patches per protein should be fixed in advance by a rule, not tuned. Overlapping patches from
one protein are not independent evidence and must not be pooled as if they were.

### 4. Aggregate patch scores to protein level

This is the step most likely to go wrong, because the labels are whole-protein and the scores
are per-patch. The aggregation function is a modelling choice that can be tuned to fit, so:

- Declare the aggregation rule **before** looking at any held-out label. Mean, max and
  top-k mean are all defensible; pick one and say why.
- If more than one is tried, report all of them, and treat the extras as exploratory.
- Never select the aggregation on test labels. The plan's own wording: "Define protein-level
  aggregation of patch scores without tuning on test labels."

### 5. Baselines, declared in advance

A patch score has to beat these, not just correlate with the label:

- composition alone (amino-acid fractions)
- the SAP surrogate, which is the closed-form external baseline
- frozen ProteinMPNN likelihood, vanilla and soluble
- chain length, which correlates with solubility on its own and is the most likely confound in
  a cohort selected by length

Length deserves particular care here: the cohort is defined by a length window, and solubility
varies with length within it. Report the patch score's contribution with length partialled out.

## Analysis, pre-registered

**Primary.** Spearman correlation between the protein-level patch score and measured solubility,
on the held-out clusters only, with a pre-declared threshold.

**Secondary.** Incremental contribution over the baselines above. Underpowered at n = 185;
report the estimate with its interval and label it exploratory.

**Stratified reporting.** The usable set under-represents the aggregation-prone range. Report by
solubility band rather than pooling, so the soluble majority does not carry the result.

**Negative results are reportable.** If the patch score does not predict solubility, that is an
answer about this proxy on this organism at this size, and it belongs in the report.

## What this plan does not claim

- It does not make the permuted-pair experiment a measured-outcome study.
- eSOL is chaperone-free cell-free *E. coli* solubility: one organism, one condition, not
  therapeutic formulation behaviour and not an aggregation rate.
- Transfer to designed miniproteins or binders is unestablished.
- These proteins are almost certainly inside ProteinMPNN's pretraining set, as the pilot
  structures are. A cluster-held-out split does not establish otherwise.

## Decided: the cohort is built

The structure filter and clustering have run, and the cohort is fixed at
[Analysis/esol_cohort/cohort.json](../Analysis/esol_cohort/cohort.json). Each correctness step
cost statistical power, and all three numbers are recorded because the last one is the only
honest one:

| Stage | Unit | n | Smallest detectable true r |
| --- | --- | ---: | ---: |
| Unfiltered candidates | proteins | 185 | ~0.28 |
| After structure filter | proteins | 138 | 0.309 |
| **After homology clustering** | **clusters** | **103** | **0.359** |

Structure filtering kept 138 of 185, dropping 20 for resolution, 17 for construct length and 10
for engineered mutations. It also tightened the label spread, from SD 24.1 to 22.2, because it
removes proteins non-randomly.

Clustering then found 35 related pairs, placing **60 of the 138 proteins in a multi-protein
cluster**. Nearly half the cohort has a relative in it. Treating proteins as independent would
inflate significance, so the cluster is the unit of analysis and n is 103, not 138.

The split is fixed: 97 proteins in development, 41 in holdout, assigned by whole cluster and
verified so that no cluster is divided. **Do not redraw it.**

### What this cohort can and cannot establish

It can detect a true correlation of about **0.36 or stronger** between a protein-level patch
score and measured solubility. That is a moderate-to-strong relationship.

It probably cannot establish the **incremental** claim, that a patch score adds predictive
value beyond composition and frozen ProteinMPNN, because an increment would have to be that
large on its own after the baselines have taken their share. The raw-correlation test stays
primary and powered; the incremental test stays secondary and exploratory, exactly as this plan
declared before the numbers were known.

The resolution cutoff was not revisited after seeing the power. It was declared before
inspection so that it could not be tuned toward a more comfortable answer.

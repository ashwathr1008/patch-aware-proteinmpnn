# Patch scorer: specification and remaining work

Rewritten November 6, 2026, after the measured baseline was established. This replaces the
earlier draft, which assumed the scorer would be trained self-supervised on generated
permutations because no measured arrangement labels existed. They do exist, which changes the
training signal, the success criterion and the risk profile.

## What is now known, and what it changes

Four results from this week set the terms.

**There are measured arrangement labels.** The mega-scale stability dataset measured all 400
identity combinations at 578 contacting site pairs across 208 domains. Among them are
**100,572 composition-matched swap contrasts**: the same two residues in opposite arrangements
on the same backbone, both measured. **69,633 of those resolve above twice their own
measurement error.** The earlier claim in this project's planning, that such labels cannot
exist, was correct only for patch-wide permutations.

**The pairs are genuinely contacts**, verified rather than assumed: median 5.6 Å apart against
14.0 Å for all pairs in the same domains, 5.2× enriched, and a median sequence separation of 12
residues, so they are tertiary contacts rather than helix neighbours.

**The frozen baseline is 0.48, not 0.44.** Both checkpoints were scored on an L40S. The full
autoregressive model reaches a per-domain median Spearman of **−0.497** (vanilla) against
measured stability; its first-order term alone reaches **−0.448**. The full model is stronger in
135 of 199 domains, sign test p = 5.3 × 10⁻⁷. Any scorer must beat the full model, not the
first-order term.

**There is room.** The measurement-noise ceiling is **0.98**, so roughly **0.50 of correlation
remains unexplained**. The target is not a sliver.

Two consequences for the design. The head trained on these contrasts has **real labels**, so it
is a stability head and may be called one; the naming problem in the original proposal applied
to permutation-derived labels and does not apply here. And the fine-tuning gate, which the plan
held shut until a reward was validated, is now open in principle, because measured ΔΔG is a
validated target.

## Architecture

Two measurements constrain this, and the first rules out the obvious design.

### The encoder cannot be the feature source

**ProteinMPNN's encoder takes no sequence.** In `unconditional_probs` and in `forward` alike,
the node state is initialised as `h_V = torch.zeros(...)` and the edge features come from
`self.features(X, mask, residue_idx, chain_encoding_all)` — backbone geometry only. Sequence
enters at the decoder.

So encoder embeddings are **identical for every assignment on one backbone**. A head built on
them would receive the same input for the original and the permuted arrangement and could not
distinguish them at all, let alone notice a third residue. An earlier draft of this spec
specified exactly that; it would have failed silently at the first training step, with a model
that learned a constant.

**Features therefore come from the decoder, conditioned on sequence.** `conditional_probs`
(`protein_mpnn_utils.py:1292`) gives p(aa_i | backbone, rest of the sequence), so a
representation read at position i already depends on everything else in the chain. Mask the
patch's own positions, condition on the remainder, and read the decoder states there.

This keeps the argument that motivated a frozen design: the scorer still sees exactly what
ProteinMPNN sees, so a negative result means the objective failed rather than that the
representation was starved. Only the method called changes.

The cost is that features depend on the sequence, so they cannot be cached once per structure
the way the first-order term could. One pass per variant, the same cost as the scoring already
done.

### How much many-body capacity is needed

Measured, rather than assumed. Across 74 networks from the 28 triples whose three pairs were
all scored, holding a pair's residues in both arrangements while setting a third site in the
same network to each of the twenty identities:

| | |
| --- | ---: |
| Median swap effect | 4.730 nats |
| Median spread across the third residue | 0.458 nats |
| **Median relative spread** | **10.0%** |
| Networks under 20% spread | 58 of 74 |
| Networks where the preference reverses | 5 of 74 |

**About ninety per cent of a swap effect is independent of the third residue.** Pairwise labels
therefore carry to patch scale, and the 100,572 measured contrasts are usable for training a
patch-level model. This is the result that keeps the whole strategy viable.

The remaining tenth is real. The five networks that reverse are mostly weak-effect pairs, mean
magnitude between 0.1 and 1.2 nats, where context decides a question the pair alone leaves
open. Those are also the cases a scorer would be most useful on, since they are where the model
is currently least certain. Relative spread is unstable for the seven networks whose swap
effect is under one nat, so the median is the figure to quote rather than the maximum.

Sequence-conditioned features cover this tenth without extra machinery, which is the second
reason to take them over encoder features. A pairwise-only model would capture most of the
effect and systematically miss the marginal cases.

### The design

- **Node features**: decoder states from frozen ProteinMPNN under `conditional_probs`, with the
  patch's positions masked and the rest of the sequence conditioned on. Same k = 48 neighbour
  graph the checkpoint defines. The whole chain is encoded; localization happens at the
  readout, never by cropping the input.
- **Readout**: attention pooling over the patch's members **and their nearest neighbours**,
  each carrying its identity embedding alongside its geometry, so three-body terms can be
  represented explicitly rather than having to survive pooling. Mean pooling over members
  alone is the ablation, and the difference between them measures what the many-body capacity
  buys.
- **Head**: two or three dense layers to one scalar. Deliberately small; a large head over
  frozen features will memorize domains rather than learn arrangement.
- **Output**: one scalar per patch per assignment, where a higher value means more stable.

**Excluded from the features**, because both would let the model win the wrong way: native
side-chain coordinates after identities are reassigned, and any composition summary of the
patch. Composition is matched exactly within every training contrast, so a composition feature
carries no gradient there, but it would carry one across domains.

Fine-tuning is now permissible, since measured ΔΔG is a validated target, but it is **not** the
starting point. Start frozen: it trains in minutes, it isolates the objective, and a failure is
informative. Only once a frozen head is understood does unfreezing answer a question rather
than muddy one.

## Training signal

**Supervised, on measured ΔΔG.** For each swap contrast, the model must order the two
arrangements as the measurement orders them.

Loss is **ranking, not regression onto ΔΔG values**. Two reasons: absolute stabilities vary by
domain and construct in ways the model has no access to, and the quantity of interest is the
ordering. A margin ranking loss over each contrast, with the margin scaled by the measured
difference, uses the magnitude without demanding the model reproduce it.

**Drop contrasts below the measurement floor.** Of 100,572 contrasts, 30,939 have an effect
smaller than twice their own error. The frozen baseline scores ρ = −0.069 on those against
−0.477 on resolved ones, which is the expected behaviour and confirms they carry no learnable
signal. Train on the **69,633 resolved** contrasts. This is a rule declared before training,
not a filter tuned afterwards.

## Splits

**Cluster the 208 domains with the homology module and assign whole clusters.** The 85-pair
pilot's groups were hand-assigned and turned out over-merged; the eSOL cohort collapsed from
138 proteins to 103 clusters, with 60 proteins having a relative in the set. There is no reason
to expect 208 structural-genomics domains to be free of paralogy.

Fix the split before any training and never redraw it. `Code/homology/cluster.py` and
`verify_split` already do this and refuse a split that divides a cluster.

Evaluate **only on held-out clusters**, and report the per-domain distribution rather than the
pooled correlation. Pooling 70,000 contrasts from 208 domains overstates significance; the
baseline is reported both ways for exactly this reason.

## Controls

Each falsifies a specific way the result could be hollow. None is optional.

| Control | What it rules out | Expected |
| --- | --- | --- |
| Composition-only model | Broken contrasts | Chance, by construction |
| Shuffled labels | A leaking split | Chance |
| Contrasts below measurement error | A pipeline artefact | Near zero; baseline gives −0.069 |
| Frozen ProteinMPNN, full model | The scorer adding nothing | **−0.497 per-domain median** |
| Frozen ProteinMPNN, first-order | Crediting the joint term wrongly | −0.448 |
| SAP surrogate | Re-deriving a closed-form statistic | measured on the same contrasts |

The composition control is the cheapest and should run whenever the dataset changes:
composition is identical within every contrast, so it is a guaranteed null.

## Success criteria, declared in advance

**Primary.** On held-out clusters, per-domain median Spearman against measured ΔΔG exceeds the
frozen full model's **−0.497**, tested paired across domains, with the same sign test used for
the baseline comparison.

**What counts as meaningful.** The joint term is worth about +0.03 to +0.05 over the first-order
term. A scorer that adds less than the model's own joint term adds little. A reasonable target
is **−0.55 or better**, which would roughly double the joint term's contribution while leaving
most of the 0.50 headroom still open.

**Reporting rule.** Improving a score the model was trained on is expected by construction. The
held-out cluster result is the claim; everything else is diagnostic.

## From two residues to a patch

The training contrasts are two-site swaps. The project's premise is patch-level, and the bridge
is a generalization axis rather than a change of premise: **patch size is what varies between
training and evaluation**, not the architecture, which takes a patch of any size.

Three steps, in order of evidence:

1. **Train on k = 2, evaluate on k = 2** held-out clusters. This is the powered experiment.
2. **Evaluate on the existing 85 pilot pairs**, which are 8–10 residue patches with no measured
   label. Agreement with frozen ProteinMPNN there is a consistency check, not validation.
3. **Test additivity directly.** The dataset holds 66 complete triangles, three sites with all
   three pairs measured, across 33 domains, and two complete four-site sets. Whether a
   three-site arrangement effect is predicted by its three pairwise effects is a measurable
   question, and the answer bounds how far pairwise training can carry. `Triple_DMS_list.csv`,
   in the part of the archive not yet retrieved, holds genuine triple mutants.

If additivity holds at k = 3, pairwise training generalizes. If it fails, patch effects are
irreducibly many-body and that is a finding worth reporting rather than a setback.

## Secondary validation: a different property

The stability head answers whether arrangement scoring predicts folding stability. The
project's motivation is developability, which is a different property. The
[eSOL cohort](eSOL_Dataset_Plan.md) provides it: 138 *E. coli* proteins, 103 clusters, with
measured solubility and quality-filtered structures.

That cohort detects a true correlation of about 0.36 or stronger and probably cannot establish
an incremental claim over baselines. It is a secondary, exploratory test, and transfer between
the two properties is itself the question rather than an assumption.

## Remaining work

| Dates | Work |
| --- | --- |
| November 7–10 | Cluster the 208 domains, fix the split, cache frozen encoder features, run the composition-only and shuffled-label controls. Retrieve `Triple_DMS_list.csv`. |
| November 11–17 | Train the frozen-encoder head on the 69,633 resolved contrasts. Compare against all six controls on held-out clusters. Pooled and per-domain, paired sign test. |
| November 18–24 | Patch-size generalization: the 85 pilot pairs, then the additivity test on triangles. Ablate attention versus mean pooling, and the SAP surrogate as an input feature. |
| November 25–December 1 | Secondary validation on the eSOL cohort. Failure analysis. Fine-tune the encoder only if the frozen head has cleared the baseline and the result is understood. |
| December 2–5 | Final report and poster. |

**No cluster dependency.** Bridges-2 enters a migration outage on November 9 with an expected
four to six week duration, past the December 6 deadline. Everything above runs on a laptop: the
features are cached once, the head is small, and the baseline scoring is already done and
retrieved.

## Risks, and what each would mean

- **The head does not beat 0.497.** The most likely single outcome. It would mean the frozen
  representation already carries the arrangement information a patch-level objective can
  extract, which is a real finding and the one the project should be prepared to report.
- **It beats the baseline on pooled contrasts but not per-domain.** An artefact of domain
  imbalance. The per-domain test is primary for this reason.
- **It beats k = 2 but fails to generalize to larger patches.** Then the claim narrows to
  pairwise arrangement, which is still measured and still novel.
- **Additivity fails at k = 3.** Pairwise training cannot reach patch scale, and the honest
  conclusion is that patch-level arrangement needs patch-level labels, which do not exist.

Each of these is reportable. The design's purpose is that none of them can be mistaken for
success.

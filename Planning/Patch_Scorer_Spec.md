# Patch scorer: specification and remaining work

Rewritten November 6, 2026, after the measured baseline was established. This replaces the
earlier draft, which assumed the scorer would be trained self-supervised on generated
permutations because no measured arrangement labels existed. They do exist, which changes the
training signal, the success criterion and the risk profile.

## What is now known, and what it changes

Four results from this week set the terms.

**There are measured arrangement labels.** The mega-scale stability dataset measured all 400
identity combinations at 578 contacting site pairs across 208 domains. Among them are
**98,505 composition-matched swap contrasts**: the same two residues in opposite arrangements
on the same backbone, both measured. **68,273 of those resolve above twice their own
measurement error.** The earlier claim in this project's planning, that such labels cannot
exist, was correct only for patch-wide permutations.

**Repeated measurements are averaged when they agree, and dropped when they do not.** The
source table measures 5,952 variants more than once. The original extraction assigned straight
into a table, so whichever row the reader reached last silently won and the other measurement
vanished. Of those 5,952, **2,930 have replicates agreeing within their own stated error and are
averaged**; the remaining **3,022 disagree and are dropped**, with a median disagreement of
0.214 kcal/mol against a median measurement uncertainty of 0.12. The consistency test is the
negation of the `resolved` rule — two replicates agree exactly when their difference would not
be called resolved — so one threshold governs both decisions. Declared before any scorer was
trained.

This cost 2,067 contrasts and moved the frozen per-domain median by 0.004, from −0.497 to
−0.501. **That the statistic barely moved is not evidence the defect was harmless.** A median
over 199 domains is robust to a 2% change in membership almost by construction; 3,022
arbitrarily-resolved measurements sitting inside a *training* signal is a different matter, and
is why this was fixed before training rather than after.

**The pairs are genuinely contacts**, verified rather than assumed: median 5.6 Å apart against
14.0 Å for all pairs in the same domains, 5.2× enriched, and a median sequence separation of 12
residues, so they are tertiary contacts rather than helix neighbours.

**The frozen baseline is 0.48, not 0.44.** Both checkpoints were scored on an L40S. The full
autoregressive model reaches a per-domain median Spearman of **−0.501** (vanilla) against
measured stability; its first-order term alone reaches **−0.448**. The full model is stronger in
135 of 199 domains, sign test p = 5.3 × 10⁻⁷. Any scorer must beat the full model, not the
first-order term.

**There is room.** The measurement-noise ceiling is **0.98**, so roughly **0.50 of correlation
remains unexplained**. The target is not a sliver.

**Part of the frozen model's 0.501 is native-sequence recognition, not arrangement.** Splitting
the resolved contrasts by whether both arrangements are genuine double mutants, the frozen model
reaches **−0.680** on the 14,481 contrasts where one arrangement restores a wild-type residue
and only **−0.423** on the 53,792 where neither does. ProteinMPNN is trained to recover native
sequence, so it prefers the arrangement containing a wild-type residue strongly, and the
measurement usually agrees because wild type is usually more stable. Composition is matched
inside every contrast, so this is not a composition artefact, but it is a nativeness one.
**The double-mutant subset is the honest measure of arrangement sensitivity**, and the number to
beat there is −0.423 rather than −0.501.

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
therefore carry to patch scale, and the 98,505 measured contrasts are usable for training a
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

**Drop contrasts below the measurement floor.** Of 98,505 contrasts, 30,232 have an effect
smaller than twice their own error. The frozen baseline scores ρ = −0.069 on those against
−0.477 on resolved ones, which is the expected behaviour and confirms they carry no learnable
signal. Train on the **68,273 resolved** contrasts. This is a rule declared before training,
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
| Frozen ProteinMPNN, full model | The scorer adding nothing | **−0.501 per-domain median** |
| Frozen ProteinMPNN, first-order | Crediting the joint term wrongly | −0.448 |
| SAP surrogate | Re-deriving a closed-form statistic | measured on the same contrasts |
| ThermoMPNN-D (published) | A trained head on these features already doing better | **−0.459 vs frozen −0.392** on the 81 domains it did not train on, 45/81, p = 0.37 |

The composition control is the cheapest and should run whenever the dataset changes:
composition is identical within every contrast, so it is a guaranteed null.

## Success criteria, declared in advance

**Primary.** On held-out clusters, per-domain median Spearman against measured ΔΔG, tested
paired across domains with the same sign test used for the baseline comparison.

**The comparator must be recomputed on whatever subset the scorer is evaluated on**, never
quoted across subsets. This rule is not pedantry; two subset effects measured here are each
larger than the signal being chased. Restricting to double mutants moved the frozen baseline by
0.074. Restricting to domains ThermoMPNN did not train on moved it by a further 0.031, and
moved ThermoMPNN-D by 0.083.

The bar for this project is therefore **frozen ProteinMPNN recomputed on our own held-out
clusters, over exactly the contrasts the scorer is scored on**. Nothing else is a valid target.
For orientation only, on resolved double-mutant contrasts:

| | All 198 domains | The 81 ThermoMPNN never trained on |
| --- | ---: | ---: |
| Frozen ProteinMPNN, full model | −0.423 | −0.392 |
| ThermoMPNN-D | −0.501 | −0.459 |
| Measurement-noise ceiling | ~0.98 | ~0.98 |

**What counts as meaningful.** Beating the frozen model on held-out clusters, paired, with the
sign test significant. The headroom to the noise ceiling is roughly 0.6, so this is not a
contest for hundredths. ThermoMPNN-D is the nearest published analogue of the proposed
architecture and it improves on the frozen likelihood by about 0.07 in median on clean domains
without that improvement being consistent enough to reach significance, which is the realistic
expectation to hold for a first frozen head here too.

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
3. **Test additivity directly. Done, and it passed.** `Triple_DMS_list.csv` has been retrieved
   and is committed. Across 74 networks from the 28 triples whose three pairs were all scored,
   the median swap effect was 4.730 nats and the median spread across the third residue was
   0.458 nats — **10.0% relative spread**, with the preference reversing in only 5 of 74. The
   measured result in the source paper is stronger still: substituting any member of a
   three-residue network for alanine removed the coupling between the other two.

**Pairwise training therefore generalizes**, which is the result that keeps this strategy
viable: about ninety per cent of a swap effect is independent of its surroundings, so the
98,505 measured pairwise contrasts are legitimate training data for a patch-level model. The
remaining tenth bounds how far they carry, and the sign-reversing networks are the subject of
route D in [Next steps](#next-steps).

## Secondary validation: a different property

The stability head answers whether arrangement scoring predicts folding stability. The
project's motivation is developability, which is a different property. The
[eSOL cohort](eSOL_Dataset_Plan.md) provides it: 138 *E. coli* proteins, 103 clusters, with
measured solubility and quality-filtered structures.

That cohort detects a true correlation of about 0.36 or stronger and probably cannot establish
an incremental claim over baselines. It is a secondary, exploratory test, and transfer between
the two properties is itself the question rather than an assumption.

## Next steps

Three things, in order. The first is dated by a machine outage rather than by choice. The
second is the experiment this project exists to run. The third is what to do if it does not
work, written down now rather than after seeing the result, so the choice is not made to
flatter whatever happens.

### 1. ThermoMPNN-D across all 208 domains, before November 9

**Why first.** Bridges-2 enters a migration outage on November 9, expected to last four to six
weeks, past the December 6 deadline. This is the only remaining item that wants a GPU, so it
runs now or not at all.

**The environment is lighter than it looks.** Its `environment.yaml` pins `pytorch-cuda=11.7`
and `cudatoolkit=11.3`, but inference needs none of that. On top of the PSC module already
used for the swap scoring (`pytorch/26.05-2.11-py3`), the whole import chain of `v2_ssm.py`
needs five pure-Python packages: `biopython`, `omegaconf`, `pytorch_lightning`, `torchmetrics`
and `wandb`. `numpy`, `pandas`, `scipy` and `tqdm` come from the module. `mmseqs2` is a
training dependency and is not needed. So a venv and one `pip install`, not a conda solve.

**Transfer.** The repository without `.git`, plus `model_weights` (53 MB) and
`vanilla_model_weights` (26 MB), plus the 208 AlphaFold structures. Bridges-2 login nodes have
neither `scp` nor `rsync`, so `tar czf - | ssh ... tar xzf -` as before.

**Two defaults will quietly ruin the run.** `--threshold` keeps only mutations below
−0.5 kcal/mol, and `--distance` keeps only pairs within 5 Å Cα–Cα, which covers **71 of our 578
site pairs**. Reproducing our contrasts needs `--mode epistatic --threshold 100 --distance 12`;
12 Å covers 534 of 578 (92%). The script takes no mutation list — eight arguments, enumeration
only — so the radius is the only control over what gets predicted, and raising it to 12 Å is
roughly 14× the work per domain. 1VII produced 14,440 predictions in 22.75 s on a laptop CPU,
so the full set is well under an hour on an L40S and infeasible locally.

**Validate before trusting it.** This checkout carries a device patch, a config rewrite and two
dependency downgrades. Reproduce one number from their `benchmark.ipynb` first. A baseline
patched into running is only a baseline if it still computes what the paper computed.

**The caveat that decides how to read the result.** ThermoMPNN-D was trained on the Tsuboyama
mega-scale dataset — the same source as our 98,505 contrasts. Its config references a
`mega_splits.pkl` that **is not in the repository**, existing only as a path on the author's
machine, so nothing in the checkout reveals whether our 208 domains were in its training set.
Until that file is obtained, its score is **not a clean baseline**: for any domain it trained
on, the number reflects memorization as much as generalization. Retrieve `mega_splits.pkl` from
the paper's data release and report the comparison split by whether each domain was in its
training set. If it cannot be obtained, the comparison is reported as an upper bound and
labelled as one.

**Join** on `(domain, site_i, site_j, aa_i, aa_j)` and compute the same per-domain Spearman
against measured ΔΔG, so the number is directly comparable to the frozen model's −0.501.

### 2. The first optimization run

Deliberately the smallest thing that could work. A large model over frozen features would
memorize domains, and a negative result from it would be uninterpretable.

| | |
| --- | --- |
| Features | `Code/scorer/features.py`: per patch member, 21 conditional log-probabilities and a 21-way identity one-hot. **84 dimensions at k = 2.** |
| Head | Two dense layers, width 64, one scalar output. Roughly 6k parameters. |
| Loss | Margin ranking per contrast, margin scaled by the measured ΔΔG difference. Not regression onto ΔΔG. |
| Training set | The **68,273 resolved** contrasts. The 30,232 below twice their own error are dropped, a rule declared before training and not tuned after. |
| Splits | Whole clusters from `Code/homology/cluster.py`, fixed before training and never redrawn. |
| Primary test | Per-domain median Spearman on held-out clusters against the frozen model's **−0.501**, paired sign test. |

**Cache the features once.** They depend on sequence, so they cannot be cached per structure
the way the first-order term was. But each contrast needs only its two target positions:
578 pairs × 400 identity combinations × 2 positions ≈ **460k decoder passes**, which batches to
roughly 7k forward passes and well under an hour on a laptop. Cache keyed by domain, position
and sequence hash, then training is minutes and ablations are free.

**Blocking prerequisite:** `parasail` is missing from the local environment, so
`Code/homology/cluster.py` cannot run and the split cannot be fixed. Reinstall it first — and
check it does not drag `numpy` back above 2, which broke every scoring script last time.

### 3. If the scorer does not clear −0.501

The most likely single outcome, and the decomposition says why: 89% of the pilot effect was
first-order burial and only 11% joint structure. There may simply not be much left for a
patch-level objective to extract.

**ThermoMPNN-D has been run, and its split membership resolved. The result is inconclusive,
and that is itself informative.**

Across all 198 domains it reached −0.501 against the frozen model's −0.423, stronger in 137 of
198, p = 6.9 × 10⁻⁸. That comparison is contaminated: `mega_splits.pkl`, retrieved from the
original ThermoMPNN repository, labels splits at the domain level and shows **111 of our 208
domains in its training split and 13 more in validation**. Splitting on membership:

| Domain group | n | ThermoMPNN-D | Frozen | Stronger | p |
| --- | ---: | ---: | ---: | ---: | ---: |
| Trained or validated on | 117 | −0.542 | −0.463 | 92/117 | 3.4 × 10⁻¹⁰ |
| **Never trained on** | **81** | **−0.459** | **−0.392** | **45/81** | **0.37** |

**The significance came almost entirely from the contaminated domains.** On the 81 it never
saw, ThermoMPNN-D wins 45 of 81, which is chance. The median gap of 0.067 survives, so the
advantage is probably real but inconsistent: it is carried by a subset rather than holding
broadly.

Two consequences. First, **route C is back in contention** rather than ruled out: the closest
published analogue of this architecture does not reliably beat the frozen likelihood out of
distribution, which is weak evidence that the frozen representation, not the head, is the
constraint. Second, **route B becomes the most attractive first move**, because ThermoMPNN-D's
objective is absolute ΔΔG while route B targets the interaction term directly; its difficulty
is evidence about its own objective, not about a sharper one.

A confound to carry with this. Clean and contaminated groups are not a controlled contrast: the
splits came from homology clustering, and the frozen model also drops on the clean group
(−0.392 against −0.463) despite having no exposure to those splits at all. Part of the gap
between groups is domain difficulty. The paired comparison within the clean group is the part
that holds, and that is the comparison reported above.

**Route A — unfreeze the top of the decoder.** Train the last decoder layer with the head,
everything below frozen. The gate the plan held shut is open now that the target is measured
ΔΔG. Cheap, and the obvious next increment if capacity is the limit. Risk: 68,273 contrasts
across 208 domains is a small set for fine-tuning, so cluster-held-out evaluation is the only
number worth reporting.

**Route B — a sharper target than total stability.** The model already predicts the additive
part well; that is what the −0.448 first-order term is. What it misses is the interaction:
ΔΔG(ij) − ΔΔG(i) − ΔΔG(j). The dataset has singles at every site, so this label is
constructible, and training on it attacks exactly the residual instead of re-learning burial.
Our own measurement supports this framing: coupling pooled to +0.250 but only +0.105
one-per-domain, so the interaction is real but small and currently drowned by the additive
signal in a total-stability objective. **This is the strongest route and may be worth trying
before A.**

**Route C — information ProteinMPNN does not have.** Its features are backbone-only: N, Cα, C,
O and a virtual CB. It never sees side-chain volume or burial depth. If a trained head
saturates near −0.501, the ceiling is the representation, not the model, and the test is to add
explicit per-site SASA and a SAP term as input features. The machinery already exists in
`Code/descriptors/sap.py`. A gain here would be a cleaner finding than a gain from more
capacity, because it names what was missing.

**Route D — narrow the claim to where arrangement decides the answer.** The additivity test
found 5 of 74 networks where the preference reverses with a third residue, concentrated in
weak-effect pairs of 0.1–1.2 nats. Those are exactly the cases the model is least certain
about and where a scorer would be most useful. Stratify by model confidence and report the
conditional improvement. A scorer that helps only on low-confidence pairs is a narrower claim
but still a measured one.

**And the honest floor.** If none of these clear the baseline, the result is that frozen
ProteinMPNN already carries the arrangement information a patch-level objective can extract.
That bounds what patch rescoring can add, the decomposition explains the mechanism, and it is
reportable. The design's purpose is that this cannot be mistaken for failure to try.

## Schedule

| Dates | Work |
| --- | --- |
| November 7–8 | **PSC, before the outage.** Build the venv, validate against one published number, run all 208 domains at `--distance 12 --threshold 100`, retrieve results. In parallel: reinstall `parasail`, cluster the 208 domains, fix the split. |
| November 9–12 | Cache decoder features for the resolved contrasts. Train the frozen head. Run the composition-only, shuffled-label and sub-floor controls. |
| November 13–17 | Primary test on held-out clusters. Join and compare ThermoMPNN-D, split by training-set membership. Choose a contingency route if needed. |
| November 18–24 | Patch-size generalization on the 85 pilot pairs. Ablate attention versus mean pooling, neighbour identities, and SAP as an input feature. |
| November 25–December 1 | The chosen route from section 3, or eSOL secondary validation. Failure analysis. |
| December 2–5 | Final report and poster. |

**One cluster dependency, and it is first.** Only the ThermoMPNN-D run wants a GPU, which is
why it is scheduled before November 9. Everything after it runs on a laptop: the features cache
once, the head is small, and the baseline scoring is already done and retrieved.

## Risks, and what each would mean

- **The head does not beat −0.501.** The most likely single outcome. Section 3 is the response,
  and the finding is real either way.
- **ThermoMPNN-D trained on our domains.** Then it is a memorization ceiling, not a baseline,
  and the comparison must be labelled as one. `mega_splits.pkl` decides this and is not in the
  repository.
- **It beats the baseline pooled but not per-domain.** An artefact of domain imbalance. The
  per-domain test is primary for this reason.
- **It beats k = 2 but does not generalize to larger patches.** The claim narrows to pairwise
  arrangement, which is still measured and still novel.
- **Additivity fails at k = 3.** Already partly answered: 90% of a swap effect is independent of
  a third residue, so pairwise labels do carry. The remaining tenth bounds how far they carry.
- **The outage slips earlier than November 9.** Then ThermoMPNN-D is unavailable, section 3 has
  no selector, and the routes are tried in the order B, A, C.

Each of these is reportable. The design's purpose is that none of them can be mistaken for
success.

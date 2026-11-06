# Patch scorer: specification

Draft of November 6, 2026. The scorer is the project's central unbuilt piece: the proposal's
October 26 milestone (encoder and contrastive loss) and November 2 milestone (property heads)
were both missed, and the mid-report plan records that a new scorer is not required for the
midpoint. This specifies what to build, what it is allowed to be called, and what would count
as it working.

## Prerequisite, stated first

**This cannot be trained on the current seven proteins.** Seven independent clusters cannot
support a train/held-out split that means anything, and the plan is explicit that tiny-set
training is debugging rather than evidence of generalization. The scorer depends on the
expanded dataset described in the [eSOL dataset plan](eSOL_Dataset_Plan.md). Building the
model before the data exists produces a number nobody can defend.

## What this model is

A **patch arrangement scorer**: given a fixed backbone and an assignment of residue identities,
it scores how native-like the arrangement within a surface patch is.

It is not an aggregation predictor, and the distinction is load-bearing. The proposal names the
permutation-trained head "aggregation," which builds an unproven inference into the
architecture where it stops being visible. Permuted patches are not aggregation-positive
examples; the instructor's feedback says so directly. Train it, name it for what its labels
are, and test separately whether that quantity predicts anything measured.

| Head | Pair construction | What the label actually is | Honest name |
| --- | --- | --- | --- |
| 1 | within-patch permutation | native vs. rearranged | **arrangement nativeness** |
| 2 | SKEMPI mutation pairs | measured ΔΔG of binding | **interface effect** |
| 3 | per-variant stability data | measured folding stability | **stability effect** |

Only heads 2 and 3 carry property labels. Head 1 is self-supervised and is the only one that
isolates arrangement at fixed composition, which is why it is the primary head despite not
being a property head at all.

## Architecture: frozen features, small trained head

**Use the frozen ProteinMPNN encoder as a feature extractor and train a small patch head on
top of it.** No gradient flows into ProteinMPNN.

Three reasons, in order of importance.

**The receptive field is matched by construction.** The project's claim is that a patch term
adds something beyond frozen ProteinMPNN. If the scorer sees less of the protein than
ProteinMPNN does, a negative result is uninterpretable: the objective may have failed, or the
encoder may simply have been starved of context. Sharing the representation makes the
comparison isolate the one variable that matters, which is the training objective.

**It avoids the fine-tuning gate.** The plan allows fine-tuning only once a reward is
validated, and nothing is validated. Freezing the encoder keeps this on the near side of that
gate.

**It is cheap.** A small head over cached features trains in minutes on the hardware already
in use, and features can be precomputed once per structure.

### Concretely

- **Node features**: per-residue encoder embeddings from frozen ProteinMPNN, taken with the
  same k = 48 neighbour graph the checkpoint defines (`k_neighbors=checkpoint["num_edges"]`).
  The whole chain is encoded; localization happens at the readout, not by cropping the input.
- **Readout**: pool over the patch's member positions only. Start with attention pooling over
  patch members; mean pooling is the ablation.
- **Head**: two or three dense layers to a single scalar. Keep it small deliberately — a large
  head on frozen features will memorize structures rather than learn arrangement.
- **Output**: one scalar per patch per assignment. Higher means more native-like.

### What must not be in the features

Anything that leaks composition or native chemistry:

- No native side-chain coordinates after identities are reassigned. The existing pipeline
  already forbids this and the invariant checks enforce it.
- No composition summary of the patch as an input feature. Composition is matched exactly
  within every training pair, so such a feature carries no gradient — but it would carry one
  across proteins and let the model solve the cross-protein task the easy way.

## Training signal

### Head 1, the primary head

For each patch, one native assignment and **K permutations** of it, composition-matched. This
is the regime the whole project is built around and the data is free to generate.

Loss is **ranking, not regression**. The measured arrangement effect is roughly 1.4 nats
against an order-noise floor near 1.0 nat, so individual pair gaps are not reliable targets;
only the ordering is. Use an InfoNCE-style objective over one native against its K permutations:

```
loss = -log( exp(s_native / T) / sum_j exp(s_j / T) )
```

with the permutations as in-batch negatives. Permutations of the *same* patch are the only
valid negatives for this head: a permutation of a different protein differs in composition and
would let the model win on composition alone.

**Margin calibration.** Do not ask the model to resolve differences below the measurement
floor. Pairs whose full-chain gap is within the order-noise band are unreliable labels — the
1ENH pair is the worked example, with noise two to ten times its own effect. Either drop them
or down-weight them, and declare which before training.

### Heads 2 and 3

Ranking losses over measured values: for a pair of variants with measured ΔΔG or stability,
the model must order them correctly. These heads need their datasets audited first, exactly as
eSOL was. Head 3 depends on the per-variant stability audit that has not been run.

### Shared trunk

The proposal's shared trunk with per-property heads is sound, with one condition: **train head
1 first and alone**, and establish what it learns, before adding heads. A multi-task model that
underperforms gives no information about which objective was at fault.

## Controls that must exist before any result is believed

These are not optional extras; each one falsifies a specific way the result could be hollow.

| Control | What it rules out |
| --- | --- |
| Composition-only model | Scores at chance by construction, since composition ties exactly within every pair. If it does not, the pairs are broken. |
| Shuffled labels | Scores at chance. If it does not, the split leaks. |
| Frozen ProteinMPNN likelihood | The baseline to beat. A scorer that does not beat it adds nothing. |
| SAP surrogate | The closed-form baseline. Also the one most likely to be mistaken for the model's own behaviour. |
| Held-out clusters only | Use the [homology clustering](../Code/homology/cluster.py); never report on training clusters. |

The composition control is the strongest of these and the cheapest: it is a guaranteed null.
Run it first, every time the dataset changes.

## What counts as success

Declare these before training, not after.

**Primary**: on held-out homology clusters, the scorer ranks native above permuted more often
than frozen ProteinMPNN does, by a margin that exceeds the order-noise floor.

**Secondary**: the scorer's patch-level output, aggregated to protein level by a rule fixed in
advance, correlates with a measured property on held-out clusters. This is where the honest
naming pays off — if head 1's output predicts measured solubility, that is a finding, and it is
the step that earns the word "aggregation".

**Reporting rule**: improving a score the model was trained on is expected by construction and
is not evidence. The scope note already says this; it applies with full force here.

## What this scorer still will not show

Even if everything above succeeds, the result is that a patch score **predicts** a property
better than existing baselines. It does not show that redesigning a patch **improves** that
property. Prediction and intervention are different claims, and closing the gap requires making
the redesigned protein and measuring it.

## Build order

1. Expand the dataset per the eSOL plan; run the structure filter and recompute power.
2. Cluster with the homology module and fix the split. Never redraw it afterwards.
3. Cache frozen ProteinMPNN encoder features per structure.
4. Implement the composition-only control and confirm it scores at chance.
5. Train head 1 alone. Compare against frozen ProteinMPNN and SAP on held-out clusters.
6. Only then consider further heads, and only then consider fine-tuning, which remains gated
   behind a validated reward.

Steps 1 and 2 are prerequisites, not parallel work. Step 4 is the cheapest insurance in the
whole plan.

# Supervised patch scorer: architecture and data contract

Draft, November 6, 2026. Companion to [Patch_Scorer_Spec.md](Patch_Scorer_Spec.md).
This is a proposed implementation contract, not a report of trained models.
It makes explicit choices where the existing specification and feature code differ.

## 1. Task and model comparisons

Given one backbone, one patch and two composition-matched sequence assignments,
predict which assignment has greater measured folding stability. Train on measured
two-site swaps; larger-patch predictions are an unvalidated transfer experiment.

Use these distinct model names:

| Model | Inputs | Output and purpose |
| --- | --- | --- |
| M0: frozen ProteinMPNN | Whole-chain backbone and sequence | Negative NLL, with higher meaning preferred; reference baseline, no training. Keep full autoregressive and first-order baselines separate. |
| M1: supervised additive | Candidate identity plus geometry-only features at each patch position | Sum of independent learned residue scores. Tests learning stability-specific position preferences. No neighboring identities or sequence-conditioned features. |
| M2: contextual pooled head | Frozen target-last log probabilities, identities, geometry and role flags | Shared token projection, mean over patch tokens, small scalar head. Tests learning from ProteinMPNN context without new attention. This is NOT a position-independent additive model. |
| M3: contextual interaction head | Exactly the same node inputs as M2, plus explicit pair geometry | Small self-attention block over patch and context tokens, patch pooling, scalar head. Primary proposed scorer. M3 versus M2 tests added attention/pair geometry, not all physical interactions. |

M1 remains necessary: the frozen contextual features already encode sequence
interactions. Comparing M3 only with M2 cannot isolate the benefit of learning
interactions from scratch. A wider M2 with comparable parameter count is a capacity
control. An optional geometry-only M1 plus explicit pair terms provides a further
clean additive-versus-pairwise comparison; it is not required for the first head run.

## 2. One training example

| Field | Type / example | Role |
| --- | --- | --- |
| `contrast_id` | Unique stable string | Audit and deduplication; never a model feature. |
| `parent_id`, `cluster_id`, `split` | Strings | Provenance, leakage control and grouped evaluation; never model inputs. |
| `backbone_path`, `backbone_sha256` | Path, hash | One sequence-verified parent backbone shared by both assignments. |
| `residue_mapping` | PDB residue ID to observed/model index | Explicitly reconcile mutation numbering and backbone indexing. |
| `sequence_A`, `sequence_B` | Canonical amino-acid strings, each length L | Full sequences presented separately to the same scorer. |
| `patch_indices` | Two zero-based indices for training | Same positions for both assignments; not literal PDB residue numbers. |
| `stability_A`, `stability_B` | Finite measured estimates, kcal/mol | Labels only. Larger means more stable under the verified source convention. |
| `difference` | `stability_A - stability_B` | Supervision; not a feature. A positive value means A should score higher. |
| `interval_A`, `interval_B`, `resolved` | Source uncertainty and eligibility metadata | Filtering/audit only; not features. |
| `source_rows`, `quality_policy_version` | IDs / string | Reproduce the source labels and exclusions. |

Validate before loading: A and B have the same length, patch counts and outside-patch
identities; both match their source mutation assignments; the backbone sequence maps
to their parent. Emit each unordered swap contrast once. All appearances of a parent,
sequence, site pair and close homologues must remain in the assigned cluster split.

Do not substitute generated-pair labels for measurements. Do not silently bypass
the source `_ML` exclusions by reading raw `deltaG`. The current contrast extractor
uses raw `deltaG` by default; establish and version a quality policy before declaring
the final training cohort. The reported 69,633 resolved contrasts are a candidate
cohort from that extractor, not a guarantee that the final audited count is identical.

Its current uncertainty rule is `u = sqrt(interval_A**2 + interval_B**2)/2` for full
confidence-interval widths, followed by `abs(difference) > 2*u`. Retain this as a
named, versioned screening heuristic if used. A confidence-interval half-width is
not a standard error; do not describe the rule as a calibrated two-sigma test or
assume independent measurement errors without checking shared assay uncertainty.

## 3. Shapes and region definition

Notation: B = number of assignments in a feature batch; L = full chain length;
k = patch size; n = context count (at most 8); T = k+n; D = 64; H = 4 attention heads.
Paired training batches contain C contrasts and therefore 2C assignments.

Backbone inputs follow the existing ProteinMPNN wrapper:

| Tensor | Shape | Contents |
| --- | --- | --- |
| `X` | `[B,L,4,3]` | N, CA, C, O coordinates, Angstrom. |
| `S` | `[B,L]` | Integer identities in the checkpoint's 21-symbol alphabet. Real sequences use canonical residues. |
| `mask` | `[B,L]` | Valid residue positions. Reject unsupported missing mappings rather than guessing. |
| `chain_M` | `[B,L]` | Existing ProteinMPNN design/score mask; not a mask that hides all patch identities. |
| `residue_idx` | `[B,L]` | Model residue indices retaining the staging convention. |
| `chain_encoding_all` | `[B,L]` | Model chain identifiers. Initial task uses one staged chain. |

Define context N as up to eight non-patch residues in the same chain with smallest
minimum CA distance to any patch member. Break ties by model sequence index. Choose
N from geometry only and share it between A and B. The region U is P union N.
Eight is an initial frozen configuration, not a measured optimum. It is distinct
from ProteinMPNN's internal k=48 neighborhood graph.

Encode the full chain; do not crop it to U. Geometry may be cached per parent/region.
Sequence-dependent features must be cached per unique full sequence, region and order.

## 4. Unit-by-unit primary model, M3

### Unit A: frozen whole-chain ProteinMPNN

**Input:** whole-chain tensors above, with either sequence A or B.

**Operation:** fixed vanilla `v_48_020` checkpoint initially; evaluation mode, zero
backbone noise, no gradients. Use the checkpoint and source hashes recorded for the
baseline. A soluble-checkpoint run is a separate experiment, not additional channels
silently added to the initial head.

**Output:** internal geometry representations reused by the model, followed by the
target-last decoder calls below. Backbone-only encoder outputs do not vary between
A and B. They are useful geometry inputs when combined with identity, but cannot
distinguish assignments on their own.

### Unit B: target-last sequence-context extraction

For each target i in P, use the same deterministic order of all other positions
and put i last. Other patch positions remain visible through teacher-forced sequence
conditioning. Use identical orders for A/B; order tags must not depend on sequence
identity, class label or contrast orientation.

**Input:** Unit A's full-chain inputs plus target i and the recorded decoding order.

**Output:** `ell_i`, 21 log probabilities at i; stack into `[B,k,21]`.

This is what `Code/scorer/features.py::conditional_logprobs_at` currently exposes.
It returns output probabilities, not hidden decoder embeddings. Interpret these as
target-last autoregressive features: placing i last exposes other identities, but
does not prove equality with an order-independent conditional obtained from one
consistent joint distribution. Order sensitivity remains a model property.

Cost: k decoder evaluations per sequence, batched across variants, under the current
wrapper. Calling `model.forward` also repeats encoder work. Encoder sharing would
require a separately verified optimization; do not budget one forward per variant
or cache one contextual feature vector per backbone.

### Unit C: build residue tokens

For every t in U form a 48-channel token:

| Channels | Width | Value |
| --- | ---: | --- |
| Context vector | 21 | `ell_t` for patch members; zeros for context-only members. |
| Candidate identity | 21 | One-hot of the current sequence identity at t. |
| Geometry descriptors | 4 | Defined below; fixed across assignments. |
| Role flags | 2 | `is_patch`, `has_logprobs`; both 1 for patch members, both 0 for context-only members. |

Geometry descriptors: (1) other-CA count within 10 A divided by 32; (2) minimum CA
distance to a patch member other than self divided by 20 A; (3) mean of those
distances divided by 20 A; (4) minimum absolute sequence-index separation to such
a member divided by 32. Clip each to [0,1]. Define empty-set distances as zero and
reject empty patches; the initial measured task has k=2. These simple scalings use
no label statistics. If log-probability standardization is used, fit it on training
groups only and save the transform; do not fit on validation/test assignments.

**Input:** `[B,k,21]`, full sequence, P/N indices and fixed geometry.
**Output:** `F` `[B,T,48]`, patch mask `[B,T]`, valid-token mask `[B,T]`.

The current `patch_feature_vector` flattens differently sized regions. Add a new
token-building interface rather than feeding that flat vector to a fixed-width MLP
and assuming it supports arbitrary patch sizes.

### Unit D: token projection

Apply shared `Linear(48,64) -> ReLU -> LayerNorm(64)` to every valid token.

**Input:** `[B,T,48]`.
**Output:** `Z` `[B,T,64]`.

Do not add arbitrary token-order positional embeddings. Residue role, geometry and
sequence separation supply position information; output should be invariant to
reordering the token list while preserving those associations.

### Unit E: one geometry-aware self-attention block

For every unordered region pair, derive eight Gaussian distance features with centers
2,4,...,16 A and width 2 A, plus `min(abs(i-j)/32,1)` and an indicator for CA contact
within 8 A. This gives pair features `[B,T,T,10]`; all derive from the shared backbone.
Map them through `Linear(10,16) -> ReLU -> Linear(16,4)` to one attention bias per head.
Use symmetric pair features and count no direction-specific identity feature here.

Apply four-head self-attention (16 channels per head), scaled dot-product logits
plus those biases, with padding keys masked. Follow with a residual connection,
layer norm and a tokenwise `64 -> 128 -> 64` ReLU feed-forward network with another
residual/layer norm. Initial dropout is 0.1 during training and off in evaluation.
No global CLS token or extra attention layers in the first experiment.

**Input:** Z `[B,T,64]`, pair features, valid-token mask.
**Output:** context-updated tokens `Z_prime` `[B,T,64]`.

This is the trainable interaction unit: a patch token can respond jointly to its
partner and context tokens. It supplements interactions already present in Unit B;
it does not isolate physical three-body energies.

### Unit F: patch pooling and scalar head

Take a masked mean of updated patch-member tokens only. Context tokens influence
the patch through attention but are not independently averaged into the readout.
Then `Linear(64,32) -> ReLU -> Linear(32,1)`.

**Input:** `[B,T,64]` plus patch membership.
**Intermediate:** pooled patch representation `[B,64]`.
**Output:** score `s` `[B]`; larger means predicted more stable in a within-patch
comparison. Scores are dimensionless ranking outputs, not calibrated kcal/mol,
probabilities, aggregation risk or absolute protein stability.

The same weights accept k=2 and k=8 through padding and masks. Supporting the shape
does not establish generalization across patch sizes. Compare scores only for the
same backbone, patch and context definition; do not rank different patches by raw s.

### Unit G: shared-weight comparison

Both assignments pass through exactly the same Units A-F with the same backbone,
patch and context. Do not give the scorer an A/B identity feature.

**Inputs:** `s_A`, `s_B` `[C]`.
**Output:** `predicted_difference = s_A-s_B` `[C]`.

At inference, positive predicts A more stable; negative predicts B more stable.
No measured label is needed for inference. Define any decision/tie tolerance on
validation groups before reporting test results; raw score magnitude is not confidence.

## 5. Training objective and configuration

Let `d = stability_A-stability_B` and `y=sign(d)`. On eligible resolved contrasts:

```text
margin(d) = clip(abs(d) / (1 kcal/mol), 0.1, 2.0)
loss = max(0, margin(d) - y * (s_A-s_B))
```

The margin is dimensionless and uses the measured magnitude without a regression
target. It is a proposed initial rule to freeze before training. Reverse contrast
orientation randomly with corresponding label reversal during training. No reversal
can move an example out of its assigned split.

Train only Units D-F and the pair-bias network, with AdamW, learning rate 0.001,
weight decay 0.0001, batch size 64 contrasts, maximum 100 epochs and seeds 7/17/27.
Balance contributions by parent domain. Early-stop after 10 epochs without improved
mean per-domain validation ranking accuracy; save a deterministic tie-breaking rule
for checkpoint selection. M1/M2 use the same training examples, objective and tuning
budget. Model size and padding limits may be reduced if local memory requires it,
but record the changed configuration before the evaluated run.

No backbone fine-tuning or generator optimization in this phase. Neither valid
measured labels nor a successful training curve alone makes the learned score a
validated reward on arbitrary generated sequences.

## 6. Inputs and outputs of the controls

| Model/unit | Input | Output | Trainable |
| --- | --- | --- | --- |
| Composition null | 20 residue counts in P | Identical score for A and B | Optional fitted control; contrasts must tie regardless. |
| M0 full | Whole-chain X and S, eight shared recorded orders | Negative total autoregressive NLL per assignment; compare score differences | No |
| M0 first-order | Whole-chain X and candidate identities | Negative sum of backbone-only log-probability costs | No |
| M1 residue unit | One-hot identity (21) + geometry (4), no contextual log probabilities | One scalar per patch residue, `25 -> 32 -> 1`; sum over P | Yes |
| M2 contextual unit | Same 48-channel tokens/projection as M3, no attention/pair bias | Patch-only mean of projected tokens, then `64 -> 32 -> 1` | Yes |
| M3 interaction unit | Same tokens plus pair geometry | Attention-updated patch mean and scalar | Yes |

M1's fixed-geometry score has zero double-mutant nonadditivity. M2 does not share
that property: its input already changes with other identities. M2's context-only
tokens cannot influence a patch-only mean without attention; whole-chain sequence
context still enters through its frozen patch log probabilities. State this when
interpreting M3 versus M2. A separate pair-bias-disabled M3 isolates explicit geometry
bias from attention capacity; these ablations are distinct questions.

## 7. Evaluation and output records

Freeze whole homology-cluster assignments before feature fitting or training; use
`Code/homology/cluster.py` and its split verification. Existing public baseline
results are exploratory references, not substitute held-out numbers.

For each model, use score differences with the same higher-is-more-stable sign.
Compare with measured d on exactly the same eligible held-out contrasts. Report
per-domain Spearman (require at least 20 eligible contrasts and nonconstant values),
its median/distribution, and mean per-domain ranking accuracy. Composition ties
receive half credit. Report exclusions and domains with undefined correlations.

Recompute M0 on the same held-out subset; the published full-panel -0.497 NLL
correlation becomes approximately +0.497 when negated, but is not a fixed held-out
threshold. Compare per-domain improvements with uncertainty resampled over homology
clusters; domains in the same cluster are dependent. A domain-level sign test can
be descriptive, not a claim of independent observations within clusters. Shuffled
labels and unresolved contrasts are separate diagnostics, not extra training data.

Prediction CSV contract:

```text
contrast_id,parent_id,cluster_id,split,patch_id,model,seed,
score_A,score_B,predicted_difference,measured_difference,resolved,
predicted_preference,checkpoint_hash,feature_cache_hash
```

Record config, source/weights/data/split hashes, epoch selection, cache provenance,
all seed results and failures. Do not choose a seed by test performance. Larger
pilot-patch agreement is a consistency check only. Model-based third-residue
sensitivity is not measured experimental higher-order stability, and even successful
three-site evaluation would not prove arbitrary patch-scale generalization.

## 8. Required validation before training

1. Same assignment and order produces identical features; batching does not change results.
2. Every target is decoded last; changing its own supplied identity leaves its context
   log-probability vector unchanged under a real checkpoint, while changing a visible
   partner can change it. The candidate identity one-hot supplies the target identity.
3. Validate actual model behavior, not just a stub that records passed orders. Check
   target-last forward features against a documented upstream conditional path.
4. Composition counts tie; swapped input orientation negates the predicted difference.
5. Padded tokens and token-list permutations leave predictions unchanged in eval mode.
6. ProteinMPNN parameters receive no gradients; attention/head parameters do. Cached
   inference-mode tensors must be copied to ordinary tensors before gradient training.
7. Geometry never uses reassigned native side-chain chemistry. Region selection is
   independent of labels/sequence assignments. All numeric inputs and predictions are finite.
8. Split verification rejects shared parents/homologues; transformations use training data only.
9. M1 gives zero nonadditivity on fixed geometry; no such assertion is made for M2/M3.

## 9. Implementation state and proposed files

Present: `Code/scorer/features.py` supplies per-target conditional log probabilities,
identity encoding and a flat vector helper. Stub tests verify layout/order plumbing.
It does not yet supply this token schema, pair geometry, attention, trained heads,
ranking trainer or end-to-end real-checkpoint validation.

Proposed additions (not implemented by this document):

- `Code/scorer/dataset.py`: audited contrast loader, mappings, splits and paired batches.
- `Code/scorer/cache_features.py`: frozen extraction; cache keys include full-sequence,
  backbone, target/context membership, order/seed, alphabet, code and checkpoint hashes.
- `Code/scorer/models.py`: M1/M2/M3 and capacity controls.
- `Code/scorer/train.py`: paired margin training and validation-only checkpoint choice.
- `Analysis/evaluate_patch_scorer.py`: held-out metrics and grouped uncertainty.
- `Data/`: small versioned manifests; large inputs/caches stay under ignored raw/cache paths.
- `Analysis/patch_scorer/`: configurations, prediction tables and figures when runs exist.

Do not change the README's regression plan silently: it is an alternative measured
mutation task. Adopting this document makes within-swap ranking the primary experiment;
update the main plan explicitly when the team agrees on that choice. No README,
training code, dataset, Git history or remote service is modified by this draft.

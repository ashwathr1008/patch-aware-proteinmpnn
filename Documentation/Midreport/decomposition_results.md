# Decomposing the arrangement gap: first-order burial versus joint structure

Draft results section, November 4, 2026. This supersedes the framing of
[controlled_scoring_results.md](controlled_scoring_results.md), which it does not replace
wholesale; the closing subsection lists the specific sentences there that must change.

**Units.** This section reports **totals in nats per chain**. The earlier section reports
per-residue means. A total is a per-residue mean multiplied by the chain length, so the two
are the same measurement on different scales. Totals are used here because the decomposition
is additive only on totals.

## Why the published result was under-determined

The pilot reported that both frozen checkpoints prefer the original arrangement in 84 of 85
composition-matched pairs. The claim drawn from it — that ProteinMPNN detects rearrangements
despite identical ingredient counts — is true as stated but does not distinguish two models of
the network, which point in opposite directions:

- **First-order burial.** The network holds position-specific marginal preferences driven by
  burial and nothing else. A permutation moves a hydrophobic residue to an exposed position
  whose marginal probability for it is low, and the chain total rises with no coupling
  involved. This model alone predicts 84/85. It is also burial-weighted composition, which is
  the closed-form SAP statistic of Chennamsetty et al. (2009), not a learned representation of
  arrangement.
- **Joint structure.** The network represents which positions' identities depend on each
  other, so breaking a combination costs it beyond any position-by-position mismatch.

If the gap were entirely first-order, the premise motivating a patch-level scorer would be
restored in a sharper form. If the gap were substantially joint, the bar for an added scorer
would be higher. The experiment as published could not tell these apart, so we measured the
split directly.

## Method

ProteinMPNN exposes the first-order distribution. `ProteinMPNN.unconditional_probs` runs the
encoder and decoder with an all-zero backward order mask, so no sequence information reaches
any position and the output is `p(aa_i | backbone)` at every position. It takes no decoding
order, so the first-order score is **order-free by construction**, and one forward pass per
structure and checkpoint serves every pair on that structure. For an assignment `a`:

```
score_fo(a) = - sum_i log p_i^uncond(a_i)
delta_fo    = score_fo(permuted) - score_fo(original)
delta_joint = delta_full - delta_fo
```

`delta_full` is the order-averaged full-chain total from the existing scored run, so the
decomposition adds forward passes for the first-order term only and reuses every scored row.
Fourteen unconditional passes cover the seven structures and two checkpoints; the run took
under a minute on CPU.

**Invariant check.** Because unconditional probabilities ignore sequence context and these
pairs change identities only inside the patch, every outside-patch position must contribute
*exactly* zero to `delta_fo`. The run asserts this per pair rather than assuming it. All 85
pairs on both checkpoints returned an outside-patch first-order gap of `0.00e+00`. This is a
strong independent check that patch indices are aligned with the model's own featurization,
obtained free of any additional comparison.

Reproduce with:

```bash
python Code/scoring/score_first_order.py \
  --pairs Data/compatibility_pairs/pairs.json \
  --mpnn-dir <pinned ProteinMPNN directory> \
  --scores Analysis/paired_expanded/pair_scores.csv \
  --output Runs/first-order-control --device cpu
```

## Result: the gap is roughly nine parts burial to one part arrangement

Medians over 85 pairs, nats per chain.

| Quantity | Vanilla | Soluble |
| --- | ---: | ---: |
| Full-chain gap `delta_full` | 13.681 | 13.311 |
| First-order burial `delta_fo` | 12.194 | 11.944 |
| Joint residual `delta_joint` | 1.419 | 1.471 |
| First-order share of the gap | **0.885** | **0.893** |
| Pairs where the first-order term exceeds half the gap | 83/85 | 83/85 |
| Least-squares fit of `delta_full` on `delta_fo` | R² = 0.950, slope 1.036 | R² = 0.956, slope 1.035 |

The first-order term alone reproduces the full-chain gap almost exactly: R² = 0.95 with a
slope indistinguishable from one. **Most of the published effect is burial-weighted
composition**, which requires no network and no notion of arrangement.

The residual is nonetheless not zero and not random. Sign tests over the 85 pairs:

| Sign test on | Vanilla | Soluble |
| --- | --- | --- |
| `delta_fo` | 85 positive, 0 negative, p = 5.2e-26 | 84 positive, 1 negative, p = 4.4e-24 |
| `delta_joint` | **74 positive, 11 negative, p = 1.3e-12** | **71 positive, 14 negative, p = 2.4e-10** |

So both readings hold at once: the dominant term is first-order burial, and a small joint term
is systematically present. The joint term never overturns a conclusion — it reverses the sign
of the full gap in 1 of 85 pairs on vanilla and 0 of 85 on soluble.

### Where the joint term lives

`delta_fo` is confined to the patch by construction, so the gap at outside-patch positions is
**provably free of first-order content** and is attributable to sequence coupling.

| Component of `delta_joint` (median, nats) | Vanilla | Soluble |
| --- | ---: | ---: |
| Outside the patch | 1.112 (73/85 positive) | 0.807 (72/85 positive) |
| Inside the patch | 0.749 (58/85 positive) | 0.711 (62/85 positive) |

Roughly half the joint signal appears at positions whose identities never changed. Those
positions cannot respond to burial, so this is the cleanest available evidence that the network
is reading arrangement rather than composition alone.

### By source protein

Means over each protein's pairs, nats per chain. See
[decomposition.pdf](../../Analysis/decomposition/decomposition.pdf).

| Source | Split | Pairs | `delta_full` V/S | `delta_fo` V/S | `delta_joint` V/S | First-order share V/S |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1CRN:A | Development | 15 | 26.41 / 25.13 | 24.38 / 22.97 | 2.02 / 2.16 | 0.904 / 0.894 |
| 1UBQ:A | Development | 15 | 14.51 / 15.55 | 13.09 / 13.93 | 1.42 / 1.62 | 0.916 / 0.926 |
| 2CI2:I | Development | 15 | 15.74 / 14.93 | 12.67 / 12.24 | 3.07 / 2.69 | 0.821 / 0.831 |
| 1VII:A | Development | 15 | 8.01 / 8.11 | 7.45 / 7.60 | 0.56 / 0.51 | 0.885 / 0.925 |
| 1SHG:A | Diagnostic | 15 | 14.43 / 13.68 | 12.34 / 11.90 | 2.10 / 1.78 | 0.879 / 0.868 |
| 1L2Y:A | Diagnostic | 5 | 15.30 / 14.71 | 14.26 / 13.71 | 1.04 / 1.01 | 0.915 / 0.947 |
| 1ENH:A | Diagnostic | 5 | 8.65 / 9.56 | 7.72 / 7.83 | 0.93 / 1.73 | 0.852 / 0.841 |

Two patterns are worth one sentence each and no more, since seven proteins cannot support a
structural claim. **1CRN has by far the largest gaps and they are ~90% first-order**, which is
consistent with crambin's high hydrophobic content: the permutations have more burial mismatch
available to create. **2CI2 is the most arrangement-sensitive** of the seven, with the lowest
first-order share (0.82) and the largest joint term, reaching 15/15 positive on vanilla.

## Decoding-order inconsistency, measured

For an autoregressive model whose conditionals come from one consistent joint distribution, the
chain total is identical under every decoding order, because a joint does not depend on its
factorization. Any spread across the eight shared orders is therefore amortization error in the
network. The existing 1,360 paired-order rows measure it at no compute cost, which converts the
order control into a measurement and answers the decoding-order challenge directly.

| Median over 85 pairs | Vanilla | Soluble |
| --- | ---: | ---: |
| Order SD of the chain total | 0.933 nats | 1.091 nats |
| Absolute full-chain gap | 13.681 nats | 13.311 nats |
| Ratio, order SD to gap | **0.063** | **0.066** |
| Pairs where order SD ≥ the gap | 1/85 | 1/85 |

The conditionals are measurably inconsistent, but at roughly **one-fifteenth of the effect
being reported**. The 84/85 result is therefore not an artifact of decoding order; it clears
the measurement floor with an order of magnitude to spare. See
[order_consistency.pdf](../../Analysis/decomposition/order_consistency.pdf).

## The 1ENH exception is below the instrument's resolution

`1ENH_A_patch_01_perm_005` is the single pair whose order SD exceeds its own gap, on both
checkpoints. Its decomposition is incoherent across checkpoints:

| Checkpoint | `delta_fo` | `delta_joint` | `delta_full` | Order SD | SD ÷ \|gap\| |
| --- | ---: | ---: | ---: | ---: | ---: |
| Vanilla | +0.280 | −0.777 | −0.498 | 1.006 | **2.02** |
| Soluble | −0.357 | +0.226 | −0.131 | 1.321 | **10.10** |

Both component terms change sign between checkpoints, and the measurement noise is two to ten
times the quantity being measured. This pair carries no information about arrangement. It
should be reported as resolution-limited rather than as a model-preference exception inviting
further diagnostics.

## Limitations specific to this section

- **`delta_joint` is a residual, not a coupling estimate.** It absorbs genuine sequence
  coupling together with any amortization error in the autoregressive conditionals. The two are
  not separated here.
- **No per-pair joint claim is supported.** The median joint term (~1.4 nats) is the same order
  of magnitude as the per-sequence order SD (~0.9–1.1 nats). Order noise is zero-mean, so the
  systematic result across 85 pairs stands, but individual pairs' joint values should not be
  quoted or ranked.
- The pairs are not independent: 85 pairs derive from 17 patches on seven proteins in six
  convenience groups, with overlapping patches within proteins. The sign-test p-values quantify
  consistency across pairs, not across independent proteins.
- First-order probabilities come from the same network, so "first-order" means this model's
  marginals, not an external burial measure. A SASA-weighted SAP baseline is the correct
  external comparison and is not yet computed.
- Inputs retain the caveats already recorded: designed 1L2Y, first NMR conformers, truncated
  domains, unconstrained disulfides, and possible pretraining overlap.

## What this changes in the existing results section

[controlled_scoring_results.md](controlled_scoring_results.md) needs four edits; everything
else there remains accurate.

1. **Closing claim.** "Existing ProteinMPNN detects most rearrangements despite identical
   ingredient counts" should become a decomposed statement: ProteinMPNN discriminates
   composition-matched rearrangements, and roughly 89% of that discrimination is first-order
   burial preference, with a small but systematic joint component (74/85 and 71/85 positive).
2. **Headline statistic.** The sign test should be reported on `delta_joint` as well as
   `delta_full`. Reporting 84/85 alone overstates what the experiment establishes about
   arrangement.
3. **Failure/edge case.** The 1ENH paragraph should state that the pair is resolution-limited
   — order SD 2× to 10× its gap — and drop the suggestion that it needs more diagnostics.
   Stratifying by changed-position count across more proteins remains worthwhile for other
   reasons.
4. **Order-averaging caveat.** The existing note that arithmetic averaging across orders is not
   an order-marginalized likelihood is correct and should now be followed by the measured
   consequence: the order spread is 6–7% of the effect.

A fifth item is outside this section's scope but follows from it: the hydrophobic-contact
descriptor ties in 44 of 85 comparisons, which makes it insensitive in half the dataset. A
continuous SASA-weighted form would tie far less often and is the external baseline the
limitations above call for.

## Artifacts

| Path | Contents |
| --- | --- |
| [`Analysis/decomposition/first_order_scores.csv`](../../Analysis/decomposition/first_order_scores.csv) | Per pair and checkpoint: `delta_fo`, `delta_joint`, shares, invariant column |
| [`Analysis/decomposition/first_order_summary.json`](../../Analysis/decomposition/first_order_summary.json) | Distributions, sign tests, fits, per-protein breakdown |
| [`Analysis/decomposition/order_consistency_pair.csv`](../../Analysis/decomposition/order_consistency_pair.csv) | Per pair: order SD, inconsistency ratio, outside-patch gap |
| [`Analysis/decomposition/order_consistency_summary.json`](../../Analysis/decomposition/order_consistency_summary.json) | Order-spread and outside-patch distributions and sign tests |
| [`Analysis/decomposition/first_order_run_record.json`](../../Analysis/decomposition/first_order_run_record.json) | Provenance, checksums, invariant checks |
| `Analysis/decomposition/decomposition.{png,pdf}` | Per-protein decomposition and joint-residual figure |
| `Analysis/decomposition/order_consistency.{png,pdf}` | Order spread against effect size |

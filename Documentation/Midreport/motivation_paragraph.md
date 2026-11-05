# Motivation: what a patch-level term could add that is not already there

Draft motivation section, November 4, 2026. Supporting figure:
[sap_two_regime.pdf](../../Analysis/sap_baseline/sap_two_regime.pdf). Numbers come from
[sap_summary.json](../../Analysis/sap_baseline/sap_summary.json) and the
[decomposition](decomposition_results.md).

## The paragraph

> Any case for scoring local arrangement has to survive two objections: that frozen
> ProteinMPNN already captures the effect, and that whatever remains is recovered by Spatial
> Aggregation Propensity (SAP), a closed-form statistic available since 2009. We tested both
> against the same surrogate, in two regimes that separate *which* residues a model chooses
> from *where* it puts them. When composition varies freely, across ProteinMPNN's own 48
> generated candidates, the model's likelihood ranking agrees with SAP substantially: Spearman
> +0.33 for the vanilla checkpoint and +0.71 for the solubility-trained one, positive in all
> six structure and checkpoint pools. When composition is held exactly equal and only the
> arrangement of residues within a surface patch changes, across 85 matched pairs on seven
> proteins, that agreement disappears: +0.05 vanilla and +0.00 soluble. The contrast is
> sharpest in the checkpoint designed for developability, which is simultaneously the most
> SAP-aligned of the two at selecting residues and the least at placing them. Solubility-
> oriented training therefore buys composition-level aggregation awareness and no arrangement
> sensitivity at all. A patch-level term addresses a blind spot that neither released
> checkpoint covers and that the obvious remedy, switching to the soluble weights, does not
> close. This establishes that such a term would not be redundant. It does not establish that
> it would help, which requires a trained scorer, reranking against frozen baselines at equal
> selection budget, and an outcome measured independently of the score being optimized.

## Supporting table

Spearman correlation with the SAP surrogate. The two regimes use the same statistic so they
are directly comparable.

| Regime | What varies | Unit of analysis | Vanilla | Soluble |
| --- | --- | --- | ---: | ---: |
| Composition (generation) | Which residues are chosen | 6 pools of 8 samples | +0.33 | **+0.71** |
| Arrangement (matched pairs) | Where residues sit | 85 pairs, 7 proteins | +0.05 | **+0.00** |

In the arrangement regime the model term is the first-order burial score from the
decomposition, which carries roughly 89% of the full arrangement gap. Correlating SAP against
the full-chain gap instead gives the same picture (+0.07 vanilla, +0.05 soluble).

## What the figure shows, and the honest reading of it

The open dots behind each bar are the per-structure values (arrangement) and per-pool values
(generation). They matter: in the arrangement regime the pooled near-zero is **cancellation,
not uniform blindness**. Per-structure correlations run from −0.33 (1SHG) to +0.62 (1UBQ) with
no consistent sign. The correct statement is that the model's response to rearrangement bears
no *consistent* relationship to SAP, not that it is independent of SAP on every protein.

Three further limits belong in the text rather than a footnote.

- **The burial term is a surrogate.** Staged inputs carry backbone atoms only, so accessibility
  is computed on the backbone plus a virtual CB rather than packed side chains. Part of the
  near-zero arrangement correlation could be surrogate error. This is deliberate as well as
  forced: an identity-dependent burial term would let native side-chain chemistry leak into the
  permuted features and would decide the comparison by construction.
- **The checkpoint difference in the generation regime is not established.** Soluble exceeds
  vanilla on all three structures, but that is three paired observations, p = 0.25. The
  direction is consistent; the magnitude is not supported.
- **Low SAP is not the objective.** SAP is a proxy developed largely on antibodies, and it has
  no upper guard: minimizing it rewards stripping surface hydrophobics without limit, at a cost
  to folding and function that SAP does not score. Stock ProteinMPNN already produces sequences
  well below native SAP — 47 of 48 generated sequences score lower than their native sequence,
  and in five of six pools every sample does. The opportunity is therefore not "ProteinMPNN
  makes aggregation-prone sequences." It is that ProteinMPNN is insensitive to arrangement at
  fixed composition, which is the regime where compatibility and developability can be traded
  against each other without changing the ingredient list.

## Why this framing and not a stronger one

"Superior" is not available from these experiments and should not appear in the mid-report. No
scorer has been trained, no candidate pool has been reranked, and no outcome has been measured
independently of the score being optimized. Reporting that a SAP-targeting method achieves good
SAP would be true by construction, which the [scope note](../../Planning/Midreport_vs_Final_Scope.md)
already identifies as the failure mode to avoid.

The claim this section supports is non-redundancy, which is a genuine precondition for the
project and is now measured rather than assumed. The step that would convert it into a claim of
benefit is the final-report core: one scorer, protein-separated evaluation, reranking the same
fixed pools at equal budget against strong frozen baselines, and measured outcomes where the
mapping audit shows they are feasible.

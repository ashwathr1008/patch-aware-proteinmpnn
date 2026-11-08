# What counts as a homology split, and what this project will declare

Written November 8, 2026, after `parasail` was installed and the 208 scored domains were
clustered for the first time. The threshold choice changes the available statistical power by
an order of magnitude, so it is settled here, in advance, against the literature and against
what was done to this dataset by others — not after seeing which choice flatters a result.

## Why the module's default was wrong for this dataset

At the module's default identity floor of 0.25, the 208 domains collapse to **42 clusters with
127 of them in one**, an effective sample size of 2.8. That cluster holds 1PGA, 1PGX and 1IGD,
which genuinely are protein G domains, alongside 1E0L, a WW domain, and 1A0N, an SH3. It is
single-linkage chaining, not homology.

The cause is measurable rather than suspected. These domains are a **median of 57 residues**
(range 30–72), and the **median identity among candidate pairs is 0.231**. The default floor
therefore sits inside the chance distribution for sequences this short. The module's own
`identity` docstring already warned of this: it "reaches 39% between unrelated proteins" on
short domains.

## What the literature says

**The ddG convention is 25%, and it is calibrated on full-length proteins.** S669 was built to
be non-redundant at 25% identity against training sets in common use, and the ProtDDG benchmark
clusters with MMseqs2 at a 25% cutoff. Neither was assembled from 57-residue domains.

**Rost's twilight zone makes the length dependence explicit.** Above roughly 30% identity, 90%
of pairs are homologous; below 25%, fewer than 10% are. Alignments separate similar from
non-similar structure unambiguously only above about 40% *for long alignments*. The curve being
revised there, Sander and Schneider's HSSP curve, is roughly proportional to the inverse square
root of alignment length over the range 7–80 residues — which contains this dataset's entire
length distribution.

**Recent protein-design practice uses 40%.** So the field brackets this project's domains from
both sides, and the bracket is wide because the right number depends on length.

**Sequence identity may not be sufficient at any threshold.** Several groups report that
MMseqs2-based splits still leak; that structural similarity above TM-score 0.5 survives
conventional sequence splitting; and, in one ddG preprint, that random splits inflate Pearson
R to about 0.55 through homologous leakage alone. `DataSAIL` exists for this problem.

## The HSSP curve applied to these lengths

`identity_floor(L) = 290.15 · L^(-0.562)`, as a percentage, clamped to the curve's calibrated
range of 10–80 residues rather than extrapolated beyond the data it was fitted on:

| Alignment length | Floor |
| ---: | ---: |
| 30 | 42.9% |
| 57 (this dataset's median) | 29.9% |
| 72 | 26.2% |
| 100 | 21.8% |

Across the 21,528 pairs here the floor runs from 26.2% to 42.9%, median 32.9%. It is
implemented as `--identity-threshold hssp`, evaluated per pair on the shorter sequence.

## The four clusterings, measured

| Clustering | Clusters | Largest | Effective (Kish) | Wins needed for p<0.05 |
| --- | ---: | ---: | ---: | ---: |
| identity ≥ 0.25 (module default) | 42 | 127 | 2.8 | 28 of 42 (67%) |
| **HSSP length-dependent** | **84** | **35** | **15.1** | **52 of 84 (62%)** |
| identity ≥ 0.40 | 106 | 16 | 44.5 | 64 of 106 (60%) |
| ThermoMPNN published (MMseqs2) | 153 | 8 | 50.7 | 90 of 153 (59%) |

Two things stand out.

**This project's clustering is more conservative than the published one at every threshold.**
Even the length-aware curve leaves a 35-domain cluster where ThermoMPNN's largest is 8. The
threshold is not the only reason: **linkage differs**. Single linkage merges transitively, so
one bridging pair joins two families, while MMseqs2's cascaded clustering is closer to
centroid-based and does not chain the same way. Some of the gap between 84 and 153 clusters is
that algorithmic difference rather than any disagreement about which pairs are related.

**Over-merging is the safe direction, and it is not free.** An over-merged cluster costs power;
an under-merged one invalidates the evaluation. But the cost here is real: Kish 15.1 against
50.7 is more than a threefold reduction in effective evidence.

## What this project declares

**Primary: `z ≥ 4` with the HSSP length-dependent identity floor.** Justified by Rost's length
dependence and this dataset's 30–72 residue range, chosen before any scorer was trained, and
conservative relative to the published clustering of the same data.

**Reported alongside, as sensitivity rather than as alternatives to pick from:** the same
analysis under identity ≥ 0.40, and under ThermoMPNN's own MMseqs2 clustering. A result that
holds under the conservative primary clustering is strong. A result that appears only at 0.40
is to be reported as exactly that.

**Secondary leakage audit: structural similarity.** Structures exist for all 208 domains, so a
Foldseek or TM-align check of the final split is cheap and catches what sequence identity
misses. The threshold for it needs its own justification, because TM-score is length-normalised
in a way that can make small structures look more similar than they are — the same trap as a
fixed identity floor, in a different coordinate.

## Consequence for what can be claimed

The clean subset, the only part where ThermoMPNN-D is a valid comparator, holds **40 clusters
under the primary clustering**. Paired at cluster level that needs **27 of 40 wins (68%)** for
significance, against the 56% ThermoMPNN-D itself achieves at domain level.

So a modest but consistent improvement is detectable, and a larger average gain carried by a
minority of families is not. That is a constraint on the experiment, and it is the reason route
B — a sharper objective, which should improve consistency — is preferred over simply adding
capacity, which would tend to improve the average.

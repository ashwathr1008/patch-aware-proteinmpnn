# Scope: small single-domain systems

Draft scope section, November 5, 2026. Supporting artifacts:
[size_trend.csv](../../Analysis/decomposition/size_trend.csv),
[size_trend.json](../../Analysis/decomposition/size_trend.json). Merge with the
[motivation section](motivation_paragraph.md) when assembling the report.

## The paragraph

> We scope this project to **small single-domain proteins**, in the range of roughly 20 to 100
> residues, scored as monomers on a fixed backbone. This is the regime where a surface patch
> is a large fraction of the total surface, where a designer has the most control over every
> position, and where developability failures are least likely to be rescued by the rest of
> the structure. It is also the regime the present dataset occupies: the seven structures run
> here span 20 to 76 residues and are all single chains. We state two things plainly about
> that. The present structures were selected for tractability rather than chosen to represent
> this regime, so the match is a starting point and not a designed sample; a scoped dataset
> would be assembled deliberately. And we tested whether the arrangement signal we measure
> depends on chain length, because scoping to small systems would be self-defeating if it did.
> Across seven structures, no relationship between length and the joint arrangement term
> reaches significance under an exact permutation test, and the three reasonable
> normalizations do not agree in sign. We therefore report no size dependence either way, and
> treat the variation we observe as protein-specific rather than size-specific.

## The size test, in full

The question is fair and had to be answered before fixing the scope: if the joint arrangement
term grew with chain length, then scoping to small systems would be scoping away from the
effect the project depends on.

Significance uses an **exact permutation test** over all 5,040 orderings of seven structures,
rather than a normal approximation, which is not trustworthy at this sample size.

| Measure | Vanilla r (exact p) | Soluble r (exact p) |
| --- | --- | --- |
| Joint term, total | +0.497 (p = 0.259) | +0.615 (p = 0.138) |
| Joint term, per residue | −0.325 (p = 0.469) | −0.345 (p = 0.449) |
| First-order share of the gap | −0.326 (p = 0.474) | −0.499 (p = 0.256) |
| Full gap, total | +0.020 (p = 0.962) | +0.080 (p = 0.851) |

**Nothing reaches significance**, and the normalizations point different ways. The total joint
term rises with length, which is close to arithmetic: a longer chain holds more terms. Per
residue it falls, which would favour small systems. The first-order share also falls, meaning
longer chains are proportionally *more* arrangement-driven, which would favour large ones. The
two informative normalizations disagree, so the honest conclusion is that seven structures
cannot answer the question.

### Per-structure values

| PDB | Residues | Joint total | Joint per residue | First-order share |
| --- | ---: | ---: | ---: | ---: |
| 1L2Y | 20 | 1.04 | 0.0520 | 0.915 |
| 1VII | 36 | 0.56 | 0.0155 | 0.885 |
| 1CRN | 46 | 2.02 | 0.0440 | 0.904 |
| 1ENH | 54 | 0.93 | 0.0173 | 0.852 |
| 1SHG | 57 | 2.10 | 0.0368 | 0.879 |
| 2CI2 | 65 | 3.07 | 0.0473 | 0.821 |
| 1UBQ | 76 | 1.42 | 0.0187 | 0.916 |

Vanilla checkpoint, nats per chain. The ordering is not monotone in either direction. The
largest protein, 1UBQ, has the *highest* first-order share of all seven, which is the clearest
single reason not to read a size trend into these numbers.

## The caution that belongs with this scope

One observation should be recorded rather than left for a reviewer to find. **1L2Y is the only
designed miniprotein in the set, and it is among the most burial-dominated**, with a
first-order share of 0.915. If the eventual target is small *designed* systems such as
miniprotein binders, then the closest analogue available here shows proportionally less
arrangement structure, not more.

This rests on five pairs from a single patch on a 20-residue chain, so it is weak evidence and
is not a reason to abandon the scope. It is a reason to include several designed miniproteins
in the final dataset rather than only natural domains, so the target regime is represented by
more than one structure.

## What this scope does and does not commit us to

**Does:** single-chain monomers; fixed backbones; patches defined on surface neighbourhoods;
evaluation on proteins in the 20–100 residue range, selected deliberately rather than for
convenience; several designed sequences alongside natural domains.

**Does not:** complexes, interfaces or binding partners. Nothing in the present dataset
involves an interface, and the project's longer-term motivation points toward designed
binders. Closing that gap requires an interface-aware patch definition and a dataset that
contains interfaces; both are out of scope for the mid-report and belong in the final
project's plan if the direction is pursued.

Reproduce the size test with:

```bash
python Analysis/summarize_size_trend.py \
  --scores Analysis/decomposition/first_order_scores.csv \
  --output Results/my-size-trend
```

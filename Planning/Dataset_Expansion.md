# Can this dataset be made bigger, and is it worth it?

Written November 8, 2026, after the homology clustering showed that effective evidence, not
contrast count, is the binding constraint: 69,633 resolved contrasts on 208 domains reduce to
84 clusters and a Kish effective sample size of 15.1.

So the question is not "is there more data" but "is there more *independent* data".

## There is more data

The 208 domains came from `dG_for_double_mutants_Fig4.csv` in the December 2022 preprint
release — a figure table, not the curated dataset. The final release
([Zenodo 7844779](https://zenodo.org/records/7844779), `v2_230420`) indexes its double-mutant
experiments in `Processed_K50_dG_datasets/Double_DMS_list.csv`:

| | Domains | Double-mutant experiments |
| --- | ---: | ---: |
| Ours, from the v1 figure table | 208 | 578 site pairs |
| Final release `Double_DMS_list.csv` | **267** | 725 (310 with an explicit site pair, 415 networks) |

**59 additional domains, and none of ours are absent from it.** Structures exist for all of
them: the final release's `AlphaFold_model_PDBs.zip` holds 1,724 models, and all 267 were found.

Both the index and the structures were obtained without downloading the 1 GB archive, by
fetching the zip's central directory with HTTP range requests and extracting only the 90 KB
index.

## But it adds very little independence

Clustered with the declared HSSP length-dependent threshold:

| Set | Domains | Clusters | Singletons | Largest |
| --- | ---: | ---: | ---: | ---: |
| ours | 208 | 84 | 58 | 35 |
| final double list | 267 | **94** | 75 | **63** |

**59 domains (+28%) buy 10 clusters (+12%).** For the paired sign test the required win rate is
essentially unchanged: 52 of 84 (62%) becomes about 58 of 94 (62%).

And the largest cluster grows from 35 to 63. Some of the new domains do not form new families,
they **bridge existing ones**, merging clusters that were separate in the 208-domain set. The
net +10 is 17 new singletons minus the merging of families that were previously distinct.

That bridging is worth understanding rather than just noting. It does not invalidate the
84-cluster split: clustering the domains actually used is the correct procedure, because
leakage is only possible within the data in the experiment. But it does say the sequence space
is more connected than our subset reveals, so the 84 figure should be read as a property of
this subset rather than as a count of genuinely distinct families in nature.

## The cost

Using the extra 59 domains requires more than listing them:

- Their measured double-mutant values come from `Tsuboyama2023_Dataset2_Dataset3_20230416.csv`,
  718 MB inside the 1 GB archive, with the duplicate-key and quality reconciliation from the
  mid-report's Section 3.3 still outstanding on that table.
- Frozen ProteinMPNN must be re-scored on them to extend the baseline, and that was the GPU
  job. Bridges-2 is in a four-to-six week migration outage, so it would run on CPU.
- Every downstream artefact keyed to 208 domains would be regenerated: the swap contrasts, the
  baseline, the ThermoMPNN comparison, the split.

## Recommendation: not now

The binding constraint is not how many clusters exist but whether the effect is large and
consistent enough to detect in any of them. Going from 84 to 94 clusters changes the required
win rate by under a percentage point, while the work is a full dataset regeneration without a
GPU. Route B — training on the interaction term rather than total stability — addresses
consistency, which is what the power analysis says actually limits the claim.

**Revisit if** the first trained head lands close to significance on the current split, where
10 more clusters could decide the outcome. The index and structures are already retrieved, so
the remaining cost is the measured-value extraction and the re-scoring.

## One expansion route that does not fit

The **MGnify Stability Dataset** (Rocklin Lab with the OpenFold Consortium, announced
May 26 2026) reports folding stability for 1.8 million domains across more than 200,000
sequence families, restricted to 60–80 residues — which would solve the homology-diversity
problem outright. But the announcement describes wild-type domain stabilities and makes no
mention of double mutants, and without paired measurements of the same two residues in opposite
arrangements it cannot supply a single composition-matched contrast. It is the right size and
the wrong shape for this task. Worth checking directly if the released tables turn out to
include mutational scans.

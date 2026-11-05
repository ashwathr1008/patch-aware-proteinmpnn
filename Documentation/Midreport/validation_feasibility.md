# Measured-outcome validation: eSOL feasibility audit

Draft section, November 4, 2026. Satisfies the mid-report plan's *evaluation readiness*
deliverable: one experimental-solubility dataset audited, with its label definition, access,
identifiers, structure coverage, exclusions and a planned split. Artifacts in
[Analysis/esol_audit/](../../Analysis/esol_audit/). Reproduce with
[`Analysis/audit_esol.py`](../../Analysis/audit_esol.py).

## Verdict

eSOL is **usable for the final project but not for the current pilot**. It yields roughly 1,300
E. coli proteins carrying both an experimental solubility measurement and a solved structure,
which is ample. It cannot label the seven proteins scored so far, because its overlap with them
is zero. Committing to eSOL therefore means assembling a new E. coli structure set, not
attaching labels to existing results. No measured-outcome claim is made in this report.

## What the label actually is

Niwa et al. (PNAS 2009) translated individual E. coli K-12 proteins in the PURE cell-free
system, which is chaperone-free, and measured the fraction of each product that remained
soluble. The label is a **whole-protein solubility percentage, for one organism, under one
in vitro condition**.

It is not a patch-level label, not an aggregation rate, and not a therapeutic formulation
measurement. A separate 788-protein subset reports solubility with chaperones added
(TF, GroEL/ES, DnaK/DnaJ/GrpE), which is a different quantity and is excluded here rather than
merged. Transfer from cell-free E. coli solubility to designed miniproteins or to formulation
behaviour is unestablished.

**Access and licence.** Public download, no registration:
`https://dbarchive.biosciencedbc.jp/data/esol/LATEST/esol.zip` (197,452 bytes, SHA-256
`29afdd2f…43d54b`). Licensed CC Attribution-Share Alike 2.1 Japan, with a **required**
attribution string: *"eSOL © Hideki Taguchi (Tokyo Institute of Technology) licensed under CC
Attribution-Share Alike 2.1 Japan."* Share-alike applies to redistributed derivatives, which
affects what we can publish alongside the final report.

## Identifiers and mapping coverage

eSOL carries B numbers and JW IDs. UniProt holds both as ordered locus names together with PDB
cross-references, so one join resolves identity and structure availability at once.

| Stage | Records | Share |
| --- | ---: | ---: |
| eSOL records | 4,132 | |
| With a solubility measurement | 3,173 | 76.8% of records |
| Mapped to a reviewed UniProt entry | 3,139 | **98.9% of measured** |
| With at least one PDB entry | 1,453 | 45.8% of measured |
| **Usable after exclusions** | **1,328** | 41.8% of measured |

Mapping is essentially clean: 34 unmapped and **zero ambiguous** mappings. The identifier
problem that usually sinks this kind of audit does not arise here.

### Exclusions applied

| Reason | Records |
| --- | ---: |
| No solubility measurement | 959 |
| No UniProt entry for either identifier | 34 |
| Solubility above 100% | 226 |
| No experimental structure in UniProt | 1,585 |

The 226 records above 100% are a normalization artefact of the assay readout; values run as
high as 147%. They are **excluded rather than clipped**, so the decision is visible in the
counts instead of silently compressing the soluble end of the scale.

## The finding that constrains how eSOL can be used

**Structure availability is confounded with solubility.** Crystallography and cryo-EM succeed
more often on soluble proteins, so requiring a structure removes part of the aggregation-prone
range we most want to study.

| Subset | n | Median solubility | Below 30% | Above 70% |
| --- | ---: | ---: | ---: | ---: |
| With structure | 1,453 | **56.0%** | 29% | 39% |
| Without structure | 1,686 | **33.0%** | 46% | 25% |

A 23-point median shift. The usable set is not a random sample of E. coli proteins, and any
performance figure computed on it inherits that bias. See
[esol_coverage.pdf](../../Analysis/esol_audit/esol_coverage.pdf). This argues for reporting
stratified by solubility band rather than one pooled correlation, and against claiming the
evaluation covers aggregation-prone proteins proportionately.

## Overlap with the current pilot: zero

eSOL covers E. coli K-12 only. Every structure in the current 85-pair set is from another
source organism or is a designed sequence — ubiquitin (human), crambin (*Crambe abyssinica*),
villin (chicken), Trp-cage (designed), engrailed (*Drosophila*), spectrin SH3 (chicken) and
CI-2 (barley). The overlap is zero by construction, not by chance.

This is the practical consequence: **eSOL cannot validate the pilot results.** It can support a
separate evaluation on a new E. coli set in the final project. The mid-report should say that
aggregation efficacy is untested and name this as the next mapping task, which is exactly the
outcome the plan anticipates.

## Planned split — specified, not executed

A homology-aware split requires clustering the usable sequences at a fixed identity threshold
and assigning **whole clusters** to development and held-out groups. Two cautions:

- Ordered-locus proximity and gene-name similarity do not imply homology and must not be used
  as the grouping key. Neighbouring b-numbers are often an operon, not a family.
- Sequences are not present in either joined file and must be retrieved before clustering.

Not executing the clustering here is deliberate: the audit's job is to establish whether the
data supports a split, and it does. Committing to particular groups before the structure
filter below is applied would fix the split on a set that is about to change.

## Remaining work before eSOL could be used

1. **Filter structures properly.** PDB cross-references are counted per protein, not checked
   for covering the full chain, resolution, or mutation/construct status. The usable count of
   1,328 is an upper bound; a real structure filter will reduce it.
2. **Retrieve sequences and cluster** at a fixed identity threshold, then fix the split.
3. **Define protein-level aggregation of patch scores** without tuning on held-out labels.
4. Decide whether to model solubility as continuous or thresholded; the distribution is
   strongly bimodal (38% below 30%, 31% above 70%), so a threshold is defensible but the cut
   must be fixed before any evaluation.

## Why not the Jain panel

Jain et al. (PNAS 2017) report a biophysical developability panel across clinical-stage
antibodies, and it is the dataset SAP itself is most associated with. It is **antibody-specific**:
the measurements are on IgGs, and the assays, concentrations and formulation conditions are
tied to that format. Our pilot is small single-domain monomers. Using it would require changing
the project's subject matter, not just its labels. It remains the right reference if the final
project moves to antibody variable domains; it is not an option for the current scope.

## One-paragraph version for the report body

> We audited eSOL (Niwa et al., PNAS 2009), which reports chaperone-free cell-free solubility
> for E. coli K-12 proteins, as a candidate measured outcome. Of 4,132 records, 3,173 carry a
> solubility value, 3,139 (98.9%) map cleanly to reviewed UniProt entries via B number or JW
> ID with no ambiguous mappings, and 1,453 have at least one experimental structure; 1,328
> remain after excluding 226 records whose reported solubility exceeds 100%. The labels are
> whole-protein, single-condition and in vitro, not patch-level or formulation measurements.
> Two findings constrain their use. Structure availability is confounded with the label —
> proteins with solved structures have a median solubility of 56% against 33% for those
> without — so the usable subset under-represents the aggregation-prone range and results
> should be reported stratified rather than pooled. And because eSOL covers E. coli alone,
> its overlap with our seven pilot structures is zero, so it cannot validate the present
> experiment and would require assembling a new structure set. We therefore report that
> aggregation efficacy is untested, and specify structure filtering, sequence retrieval and
> identity-based clustering as the next mapping tasks.

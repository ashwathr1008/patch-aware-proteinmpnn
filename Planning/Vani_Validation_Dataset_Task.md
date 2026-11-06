# Validation-dataset investigation: reason and deliverables

Date: November 6, 2026. Task associated with Vani's provisional validation-dataset responsibility; this document does not reassign remaining team work.

## Why this task matters

The project asks whether spatial surface-patch arrangement adds useful information
beyond amino-acid composition and frozen ProteinMPNN, with eventual benefit for
developability while preserving structural and functional compatibility.

The original/permuted pilot tests model sensitivity to arrangement on a fixed
backbone. It does not establish that a preferred arrangement is more soluble.
The subsequent decomposition attributes most of the gap to the model's backbone-only,
position-specific preferences, with a smaller sequence-context residual. Those
preferences are not independently measured burial, and the residual is not a pure
coupling estimate. Independent experimental outcomes are needed to test usefulness.

The proposed next hypothesis is: do predefined patch features predict measured
whole-protein solubility, and add predictive value beyond composition and frozen
ProteinMPNN? An eSOL audit establishes whether enough trustworthy label–sequence–
structure matches exist to attempt that experiment. It is a feasibility study,
not a completed validation result. eSOL cannot establish interface suitability,
retained binding, or patch-specific aggregation risk.

## Deliverables

| Deliverable | Required contents |
| --- | --- |
| Source and endpoint audit | Official download/documentation links, acquisition provenance and hashes, license/attribution, label definition, experimental conditions and unresolved method details. |
| Data-quality results | Actual record counts, missing/invalid labels, duplicate identifiers, unusual values and a documented handling policy. Preserve source values. |
| Mapping feasibility | Available identifiers, candidate PDBs, checked reference-sequence/chain mappings, coverage, backbone eligibility, exclusions and explicit pending cases. Candidate IDs are not verified mappings. |
| Evaluation text | Predefine protein-level aggregation of patch features, composition and frozen-model baselines, protein/family separation, evaluation metrics and development-only choices. A proposed primary metric is held-out Spearman correlation with measured solubility; report uncertainty by protein/family rather than treating patches as independent. |
| Limitations and recommendation | Assess usable coverage, whole-protein versus patch labels, assay transfer, selection bias from structure availability, possible pretraining overlap, and whether further evaluation is feasible. |

Do not mix the 2009 `Solubility (%)` endpoint with the separately remeasured
`Minus Sol (%)` or chaperone-supplemented columns. The archive describes a
chaperone-free PURE cell-free assay. Temperature, incubation duration,
centrifugation settings and uncertainty must still be verified from the original
methods. Numeric labels above 100 are flagged, not automatically clipped or excluded.

## Files and local reproduction

- Download: `Data/raw/esol/esol.zip` (local; existing Git rules ignore ZIPs/raw data).
- Acquisition record: `Data/esol_source.json`.
- Local analysis/mapping implementation: `Code/validation/audit_esol.py`.
- Results: `Analysis/vaidation_data/` (requested spelling).
- No README edits, PSC submissions, GPU inference or model fitting are required.

Run from the repository root using an available local Python 3 interpreter:

```powershell
py -3.11 Code/validation/audit_esol.py
py -3.11 -m unittest discover -s Code/validation -p 'test_*.py' -v
```

For subsequent offline sequence mapping, first stage reference sequences and PDBs
locally. FASTA headers must start with the exact eSOL JW_ID. Then run:

```powershell
py -3.11 Code/validation/audit_esol.py --reference-fasta Data/raw/esol/reference_sequences.fasta --pdb-dir Data/raw/esol/structures --output Analysis/vaidation_data/mapped
```

The mapper reuses the baseline's complete canonical N/CA/C/O backbone stager,
records input hashes, and accepts only unambiguous exact contiguous sequence matches.
Partial matches remain partial; mutated chains and internal deletions require
additional alignment work. Passing this screen does not establish construct
equivalence or patch eligibility. Family grouping and final exclusions remain future work.

## Current feasibility finding

The downloaded archive has 4,132 records and 3,173 numeric primary labels; 358
labeled records have parsed candidate PDB IDs. There are 229 labels above 100%.
The current local Data directory contains none of the candidate PDBs and no
identifier-matched eSOL reference FASTA, so usable sequence/structure coverage is
not yet established. This is a promising candidate pool, not 358 validated proteins.

Sources: https://dbarchive.biosciencedbc.jp/en/esol/download.html and
https://dbarchive.biosciencedbc.jp/data/esol/LATEST/README_e.html.
The latter links Niwa et al., PNAS 2009 and the separate 2012 chaperone study.

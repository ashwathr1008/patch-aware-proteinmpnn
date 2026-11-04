# Multipatch composition-matched dataset

This builder expands the checked single-pair example into a versioned, auditable set of arrangement controls. It does not score sequences, train a model, reconstruct side chains or supply aggregation labels.

Run from the repository root with a predeclared structure manifest:

```powershell
python Code/patches/build_dataset.py --manifest Data/compatibility_manifest.json --output Results/patch-pairs-20261104 --patches-per-structure 3 --permutations-per-patch 5 --seed 7
python -m unittest discover -s Code -p test_build_dataset.py -v
```

The output directory must be new. The input JSON requires schema version 1, purpose `development_only` or `patch_compatibility_pilot`, and a `structures` list containing `structure_id`, `chain`, `pdb_path` (relative to its manifest), `sha256`, `group` and `split`. Optional `expected_length` is checked. Group/split assignments must be frozen before running the builder; it does not create homology clusters or prove pretraining exclusion.

## Frozen selection and permutation rules

The builder first uses the same complete canonical N/CA/C/O stager as the baseline. It preserves identifiers and rejects unsupported gaps, insertion codes and incomplete mappings. Noncanonical ATOM residues and incomplete backbones are excluded at the structure level; excluded inputs and all shortfalls remain in the summary. HETATM records and side chains are not used as features.

Count other CAs within 10 Å and consider centers in the bottom quartile, including ties. Sort centers by count, then observed sequence index. A patch includes CAs within 8 Å of its center; require at least eight residues and two identities. Heterogeneity is a feasibility filter for a nonidentity permutation, not a score. Skip duplicate memberships and greedily retain up to three neighborhoods whose Jaccard overlap with every earlier retained neighborhood is at most 0.5. Selection never uses a model score. If the rules yield fewer patches, report that shortfall without changing thresholds.

For each selected patch, derive a deterministic random seed from the base seed, structure/chain ID and membership. Generate five distinct nonidentity assignments by shuffling source indices. Reject repeated assignments, with a fixed cap of 10,000 attempts. A patch with too few available assignments records a permutation shortfall. Every pair retains full-chain original/current sequences and source-index provenance. Geometry and patch membership are stored once and shared exactly by both members. Overlapping patches and multiple permutations remain dependent observations within their source protein.

## Controls and descriptor

The exact twenty-amino-acid patch count vectors must match. Whole-chain counts, length and outside-patch identities must also remain unchanged. Geometry-only controls tie because both assignments share the same coordinates and contact graph.

The simple descriptor is the fraction of eligible patch contacts whose two current identities belong to the fixed set `AVILMFWY`. Eligible contacts have CA distance at most 8 Å, both endpoints in the patch, and observed sequence-index separation greater than one. Divide by the number of all eligible contacts, which stays fixed within a pair. Zero-contact patches have null fractions/deltas, not zero-valued measurements. Store original/permuted fractions and **permuted minus original** change. No direction is labeled favorable. This categorical descriptor is not SAP, a solvent-area calculation, or a validated aggregation predictor.

## Outputs

- `pairs.json`: scoring contract (`purpose: patch_compatibility_pilot`), relative staged-PDB paths/hashes, frozen configuration, source group/split metadata and complete pair sequences. `patch_indices` are **one-based observed sequence indices**, not PDB residue numbers. Pair `controls` includes count vectors, `composition_tie`, `geometry_tie`, `contact_count`, `hydrophobic_contacts_original/permuted`, `hydrophobic_fraction_original/permuted` and `hydrophobic_fraction_delta`.
- `pair_manifest.csv`: compact pair counts, invariants, patch contiguity and descriptor values.
- `patches.json`: original PDB-to-sequence mapping, all retained backbone coordinates, patch membership, center IDs, within-patch CA pair distances, descriptor contacts and permutation attempt counts.
- `inputs/`: exact staged backbone PDBs that pair scoring must use.
- `metadata/config.json` and `summary.json`: source/script hashes, parameters, actual/requested counts, exclusions, shortfalls and completion status. Failures retain their metadata.

Each pair has a `checks` dictionary of seven booleans and `controls` with composition/geometry ties. `identity_source_indices` lists the original one-based source positions in target patch order. `patch_contiguous_in_sequence` distinguishes a spatial neighborhood that happens to be contiguous from a genuinely discontinuous sequence neighborhood.

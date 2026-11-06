# Data and manifests

[`development_manifest.json`](development_manifest.json) records three small public inputs: 1UBQ chain A (76 residues), 1VII chain A (36 deposited residues, PDB numbering 41–76), and 1CRN chain A (46 residues). They were selected for pipeline development, without a held-out split or formal homology clustering. Crambin's three disulfides are not constrained during redesign. None supplies an experimental solubility/aggregation label.

After cloning, run `python Code/prepare_development_data.py --data-dir Data --fetch-only` from the repository root to fetch missing files and verify their hashes. Without `--fetch-only`, the script creates a new development manifest only when one does not already exist.

[`compatibility_manifest.json`](compatibility_manifest.json) freezes seven sources before patch scoring. Three additional diagnostic groups are Trp-cage 1L2Y, engrailed homeodomain 1ENH and spectrin SH3 1SHG. The predeclared heuristic similarity screen places 2CI2 with 1UBQ in a conservative development group; this is not a claim of established homology. All 21 pairwise screens, grouping rules, chain IDs, observed lengths, input hashes and caveats are recorded. The panel has six groups and no claim of exclusion from ProteinMPNN training.

[`compatibility_pairs/`](compatibility_pairs/) contains the small curated input to the expanded experiment: 85 pairs, 17 patches, seven staged backbones, composition/geometry checks, descriptor contacts and source mappings. Eight selected patches contain noncontiguous sequence positions. 1L2Y and 1ENH yielded one patch each under the frozen rules; others yielded three. Requested patch/pair shortfalls are preserved in `metadata/summary.json`. No raw large dataset or model weights are included.

Verify raw inputs with `python Code/prepare_compatibility_inputs.py --data-dir Data --fetch-only`; regenerate the pair dataset into a new directory using [the builder commands](../Code/patches/DATASET.md). `first_pair/` preserves the exact original demonstration as a separate one-pair scoring input.

Track small manifests here: structure source/identifier, chain and residue mapping, homology/complex group, split, patch identifiers, preprocessing configuration, and input hashes. Split by source protein/complex and close homologues before deriving patches or variants.

Keep large downloaded and processed datasets in ignored `raw/`, `processed/`, or `cache/` subdirectories. Record how to obtain or regenerate them. A local software smoke-test input is not automatically part of the biological evaluation set.

## RCSB structure metadata (for the eSOL candidate filter)

Not committed; ~11 MB, fetched per run. The candidate PDB identifiers come from the eSOL
audit's `esol_mapped.csv`; metadata is fetched from the RCSB GraphQL API in batches of 100:

```
POST https://data.rcsb.org/graphql
query($ids:[String!]!){entries(entry_ids:$ids){rcsb_id exptl{method}
  rcsb_entry_info{resolution_combined}
  polymer_entities{entity_poly{rcsb_sample_sequence_length}
    rcsb_polymer_entity{pdbx_mutation}
    rcsb_polymer_entity_container_identifiers{auth_asym_ids
      reference_sequence_identifiers{database_accession database_name}}}}}
```

2,103 unique entries cover the 185 candidates in the 20–150 residue cohort. RCSB data is
released under CC0. See [the filter](../Analysis/filter_structures.py) and
[its results](../Analysis/structure_filter/).

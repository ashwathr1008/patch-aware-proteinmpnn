# Data and manifests

[`development_manifest.json`](development_manifest.json) records three small public inputs: 1UBQ chain A (76 residues), 1VII chain A (36 deposited residues, PDB numbering 41–76), and 1CRN chain A (46 residues). They were selected for pipeline development, without a held-out split or formal homology clustering. Crambin's three disulfides are not constrained during redesign. None supplies an experimental solubility/aggregation label.

After cloning, run `python Code/prepare_development_data.py --data-dir Data --fetch-only` from the repository root to fetch missing files and verify their hashes. Without `--fetch-only`, the script creates a new development manifest only when one does not already exist.

[`compatibility_manifest.json`](compatibility_manifest.json) freezes seven sources before patch scoring. Three additional diagnostic groups are Trp-cage 1L2Y, engrailed homeodomain 1ENH and spectrin SH3 1SHG. The predeclared heuristic similarity screen places 2CI2 with 1UBQ in a conservative development group; this is not a claim of established homology. All 21 pairwise screens, grouping rules, chain IDs, observed lengths, input hashes and caveats are recorded. The panel has six groups and no claim of exclusion from ProteinMPNN training.

[`compatibility_pairs/`](compatibility_pairs/) contains the small curated input to the expanded experiment: 85 pairs, 17 patches, seven staged backbones, composition/geometry checks, descriptor contacts and source mappings. Eight selected patches contain noncontiguous sequence positions. 1L2Y and 1ENH yielded one patch each under the frozen rules; others yielded three. Requested patch/pair shortfalls are preserved in `metadata/summary.json`. No raw large dataset or model weights are included.

Verify raw inputs with `python Code/prepare_compatibility_inputs.py --data-dir Data --fetch-only`; regenerate the pair dataset into a new directory using [the builder commands](../Code/patches/DATASET.md). `first_pair/` preserves the exact original demonstration as a separate one-pair scoring input.

Track small manifests here: structure source/identifier, chain and residue mapping, homology/complex group, split, patch identifiers, preprocessing configuration, and input hashes. Split by source protein/complex and close homologues before deriving patches or variants.

Keep large downloaded and processed datasets in ignored `raw/`, `processed/`, or `cache/` subdirectories. Record how to obtain or regenerate them. A local software smoke-test input is not automatically part of the biological evaluation set.

## eSOL (candidate measured-outcome dataset, audited November 4, 2026)

Not committed; downloaded per run. Public download, no registration:

```bash
curl -L -o esol.zip https://dbarchive.biosciencedbc.jp/data/esol/LATEST/esol.zip   # 197452 bytes
# sha256 29afdd2fa5be4947c9cbb1756197f4382e5184f8a2e4ae9162689ad74643d54b
unzip esol.zip                                                                    # esol.csv
curl -s --compressed 'https://rest.uniprot.org/uniprotkb/search?query=%28organism_id%3A83333%29%20AND%20%28reviewed%3Atrue%29&fields=accession%2Cgene_primary%2Cgene_oln%2Cxref_pdb&format=tsv&size=500'
# follow the Link rel="next" header to page through all entries
```

Licensed CC Attribution-Share Alike 2.1 Japan. Any redistribution must carry the attribution
string "eSOL (c) Hideki Taguchi (Tokyo Institute of Technology) licensed under CC
Attribution-Share Alike 2.1 Japan", and share-alike applies to derivatives. The UniProt stream
endpoint returned truncated responses during this audit; the paginated search endpoint with
compression worked.

See the [feasibility audit](../Documentation/Midreport/validation_feasibility.md) and
[artifacts](../Analysis/esol_audit/). eSOL overlap with the current pilot structures is zero:
it covers E. coli K-12 only.

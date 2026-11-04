# Data and manifests

[`development_manifest.json`](development_manifest.json) records three small public inputs: 1UBQ chain A (76 residues), 1VII chain A (36 deposited residues, PDB numbering 41–76), and 1CRN chain A (46 residues). They were selected for pipeline development, without a held-out split or formal homology clustering. Crambin's three disulfides are not constrained during redesign. None supplies an experimental solubility/aggregation label.

After cloning, run `python Code/prepare_development_data.py --data-dir Data --fetch-only` from the repository root to fetch missing files and verify their hashes. Without `--fetch-only`, the script creates a new development manifest only when one does not already exist.

Track small manifests here: structure source/identifier, chain and residue mapping, homology/complex group, split, patch identifiers, preprocessing configuration, and input hashes. Split by source protein/complex and close homologues before deriving patches or variants.

Keep large downloaded and processed datasets in ignored `raw/`, `processed/`, or `cache/` subdirectories. Record how to obtain or regenerate them. A local software smoke-test input is not automatically part of the biological evaluation set.

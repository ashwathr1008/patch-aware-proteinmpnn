# Illustrative patch permutation starter

This standard-library Python 3.9+ tool constructs **one development-only example** for the mid-report handoff. It does not train a scorer, estimate SASA, label aggregation, or establish that a permutation is biologically worse. No sidechain features are used.

From the `Project` directory:

```powershell
python Code/patches/starter.py --pdb Data/raw/1UBQ.pdb --chain A --output Results/patch-demo-1ubq-seed7 --seed 7 --min-patch-size 8
python Code/patches/visualize.py --manifest Results/patch-demo-1ubq-seed7/manifest.json --output Results/patch-demo-1ubq-seed7-figure
python -m unittest discover -s Code -p "test_*.py" -v
```

Use an existing local PDB; the script makes no network requests. Output directories must be new. Keep generated artifacts in ignored `Results/`, `Runs/`, or `Data/processed/` directories.

The optional visualization requires `matplotlib`; the patch generator itself uses only the standard library. `visualize.py` writes `patch_pair.png` and `patch_pair.pdf` into a new directory. It shows two 3D CA traces with identical bounds and camera, a gray full-chain trace, identity-colored patch positions, and labels containing the identity and PDB residue number. Both views use the same color map and one shared identity legend. Trace segments longer than 4.5 Å or missing CA atoms are not connected. The example above explicitly requires eight patch residues to avoid a tiny terminal patch in 1UBQ; the generator's general default remains four.

The parser selects the first PDB model and the requested single-character chain. It preserves standard amino acid ATOM residues in observed order, with original residue numbers and insertion codes. Coherent backbone altloc choices use complete N/CA/C/O coverage first, then summed occupancy, then blank/A/lexical label; blank atoms are shared by conformers. Duplicate atom records use occupancy, serial, then coordinates as deterministic tie breakers. HETATM records, including modified amino acids, are excluded. Residues missing any required backbone atom remain in the observed-chain sequence but are excluded from patch selection.

For each eligible residue, count other eligible CA atoms within 10 Å. Low counts are a crude surface-like proxy, **not solvent-accessible surface area**. Among the bottom quartile of counts (including ties), visit centers by increasing count and observed sequence index. Choose the first center with at least four residues and two identities in its 8 Å CA neighborhood. These are illustrative defaults, not calibrated biochemical thresholds. Optional `--neighbor-radius`, `--patch-radius`, and `--min-patch-size` flags make them explicit. The center is selected deterministically; the seed controls only the identity permutation.

The patch membership and all backbone coordinates stay fixed. Shuffle existing patch identities until their assignment changes, preserving the exact amino acid multiset. Residues outside the patch are unchanged. Some patch residues may retain their original identity. The original PDB is never modified, and no sidechain coordinates are generated for the permuted assignment.

Outputs:

| File | Content |
| --- | --- |
| `manifest.json` | Schema version, development-only interpretation, source path/SHA256, parameters, selection rules, mapping diagnostics, full observed-chain sequences, original/permuted patch compositions, shared-geometry hashes, seven invariant checks, and one record per observed standard residue. |
| `residues.csv` | Same per-residue assignments and mapping as JSON, with N/CA/C/O XYZ coordinates flattened into columns. Missing coordinates are blank. |
| `original.fasta` | Original observed standard-residue chain, including incomplete-backbone residues. |
| `permuted.fasta` | Same observed chain with only patch identities permuted. |

JSON `residues` fields are `observed_sequence_index` (one-based), `chain`, `resseq`, `insertion_code`, `pdb_residue_id`, `selected_altloc`, `complete_backbone`, `in_patch`, `original_aa`, `permuted_aa`, `permuted_identity_source_index`, `ca_neighbor_count`, and `backbone_coordinates` (`N`, `CA`, `C`, `O`, each XYZ or null). `permuted_identity_source_index` traces where the assigned identity came from; it does not change the target residue's location or numbering. CSV uses these same scalar fields plus `N_x` through `O_z`.

**Scoring handoff:** FASTA covers observed standard ATOM residues, not necessarily the full biological sequence. Numbering gaps, insertion codes, incomplete backbone residues, nonmonotonic numbering, and ignored nonstandard ATOM residues are explicitly reported in `mapping`. Unobserved terminal residues cannot be inferred from ATOM records. Before ProteinMPNN scoring, compare this index mapping and sequence length with the model parser's representation; some parsers introduce gap positions. Never assume FASTA index equals PDB residue number. Even when `requires_mapping_review` is false, this tool has not validated an external parser. Multi-chain contacts and excluded residues do not contribute to the current proxy.

Use the example to demonstrate reproducible bookkeeping and exact composition control. A learned compatibility claim needs separate protein-level splits, multiple independent proteins, baselines, and evaluation; this one pair is not a dataset or a test result.

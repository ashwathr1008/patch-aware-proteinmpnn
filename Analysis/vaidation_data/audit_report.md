# eSOL local feasibility audit

Source: https://dbarchive.biosciencedbc.jp/data/esol/LATEST/esol.zip
Documentation: https://dbarchive.biosciencedbc.jp/data/esol/LATEST/README_e.html
Attribution: eSOL © Hideki Taguchi (Tokyo Institute of Technology) licensed under CC Attribution-Share Alike 2.1 Japan
Derived eSOL tables are distributed under CC BY-SA 2.1 Japan.

## Observed archive results

- Total records: 4132
- Numeric primary solubility labels: 3173
- Label status counts: {'in_0_100': 2944, 'outside_0_100': 229, 'missing': 959}
- Records with parsed PDB IDs: 430
- Numeric labels with listed PDB IDs: 358
- Unique candidate PDB IDs: 393
- Nonempty structure annotations that could not be parsed: 2 (see records.csv; never guessed)
- Duplicate JW identifiers: {}
- Raw label minimum/median/maximum: {'min': 0.0, 'median': 43.0, 'max': 147.0}
- Mapping status counts: {'structure_not_available_locally': 430}

## Interpretation and remaining work

The archive provides candidate independent whole-protein solubility labels. It does
not provide patch-level aggregation labels or binding/interface endpoints. PDB
annotations alone do not establish a usable sequence/structure mapping. No absent
local file is counted as a biological exclusion. Numeric values outside 0–100 are
preserved and flagged; their treatment requires checking the assay/publication.
Two source structure annotations are scientific-notation strings (`1.00E+59`
and `1.00E+24`), not recoverable PDB identifiers under this parser. They remain
unresolved source-data issues rather than being converted into guessed IDs.

The primary endpoint is `Solubility (%)` from the 2009 study. `Minus Sol (%)`
is a separate remeasurement for the chaperone study, and is not substituted.
The archive documents a chaperone-free PURE cell-free system. Exact temperature,
duration, centrifugation settings and measurement uncertainty remain to be checked
against the original methods before treating the audit as complete.

Next obtain identifier-matched reference sequences and candidate structures, then
rerun with --reference-fasta and --pdb-dir. FASTA headers must begin with JW_ID.
The conservative mapper accepts exact contiguous matches only; mutated chains and
internal deletions need a subsequent alignment audit. Full-length exact matches
still require checking experimental construct, backbone numbering, patch eligibility,
protein/family grouping and overlap with model pretraining.

Before evaluation, freeze exclusions, label handling, protein-level patch aggregation,
baselines and protein/family splits. Compare patch features with composition and
frozen ProteinMPNN on held-out proteins. No model has been fitted, no correlation
with experimental labels has been computed, and no PSC resources were used.

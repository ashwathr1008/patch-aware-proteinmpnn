"""Audit the archived eSOL ZIP and map listed structures using local files only.

No network, GPU, model inference, or cluster access. Optional reference FASTA
headers must be exact eSOL JW_ID values. Exact contiguous sequence mapping is
deliberately conservative; mutations/internal deletions need later alignment.
"""
import argparse
import csv
import hashlib
import io
import json
import math
import re
import statistics
import sys
import tempfile
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'baseline'))
from run_baseline import stage_backbone, read_fasta

SOURCE = 'https://dbarchive.biosciencedbc.jp/data/esol/LATEST/esol.zip'
DOC = 'https://dbarchive.biosciencedbc.jp/data/esol/LATEST/README_e.html'
ATTRIBUTION = 'eSOL © Hideki Taguchi (Tokyo Institute of Technology) licensed under CC Attribution-Share Alike 2.1 Japan'


def numeric(value):
    if not value.strip():
        return None
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except ValueError:
        return None


def pdb_ids(value):
    return sorted(set(re.findall(r'(?<![A-Za-z0-9])[0-9][A-Za-z0-9]{3}(?![A-Za-z0-9])', value.upper())))


def exact_mapping(reference, observed):
    """Return only unambiguous, exact contiguous matches; never guess identity."""
    start = reference.find(observed)
    if start < 0:
        return {'mapping_status': 'no_exact_contiguous_match', 'reference_coverage': '', 'reference_start': ''}
    if reference.find(observed, start + 1) >= 0:
        return {'mapping_status': 'ambiguous_exact_match', 'reference_coverage': '', 'reference_start': ''}
    return {'mapping_status': 'exact_full_length' if reference == observed else 'exact_partial',
            'reference_coverage': len(observed) / len(reference), 'reference_start': start + 1}


def write_csv(path, rows, fields):
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--zip', type=Path, default=Path('Data/raw/esol/esol.zip'))
    parser.add_argument('--pdb-dir', type=Path, default=Path('Data'))
    parser.add_argument('--reference-fasta', type=Path)
    parser.add_argument('--output', type=Path, default=Path('Analysis/vaidation_data'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.zip) as archive:
        rows = list(csv.DictReader(io.StringIO(archive.read('esol.csv').decode('utf-8-sig'))))
    required = {'JW_ID', 'Solubility (%)', 'Structure (PDB) ID'}
    if not rows or not required <= rows[0].keys():
        raise ValueError('Unexpected eSOL schema')
    references = {}
    if args.reference_fasta:
        for header, sequence in read_fasta(args.reference_fasta):
            key = header.split()[0]
            if key in references:
                raise ValueError('Duplicate FASTA identifier: ' + key)
            sequence = sequence.upper()
            if not set(sequence) <= set('ACDEFGHIKLMNPQRSTVWY'):
                raise ValueError('Noncanonical reference sequence: ' + key)
            references[key] = sequence
    local = {}
    for path in sorted(args.pdb_dir.rglob('*.pdb')):
        match = re.match(r'([0-9][A-Za-z0-9]{3})(?:_|$)', path.stem)
        if match:
            local.setdefault(match[1].upper(), []).append(path)
    records, mappings, values = [], [], []
    for row in rows:
        value = numeric(row['Solubility (%)'])
        ids = pdb_ids(row['Structure (PDB) ID'])
        label_status = ('missing' if not row['Solubility (%)'].strip() else 'invalid') if value is None else ('outside_0_100' if not 0 <= value <= 100 else 'in_0_100')
        if value is not None:
            values.append(value)
        records.append({'JW_ID': row['JW_ID'], 'B_number': row.get('B number', ''),
                        'gene': row.get('Gene name K-12', ''), 'solubility_raw': row['Solubility (%)'],
                        'label_status': label_status, 'pdb_field_raw': row['Structure (PDB) ID'],
                        'pdb_ids': ';'.join(ids), 'localization': row.get('Cell location', '')})
        for pdb in ids:
            base = {'JW_ID': row['JW_ID'], 'pdb_id': pdb, 'solubility_raw': row['Solubility (%)'], 'label_status': label_status}
            paths = local.get(pdb, [])
            if not paths:
                mappings.append(dict(base, mapping_status='structure_not_available_locally'))
            for path in paths:
                chains = sorted({line[21:22] for line in path.read_text().splitlines() if line.startswith('ATOM  ')})
                for chain in chains:
                    result = dict(base, pdb_path=path.as_posix(), chain=chain,
                                  pdb_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                    try:
                        with tempfile.TemporaryDirectory() as tmp:
                            staged = stage_backbone(path, chain, Path(tmp) / 'chain.pdb')
                        observed = staged['native_sequence']
                        result['observed_length'] = len(observed)
                        result['observed_sequence'] = observed
                        if row['JW_ID'] in references:
                            result.update(exact_mapping(references[row['JW_ID']], observed))
                        else:
                            result['mapping_status'] = 'reference_sequence_missing'
                    except ValueError as error:
                        result.update(mapping_status='backbone_rejected', reason=str(error))
                    mappings.append(result)
    summary = {
        'generated_utc': datetime.now(timezone.utc).isoformat(), 'source_url': SOURCE,
        'documentation_url': DOC, 'attribution': ATTRIBUTION,
        'license': 'CC BY-SA 2.1 Japan (dataset and derived tables)',
        'zip_sha256': hashlib.sha256(args.zip.read_bytes()).hexdigest(),
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'reference_fasta_sha256': hashlib.sha256(args.reference_fasta.read_bytes()).hexdigest() if args.reference_fasta else None,
        'records': len(rows), 'numeric_primary_labels': len(values),
        'label_status_counts': dict(Counter(r['label_status'] for r in records)),
        'records_with_pdb_ids': sum(bool(r['pdb_ids']) for r in records),
        'numeric_labels_with_pdb_ids': sum(bool(r['pdb_ids']) and r['label_status'] in ('in_0_100', 'outside_0_100') for r in records),
        'unparsed_nonempty_pdb_fields': sum(bool(r['pdb_field_raw'].strip()) and not r['pdb_ids'] for r in records),
        'duplicate_JW_IDs': {k: v for k, v in Counter(r['JW_ID'] for r in rows).items() if v > 1},
        'solubility_summary_raw': {'min': min(values), 'median': statistics.median(values), 'max': max(values)},
        'unique_listed_pdb_ids': len({p for r in records for p in r['pdb_ids'].split(';') if p}),
        'local_reference_sequences': len(references),
        'mapping_status_counts': dict(Counter(r['mapping_status'] for r in mappings)),
        'mapping_unit': 'one row per record/PDB/local file/chain; absent local PDB has one pending row',
        'local_pdb_search_root': args.pdb_dir.as_posix(),
        'label_policy': 'Preserve raw numeric values including values above 100; flag for investigation, do not clip or merge chaperone measurements.',
        'mapping_policy': 'Exact unambiguous contiguous reference match only; partial matches are not whole-protein validation. Listed PDB IDs are candidates, not verified mappings.',
        'status': 'archive_audited; experimental_validation_not_run',
    }
    write_csv(args.output / 'records.csv', records, list(records[0]))
    write_csv(args.output / 'mapping_audit.csv', mappings, ['JW_ID', 'pdb_id', 'solubility_raw', 'label_status', 'pdb_path', 'pdb_sha256', 'chain', 'observed_length', 'observed_sequence', 'mapping_status', 'reference_coverage', 'reference_start', 'reason'])
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    report = f'''# eSOL local feasibility audit

Source: {SOURCE}
Documentation: {DOC}
Attribution: {ATTRIBUTION}
Derived eSOL tables are distributed under CC BY-SA 2.1 Japan.

## Observed archive results

- Total records: {summary['records']}
- Numeric primary solubility labels: {len(values)}
- Label status counts: {summary['label_status_counts']}
- Records with parsed PDB IDs: {summary['records_with_pdb_ids']}
- Numeric labels with listed PDB IDs: {summary['numeric_labels_with_pdb_ids']}
- Unique candidate PDB IDs: {summary['unique_listed_pdb_ids']}
- Nonempty structure annotations that could not be parsed: {summary['unparsed_nonempty_pdb_fields']} (see records.csv; never guessed)
- Duplicate JW identifiers: {summary['duplicate_JW_IDs']}
- Raw label minimum/median/maximum: {summary['solubility_summary_raw']}
- Mapping status counts: {summary['mapping_status_counts']}

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
'''
    (args.output / 'audit_report.md').write_text(report, encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()

"""Download pinned Tsuboyama 2023 stability data; no training or cluster use.

Verify Zenodo-published MD5 plus local SHA256; retain archives without executing
their notebooks. Inspect CSV schemas directly inside ZIPs to avoid large extracts.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
from urllib.request import Request, urlopen
import zipfile

RECORD = 'https://zenodo.org/api/records/7992926'
FILES = ('Processed_K50_dG_datasets.zip', 'AlphaFold_model_PDBs.zip', 'Pipeline_figure_model.zip')


def request(url):
    return urlopen(Request(url, headers={'User-Agent': 'patch-aware-proteinmpnn-data/1.0'}), timeout=120)


def hashes(path):
    md5, sha = hashlib.md5(), hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8*1024*1024), b''):
            md5.update(chunk)
            sha.update(chunk)
    return md5.hexdigest(), sha.hexdigest()


def acquire(entry, root):
    path = root / entry['key']
    expected = entry['checksum'].removeprefix('md5:')
    if not path.exists():
        partial = path.with_suffix(path.suffix + '.partial')
        total, last = 0, 0
        with request(entry['links']['self']) as response, partial.open('wb') as handle:
            for chunk in iter(lambda: response.read(1024*1024), b''):
                handle.write(chunk)
                total += len(chunk)
                if total-last >= 64*1024*1024:
                    print(f'{path.name}: {total/1024**2:.0f} MiB', flush=True)
                    last = total
        partial.rename(path)
    md5, sha = hashes(path)
    if md5 != expected or path.stat().st_size != entry['size']:
        raise ValueError('Checksum or size mismatch: ' + str(path))
    schemas = {}
    with zipfile.ZipFile(path) as archive:
        members = [{'name': i.filename, 'uncompressed_bytes': i.file_size} for i in archive.infolist()]
        for info in archive.infolist():
            if info.filename.lower().endswith('.csv') and not info.filename.startswith('__MACOSX/'):
                with archive.open(info) as handle:
                    reader = csv.reader(io.TextIOWrapper(handle, encoding='utf-8-sig'))
                    columns = next(reader)
                    sample = next(reader, [])
                    schemas[info.filename] = {'columns': columns, 'first_row': dict(zip(columns, sample))}
    print(path.name + ': verified', flush=True)
    return {'filename':path.name,'local_path':path.as_posix(),'source_url':entry['links']['self'],
            'bytes':path.stat().st_size,'zenodo_md5':expected,'sha256':sha,'members':members,'csv_schemas':schemas}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('Data/raw/megascale'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with request(RECORD) as response:
        metadata = json.load(response)
    (args.output/'zenodo_record.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')
    entries = {e['key']:e for e in metadata['files']}
    with ThreadPoolExecutor(max_workers=3) as pool:
        files = list(pool.map(lambda name:acquire(entries[name],args.output),FILES))
    manifest = {'acquired_utc':datetime.now(timezone.utc).isoformat(),'zenodo_record':7992926,
                'doi':'10.5281/zenodo.7992926','version':metadata['metadata'].get('version'),
                'license_as_published':metadata['metadata'].get('license'),
                'paper':'https://doi.org/10.1038/s41586-023-06328-6',
                'status':'downloaded_and_checksums_verified; no_training_or_quality_filtering_run',
                'files':files}
    Path('Data/megascale_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print('Saved Data/megascale_manifest.json',flush=True)


if __name__ == '__main__':
    main()

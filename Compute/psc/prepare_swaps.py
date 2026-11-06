"""Build a self-contained run directory for the swap-scoring job.

The job needs four things on the cluster: the scoring code, the pinned ProteinMPNN source and
weights, the domain structures, and the contrast table. This assembles them into one directory
with a manifest of checksums, so that what arrives on the cluster can be verified against what
left, and a failed run can be told apart from a corrupted transfer.

Nothing is downloaded here except the pinned upstream files, and those are checked against the
hashes the repository already records. The structures and the contrast table come from paths
you give it, because both are derived from datasets that are deliberately not in version
control.

The directory is laid out the way swaps.sbatch expects:

    run/
      code/score_swaps_full.py, run_baseline.py
      inputs/ProteinMPNN/...        pinned source and both checkpoints
      inputs/structures/*.pdb
      inputs/swap_contrasts.csv
      logs/
      swaps.sbatch
      manifest.json
"""
import argparse
import json
import shutil
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Code" / "baseline"))
import run_baseline as baseline

RAW = "https://raw.githubusercontent.com/dauparas/ProteinMPNN/{}/{}"


def fetch(relative, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = RAW.format(baseline.UPSTREAM_COMMIT, relative)
    request = urllib.request.Request(url, headers={"User-Agent": "patch-aware-prepare/1.0"})
    with urllib.request.urlopen(request, timeout=300) as response:
        destination.write_bytes(response.read())
    actual = baseline.sha256(destination)
    expected = baseline.UPSTREAM_HASHES[relative]
    if actual != expected:
        raise ValueError("Checksum mismatch for {}: expected {}, got {}".format(
            relative, expected, actual))
    return actual


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True,
                        help="new run directory; must not already exist")
    parser.add_argument("--contrasts", type=Path, required=True,
                        help="swap_contrasts.csv from extract_swap_contrasts.py")
    parser.add_argument("--structures", type=Path, required=True,
                        help="directory of domain PDB files")
    args = parser.parse_args()

    run = args.output.resolve()
    run.mkdir(parents=True, exist_ok=False)
    (run / "logs").mkdir()
    code = run / "code"
    code.mkdir()
    inputs = run / "inputs"
    (inputs / "structures").mkdir(parents=True)

    here = Path(__file__).resolve().parents[2]
    manifest = {"upstream_commit": baseline.UPSTREAM_COMMIT, "files": {}}

    for source in (here / "Code" / "scoring" / "score_swaps_full.py",
                   here / "Code" / "baseline" / "run_baseline.py"):
        target = code / source.name
        shutil.copy2(source, target)
        manifest["files"]["code/" + source.name] = baseline.sha256(target)

    shutil.copy2(here / "Compute" / "psc" / "swaps.sbatch", run / "swaps.sbatch")
    manifest["files"]["swaps.sbatch"] = baseline.sha256(run / "swaps.sbatch")

    print("fetching pinned ProteinMPNN source and weights")
    for relative in baseline.UPSTREAM_HASHES:
        digest = fetch(relative, inputs / "ProteinMPNN" / relative)
        manifest["files"]["inputs/ProteinMPNN/" + relative] = digest
        print("  verified %s" % relative)

    wanted = set()
    with args.contrasts.open(newline="", encoding="utf-8-sig") as handle:
        import csv
        for row in csv.DictReader(handle):
            wanted.add(row["domain"])
    copied, missing = 0, []
    for domain in sorted(wanted):
        source = args.structures / (domain + ".pdb")
        if not source.exists():
            missing.append(domain)
            continue
        target = inputs / "structures" / source.name
        shutil.copy2(source, target)
        manifest["files"]["inputs/structures/" + source.name] = baseline.sha256(target)
        copied += 1

    shutil.copy2(args.contrasts, inputs / "swap_contrasts.csv")
    manifest["files"]["inputs/swap_contrasts.csv"] = baseline.sha256(
        inputs / "swap_contrasts.csv")

    manifest["domains_requested"] = len(wanted)
    manifest["domains_copied"] = copied
    manifest["domains_missing"] = missing
    (run / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    total = sum(f.stat().st_size for f in run.rglob("*") if f.is_file())
    print()
    print("run directory : %s" % run)
    print("structures    : %d copied, %d missing" % (copied, len(missing)))
    if missing:
        print("  missing: %s" % ", ".join(missing[:10]))
    print("files         : %d, %.1f MB total" % (len(manifest["files"]), total / 1e6))
    print()
    print("Next: transfer the directory, then set TASK_PYTHON on the cluster before sbatch.")


if __name__ == "__main__":
    main()

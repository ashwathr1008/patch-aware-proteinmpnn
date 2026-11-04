"""Stage a self-contained, pinned ProteinMPNN engineering smoke test."""
import argparse
import hashlib
import json
import re
import shutil
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def download(url, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "patch-aware-course-project"})
    with urllib.request.urlopen(request, timeout=120) as response:
        with destination.open("xb") as output:
            shutil.copyfileobj(response, output)
    return {"url": url, "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
            "bytes": destination.stat().st_size}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--commit", help="Exact commit; otherwise resolve main once and record it")
    args = parser.parse_args()
    root = args.output.resolve()
    if root.exists():
        parser.error("Output must be new; partial preparations are retained for inspection")
    commit = args.commit
    if commit is None:
        request = urllib.request.Request(
            "https://api.github.com/repos/dauparas/ProteinMPNN/commits/main",
            headers={"User-Agent": "patch-aware-course-project"})
        with urllib.request.urlopen(request, timeout=60) as response:
            commit = json.load(response)["sha"]
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        parser.error("Expected a full Git commit")
    root.mkdir(parents=True)
    files = {}
    for name in ("protein_mpnn_run.py", "protein_mpnn_utils.py", "LICENSE", "README.md",
                 "vanilla_model_weights/v_48_020.pt", "soluble_model_weights/v_48_020.pt"):
        target = root / "vendor" / "ProteinMPNN" / name
        files[target.relative_to(root).as_posix()] = download(
            "https://raw.githubusercontent.com/dauparas/ProteinMPNN/{}/{}".format(commit, name), target)
    target = root / "inputs" / "1UBQ.pdb"
    files[target.relative_to(root).as_posix()] = download(
        "https://files.rcsb.org/download/1UBQ.pdb", target)
    if b"ATOM  " not in target.read_bytes():
        raise ValueError("Downloaded structure has no ATOM records")
    here = Path(__file__).resolve().parent
    (root / "code").mkdir()
    for name in ("run_smoke.py", "bundle.py"):
        shutil.copy2(here / name, root / "code" / name)
    shutil.copy2(here / "smoke.sbatch", root / "smoke.sbatch")
    for name in ("logs", "results", "metadata"):
        (root / name).mkdir()
    config = {"purpose": "Engineering smoke test only; not a biological evaluation dataset",
              "structure": "1UBQ", "chain": "A", "model": "v_48_020",
              "samples_per_model": 4, "seed": 37, "temperature": "0.1",
              "upstream_commit": commit, "created_utc": datetime.now(timezone.utc).isoformat(),
              "source_files": files}
    (root / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"directory": str(root), "upstream_commit": commit,
                      "download_bytes": sum(item["bytes"] for item in files.values())}, indent=2))


if __name__ == "__main__":
    main()

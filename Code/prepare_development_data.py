"""Download three small development structures and record their provenance."""
import argparse
import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

STRUCTURES = (
    ("1UBQ", "Human ubiquitin", 76, "ubiquitin", "X-ray; no measured solubility label."),
    ("1VII", "Chicken villin headpiece", 36, "villin_headpiece", "Single NMR minimized-average conformer; residues 41-76."),
    ("1CRN", "Crambin", 46, "crambin", "Three native disulfides are not constrained by this backbone-only baseline."),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("Data"))
    parser.add_argument("--fetch-only", action="store_true", help="Fetch/verify inputs against the existing committed manifest")
    args = parser.parse_args()
    data = args.data_dir.resolve()
    destination = data / "development_manifest.json"
    if args.fetch_only:
        manifest = json.loads(destination.read_text(encoding="utf-8"))
        if manifest.get("purpose") != "development_only" or manifest.get("schema_version") != 1:
            raise ValueError("Unsupported manifest")
        for record in manifest["structures"]:
            path = (data / record["pdb_path"]).resolve()
            if data not in path.parents:
                raise ValueError("Manifest input must be inside the data directory")
            if path.exists():
                payload = path.read_bytes()
            else:
                request = urllib.request.Request(record["source_url"], headers={"User-Agent": "patch-aware-course-project"})
                with urllib.request.urlopen(request, timeout=90) as response:
                    payload = response.read()
            if hashlib.sha256(payload).hexdigest() != record["sha256"]:
                raise ValueError("Input checksum mismatch: " + record["structure_id"])
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("xb") as output:
                    output.write(payload)
        print("Verified {} development inputs against the existing manifest".format(len(manifest["structures"])))
        return
    if destination.exists():
        parser.error("Manifest exists; use a new directory to preserve provenance")
    raw = data / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    manifest = {"schema_version": 1, "purpose": "development_only",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "limitations": ["Convenience set, not a representative or held-out benchmark.",
                         "Group labels identify different proteins; no homology clustering is claimed.",
                         "No aggregation, binding, folding or function measurements are supplied."],
        "structures": []}
    for identifier, name, length, group, note in STRUCTURES:
        path = raw / (identifier + ".pdb")
        url = "https://files.rcsb.org/download/" + identifier + ".pdb"
        if not path.exists():
            request = urllib.request.Request(url, headers={"User-Agent": "patch-aware-course-project"})
            with urllib.request.urlopen(request, timeout=90) as response:
                payload = response.read()
            if b"ATOM  " not in payload:
                raise ValueError("No ATOM records: " + identifier)
            with path.open("xb") as output:
                output.write(payload)
        manifest["structures"].append({"structure_id": identifier, "name": name, "chain": "A",
            "pdb_path": "raw/" + path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "expected_length": length, "group": group, "split": "development",
            "source_url": url, "reference_url": "https://www.rcsb.org/structure/" + identifier,
            "notes": note})
    destination.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("Prepared 3 development structures: " + str(destination))


if __name__ == "__main__":
    main()

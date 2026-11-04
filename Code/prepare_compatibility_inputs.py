"""Prepare a fixed small compatibility pilot and a descriptive similarity screen."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import urllib.request
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent / "baseline"))
from run_baseline import stage_backbone

SOURCES = [
    ("1UBQ", "A", "Human ubiquitin", "ubiquitin", "development", "X-ray; 76 observed residues."),
    ("1VII", "A", "Chicken villin headpiece", "villin_headpiece", "development", "NMR minimized-average conformer; 36 observed residues numbered 41-76."),
    ("1CRN", "A", "Crambin", "crambin", "development", "X-ray; three disulfides; permutations do not preserve cysteine pairing."),
    ("1L2Y", "A", "Designed Trp-cage TC5b", "trp_cage", "diagnostic", "First of 38 deposited NMR models; 20 residues; designed scaffold, not a natural native sequence."),
    ("1ENH", "A", "Drosophila engrailed homeodomain", "engrailed_homeodomain", "diagnostic", "X-ray; observed residues 3-56; unobserved termini are not reconstructed."),
    ("1SHG", "A", "Chicken spectrin SH3 domain", "spectrin_sh3", "diagnostic", "X-ray; observed residues 6-62; isolated domain."),
    ("2CI2", "I", "Barley chymotrypsin inhibitor 2", "barley_ci2", "diagnostic", "X-ray; observed residues 19-83; single inhibitor chain, no binding-partner evaluation."),
]


def sha256(payload):
    return hashlib.sha256(payload).hexdigest()


def alignment_screen(a, b):
    """Global alignment: match +2, mismatch -1, linear gap -2; diagonal tie first."""
    scores = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        scores[i][0] = -2 * i
    for j in range(len(b) + 1):
        scores[0][j] = -2 * j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            scores[i][j] = max(scores[i-1][j-1] + (2 if a[i-1] == b[j-1] else -1),
                               scores[i-1][j]-2, scores[i][j-1]-2)
    i, j, aligned, matches = len(a), len(b), 0, 0
    while i or j:
        if i and j and scores[i][j] == scores[i-1][j-1] + (2 if a[i-1] == b[j-1] else -1):
            aligned += 1
            matches += a[i-1] == b[j-1]
            i, j = i-1, j-1
        elif i and scores[i][j] == scores[i-1][j]-2:
            i -= 1
        else:
            j -= 1
    identity = matches / aligned if aligned else 0.0
    coverage_a, coverage_b = aligned / len(a), aligned / len(b)
    return {"identity_among_aligned_residues": identity, "coverage_a": coverage_a,
            "coverage_b": coverage_b, "matching_residues": matches, "aligned_residues": aligned,
            "flag_for_grouping": identity >= 0.30 and min(coverage_a, coverage_b) >= 0.80}


def fetch(path, url, expected=None):
    if path.exists():
        payload = path.read_bytes()
    else:
        with urllib.request.urlopen(url, timeout=60) as response:
            payload = response.read()
    if b"ATOM  " not in payload or (expected and sha256(payload) != expected):
        raise ValueError("Input format/checksum mismatch: " + path.name)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(payload)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("Data"))
    parser.add_argument("--fetch-only", action="store_true")
    args = parser.parse_args()
    data = args.data_dir.resolve()
    manifest_path = data / "compatibility_manifest.json"
    if args.fetch_only:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for row in manifest["structures"]:
            path = (data / row["pdb_path"]).resolve()
            if data not in path.parents:
                raise ValueError("Input must remain inside data directory")
            fetch(path, row["source_url"], row["sha256"])
        print("Verified {} inputs".format(len(manifest["structures"])))
        return
    if manifest_path.exists():
        parser.error("Manifest exists; use --fetch-only or a new data directory")
    records, excluded = [], []
    with tempfile.TemporaryDirectory() as temporary:
        for code, chain, name, family, split, note in SOURCES:
            path = data / "raw" / (code + ".pdb")
            url = "https://files.rcsb.org/download/" + code + ".pdb"
            payload = fetch(path, url)
            try:
                info = stage_backbone(path, chain, Path(temporary) / (code + ".pdb"))
                mapping = info["residue_mapping"]
                if any(r["insertion_code"] for r in mapping) or any(b["residue_number"] != a["residue_number"] + 1 for a, b in zip(mapping, mapping[1:])):
                    raise ValueError("Numbering gaps/insertions unsupported by pilot")
            except ValueError as error:
                excluded.append({"structure_id": code, "chain": chain, "reason": str(error)})
                continue
            records.append({"structure_id": code, "chain": chain, "name": name, "family": family,
                            "group": family, "split": split, "pdb_path": "raw/" + code + ".pdb",
                            "sha256": sha256(payload), "expected_length": info["length"],
                            "observed_sequence": info["native_sequence"], "first_residue_number": mapping[0]["residue_number"],
                            "last_residue_number": mapping[-1]["residue_number"], "selected_model": info["selected_model"],
                            "source_url": url, "reference_url": "https://www.rcsb.org/structure/" + code, "notes": note})
    screen = []
    parents = list(range(len(records)))
    def find(i):
        while parents[i] != i:
            i = parents[i]
        return i
    for i, a in enumerate(records):
        for j, b in enumerate(records[:i]):
            row = {"structure_a": a["structure_id"], "structure_b": b["structure_id"],
                   **alignment_screen(a["observed_sequence"], b["observed_sequence"])}
            screen.append(row)
            if row["flag_for_grouping"] or a["family"] == b["family"]:
                parents[find(i)] = find(j)
    for i, row in enumerate(records):
        members = [r for j, r in enumerate(records) if find(j) == find(i)]
        row["group"] = "+".join(sorted({r["family"] for r in members}))
        if any(r["split"] == "development" for r in members):
            row["split"] = "development"
    manifest = {"schema_version": 1, "purpose": "patch_compatibility_pilot",
                "created_utc": datetime.now(timezone.utc).isoformat(), "structures": records,
                "excluded": excluded, "similarity_screen": {"method": "Global alignment; match +2, mismatch -1, gap -2; diagonal tie priority",
                "group_rule": "identity >=0.30 among residue-residue aligned columns and coverage >=0.80 of both observed chains, or same annotated family; connected components; any development member makes group development",
                "limitations": "Small convenience sample; heuristic within-pilot screen is not comprehensive homology detection. Diagnostic does not establish absence from model pretraining.",
                "comparisons": screen},
                "limitations": ["No experimental aggregation or solubility labels.", "NMR first models, incomplete terminal coverage and isolated domains limit generalization.", "Original 1L2Y is a designed scaffold; use original-versus-permuted terminology."]}
    data.mkdir(parents=True, exist_ok=True)
    manifest_path.write_bytes((json.dumps(manifest, indent=2) + "\n").encode())
    print(json.dumps({"included": [{k:r[k] for k in ("structure_id","chain","expected_length","group","split")} for r in records], "excluded": excluded, "grouping_flags": sum(r["flag_for_grouping"] for r in screen)}))


if __name__ == "__main__":
    main()

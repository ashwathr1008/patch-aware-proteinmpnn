"""Derive a self-contained swap-only v3 without changing v2 labels or splits."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import shutil
import sqlite3
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baseline.run_baseline import sha256
from scorer.data import load_dataset


def select_swaps(manifest):
    """Filter by actual composition, including WT-to-swap reference_variant records."""
    pairs = [p for p in manifest["pairs"] if p["reference_sequence"] != p["variant_sequence"]
             and Counter(p["reference_sequence"]) == Counter(p["variant_sequence"])]
    active = {p["parent_id"] for p in pairs}
    result = dict(manifest)
    result["pairs"] = pairs
    result["parents"] = [p for p in manifest["parents"] if p["parent_id"] in active]
    return result


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run(args):
    source = args.source.resolve()
    output = args.output.resolve()
    if output.exists():
        raise ValueError("Use a fresh v3 output directory")
    manifest_path = source / "stability_pairs.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset = select_swaps(manifest)
    if len(dataset["pairs"]) != args.expected_count:
        raise ValueError(f"Expected {args.expected_count} swaps, found {len(dataset['pairs'])}")
    by_parent = {p["parent_id"]: p for p in dataset["parents"]}
    pair_counts = Counter(by_parent[p["parent_id"]]["split"] for p in dataset["pairs"])
    if set(pair_counts) != {"train", "validation", "test"}:
        raise ValueError("Swap-only subset must retain nonempty train/validation/test partitions")
    output.mkdir(parents=True)
    for parent in dataset["parents"]:
        original = (source / parent["pdb_path"]).resolve()
        if sha256(original) != parent["pdb_sha256"]:
            raise ValueError("Source backbone checksum mismatch")
        # Stage within v3 even if a custom input uses an absolute/parent-relative path.
        destination = output / "structures" / (sha256(original) + ".pdb")
        destination.parent.mkdir(exist_ok=True)
        if not destination.exists():
            shutil.copyfile(original, destination)
        parent["pdb_path"] = destination.relative_to(output).as_posix()
    source_hash = sha256(manifest_path)
    dataset["selection"] = {"version": "vani_training_v3", "source_manifest_sha256": source_hash,
                            "source_directory": source.name,
                            "rule": "reference != variant AND identical amino-acid counts",
                            "labels_and_pair_ids": "unchanged from v2", "split_assignment": "inherited unchanged from v2"}
    dataset["homology"] = dict(dataset["homology"], audit_source="stage2_splits.json")
    groups = defaultdict(list)
    for parent in dataset["parents"]:
        groups[parent["homology_group"]].append(parent["parent_id"])
    audit = {"method": dataset["homology"]["method"], "parameters": dataset["homology"]["parameters"],
             "limitations": dataset["homology"].get("limitations"), "groups": dict(groups),
             "parents": dataset["parents"], "assignment": "subset of frozen v2 splits; no reclustering/resplitting",
             "full_original_audit": "source_stage2_splits.json"}
    shutil.copyfile(source / "stage2_splits.json", output / "source_stage2_splits.json")
    shutil.copyfile(source / "source_checksums.json", output / "source_checksums.json")
    write_json(output / "stage2_splits.json", audit)
    # Retain source measurements and row provenance for both members of every kept pair.
    needed = {(p["parent_id"], p[field]) for p in dataset["pairs"]
              for field in ("reference_sequence", "variant_sequence")}
    with sqlite3.connect(source / "measurements.sqlite") as original_db, sqlite3.connect(output / "measurements.sqlite") as subset_db:
        schema = original_db.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='measurements'").fetchone()[0]
        subset_db.execute(schema)
        for parent, sequence in sorted(needed):
            row = original_db.execute("SELECT * FROM measurements WHERE parent=? AND sequence=?", (parent, sequence)).fetchone()
            if row is None:
                raise ValueError("Kept pair lacks original measured provenance")
            subset_db.execute("INSERT INTO measurements VALUES (?,?,?,?,?,?,?)", row)
    write_json(output / "stability_pairs.json", dataset)
    # Free the much larger mixed manifest before running the full alignment/leakage validator.
    original_count = len(manifest["pairs"])
    del manifest, dataset
    _, parents, pairs = load_dataset(output / "stability_pairs.json")
    if len(pairs) != args.expected_count or not all(p["is_swap"] for p in pairs):
        raise ValueError("Final loader validation did not confirm all expected swaps")
    parent_counts = Counter(p["split"] for p in parents.values())
    kinds = Counter(p["pair_kind"] for p in pairs)
    summary = {"status": "prepared_and_loader_validated_not_trained", "version": "vani_training_v3",
               "source_manifest_sha256": source_hash, "source_pairs": original_count,
               "total_pairs": len(pairs), "composition_preserving_pairs": len(pairs),
               "excluded_non_swaps": original_count - len(pairs), "pair_counts": dict(pair_counts),
               "pair_kinds": dict(kinds), "parents": len(parents), "parent_counts": dict(parent_counts),
               "groups": len(groups), "split_assignment": "inherited unchanged from v2",
               "manifest_sha256": sha256(output / "stability_pairs.json"), "measurement_records": len(needed)}
    write_json(output / "summary.json", summary)
    write_json(output / "stage3_manifest.json", summary)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.report, summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("Data/processed/vani_training_v2"))
    parser.add_argument("--output", type=Path, default=Path("Data/processed/vani_training_v3"))
    parser.add_argument("--expected-count", type=int, default=51898)
    parser.add_argument("--report", type=Path, default=Path("Analysis/scorer_training_data/v3_summary.json"))
    run(parser.parse_args())

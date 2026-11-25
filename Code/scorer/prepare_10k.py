"""Select 7000/1500/1500 existing v3 pairs without reassigning any parent/group."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import shutil
import sqlite3
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baseline.run_baseline import sha256
from scorer.data import load_dataset

COUNTS = {"train": 7000, "validation": 1500, "test": 1500}


def select_subset(manifest, counts=None, seed=137, require_swaps=True):
    counts = COUNTS if counts is None else counts
    parents = {p["parent_id"]: p for p in manifest["parents"]}
    assignments = {}
    for parent in parents.values():
        group, split = parent["homology_group"], parent["split"]
        if group in assignments and assignments[group] != split:
            raise ValueError("Source homology group crosses partitions")
        assignments[group] = split
    pools = defaultdict(list)
    for pair in manifest["pairs"]:
        if pair["reference_sequence"] == pair["variant_sequence"] or (require_swaps and Counter(pair["reference_sequence"]) != Counter(pair["variant_sequence"])):
            raise ValueError("Expected composition-preserving nonidentity v3 pairs only")
        pools[parents[pair["parent_id"]]["split"]].append(pair)
    rng = random.Random(seed)
    chosen = []
    for split, count in counts.items():
        if count < 1 or len(pools[split]) < count:
            raise ValueError("Insufficient source pairs for " + split)
        chosen.extend(rng.sample(pools[split], count))
    active = {p["parent_id"] for p in chosen}
    subset = dict(manifest)
    subset["pairs"] = chosen
    subset["parents"] = [dict(p) for p in manifest["parents"] if p["parent_id"] in active]
    return subset


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run(args):
    source, output = args.source.resolve(), args.output.resolve()
    if output.exists():
        raise ValueError("Use a fresh subset directory")
    manifest_path = source / "stability_pairs.json"
    # Validate the complete source before using its split assignments.
    raw, _, _ = load_dataset(manifest_path)
    mixed = getattr(args, "mixed", False)
    version = "vani_training_v2_mixed_10k" if mixed else "vani_training_v3_10k"
    subset = select_subset(raw, seed=args.seed, require_swaps=not mixed)
    source_hash = sha256(manifest_path)
    output.mkdir(parents=True)
    for parent in subset["parents"]:
        original = (source / parent["pdb_path"]).resolve()
        destination = output / parent["pdb_path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, destination)
    shutil.copyfile(source / "source_checksums.json", output / "source_checksums.json")
    original_audit = source / "source_stage2_splits.json"
    shutil.copyfile(original_audit if original_audit.exists() else source / "stage2_splits.json", output / "source_stage2_splits.json")
    shutil.copyfile(source / "stage2_splits.json", output / "source_partition_splits.json")
    subset["selection"] = {"version": version, "seed": args.seed,
                           "source_manifest_sha256": source_hash, "requested_pair_counts": COUNTS,
                           "method": "sample without replacement separately inside each existing partition",
                           "split_assignment": "unchanged from source; parents/homology groups never reassigned"}
    subset["homology"] = dict(subset["homology"], audit_source="stage2_splits.json")
    groups = defaultdict(list)
    for parent in subset["parents"]:
        groups[parent["homology_group"]].append(parent["parent_id"])
    write_json(output / "stage2_splits.json", {"method": subset["homology"]["method"],
               "parameters": subset["homology"]["parameters"], "groups": dict(groups), "parents": subset["parents"],
               "source_assignment_audit": "source_partition_splits.json", "full_original_audit": "source_stage2_splits.json",
               "limitations": subset["homology"].get("limitations")})
    needed = {(p["parent_id"], p[field]) for p in subset["pairs"] for field in ("reference_sequence", "variant_sequence")}
    with sqlite3.connect(source / "measurements.sqlite") as original, sqlite3.connect(output / "measurements.sqlite") as destination:
        schema = original.execute("SELECT sql FROM sqlite_master WHERE name='measurements' AND type='table'").fetchone()[0]
        destination.execute(schema)
        for parent, sequence in sorted(needed):
            row = original.execute("SELECT * FROM measurements WHERE parent=? AND sequence=?", (parent, sequence)).fetchone()
            if row is None:
                raise ValueError("Missing experimental provenance")
            destination.execute("INSERT INTO measurements VALUES (?,?,?,?,?,?,?)", row)
    write_json(output / "stability_pairs.json", subset)
    _, parents, pairs = load_dataset(output / "stability_pairs.json")
    actual = dict(Counter(p["split"] for p in pairs))
    if actual != COUNTS or (not mixed and not all(p["is_swap"] for p in pairs)):
        raise ValueError("Subset count/composition verification failed")
    original_parents = {p["parent_id"]: p for p in raw["parents"]}
    original_pairs = {p["pair_id"]: p for p in raw["pairs"]}
    for parent in subset["parents"]:
        if parent != original_parents[parent["parent_id"]]:
            raise ValueError("A source parent or split definition was modified")
    if any(pair != original_pairs[pair["pair_id"]] for pair in subset["pairs"]):
        raise ValueError("A source label or pair record was modified")
    summary = {"status": "prepared_and_loader_validated_not_trained", "version": version,
               "source_manifest_sha256": source_hash, "manifest_sha256": sha256(output / "stability_pairs.json"),
               "seed": args.seed, "total_pairs": len(pairs), "composition_preserving_pairs": sum(p["is_swap"] for p in pairs),
               "composition_preserving_by_split": dict(Counter(p["split"] for p in pairs if p["is_swap"])),
               "pair_kinds": dict(Counter(p["pair_kind"] for p in pairs)),
               "pair_counts": actual, "parents": len(parents), "parent_counts": dict(Counter(p["split"] for p in parents.values())),
               "groups": len(groups), "measurement_records": len(needed), "source_labels_and_records_unchanged": True,
               "source_parent_and_group_assignments_unchanged": True,
               "selection": "subsample inside existing partitions; no random pair-level repartitioning",
               "homology_limitations": subset["homology"].get("limitations")}
    write_json(output / "summary.json", summary)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.report, summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("Data/processed/vani_training_v3"))
    parser.add_argument("--output", type=Path, default=Path("Data/processed/vani_training_v3_10k"))
    parser.add_argument("--report", type=Path, default=Path("Analysis/scorer_training_data/v3_10k_summary.json"))
    parser.add_argument("--seed", type=int, default=137)
    parser.add_argument("--mixed", action="store_true", help="Allow ordinary mutations as well as swaps from the source")
    run(parser.parse_args())

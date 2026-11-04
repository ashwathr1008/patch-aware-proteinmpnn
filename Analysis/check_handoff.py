"""Check the actual patch mapping and one repeated baseline; curate small artifacts."""
import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def completed_rows(directory):
    summary = read_json(directory / "metadata/summary.json")
    if summary["status"] != "complete":
        raise ValueError("A baseline run did not complete")
    path = directory / "candidates.csv"
    if digest(path) != summary["candidates_csv_sha256"]:
        raise ValueError("Baseline CSV checksum mismatch")
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--repeat", type=Path, required=True)
    parser.add_argument("--patch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    original_rows, repeated_rows = completed_rows(args.baseline), completed_rows(args.repeat)
    key_fields = ("structure_id", "chain", "model_type", "record_type", "sample_index")
    key = lambda row: tuple(row[field] for field in key_fields)
    original = {key(row): row for row in original_rows}
    repeated = {key(row): row for row in repeated_rows}
    if len(original) != len(original_rows) or len(repeated) != len(repeated_rows):
        raise ValueError("Duplicate candidate keys")
    configs = [read_json(path / "metadata/config.json") for path in (args.baseline, args.repeat)]
    for field in ("samples_per_model_per_structure", "seed", "temperature", "batch_size", "backbone_noise", "model", "model_types", "device", "upstream_commit", "upstream_files", "runner_sha256"):
        if configs[0][field] != configs[1][field]:
            raise ValueError("Repeated-run configuration differs: " + field)
    repeated_groups = {(r["structure_id"], r["chain"]) for r in repeated_rows}
    expected = {k for k, r in original.items() if (r["structure_id"], r["chain"]) in repeated_groups}
    if set(repeated) != expected:
        raise ValueError("Repeated run is not a complete subset of the primary run")
    if any(original[k]["sequence"] != r["sequence"] for k, r in repeated.items()):
        raise ValueError("Repeated generated sequences differ")
    max_score_delta = max(abs(float(original[k][field]) - float(r[field]))
                          for k, r in repeated.items() for field in ("score", "global_score"))
    patch = read_json(args.patch / "manifest.json")
    if not all(patch["checks"].values()) or patch["mapping"]["requires_mapping_review"]:
        raise ValueError("Patch invariants or clean-mapping requirement failed")
    structures = read_json(args.baseline / "metadata/structures.json")
    matches = [s for s in structures if s["raw_sha256"] == patch["source"]["sha256"] and s["chain"] == patch["source"]["chain"]]
    if len(matches) != 1:
        raise ValueError("Patch must match exactly one baseline input checksum and chain")
    structure = matches[0]
    residues = patch["residues"]
    if patch["sequences"]["original_observed_chain"] != structure["native_sequence"] or len(residues) != structure["length"]:
        raise ValueError("Patch/native sequence mismatch")
    for residue, baseline in zip(residues, structure["residue_mapping"]):
        left = (residue["observed_sequence_index"], residue["chain"], residue["resseq"], residue["insertion_code"], residue["original_aa"], residue["selected_altloc"])
        right = (baseline["position"], baseline["chain"], baseline["residue_number"], baseline["insertion_code"], baseline["aa"], baseline["selected_altloc"])
        if left != right:
            raise ValueError("Patch/model residue mapping mismatch")
    staged = args.baseline / structure["staged_path"]
    if digest(staged) != structure["staged_sha256"]:
        raise ValueError("Staged input checksum mismatch")
    atoms = {}
    for line in staged.read_text().splitlines():
        if line.startswith("ATOM  "):
            atom_key = (line[21], int(line[22:26]), line[26].strip(), line[12:16].strip())
            atoms[atom_key] = [float(line[i:i+8]) for i in (30, 38, 46)]
    for residue in residues:
        for atom, coordinates in residue["backbone_coordinates"].items():
            parsed = atoms[(residue["chain"], residue["resseq"], residue["insertion_code"], atom)]
            if coordinates != parsed:
                raise ValueError("Patch/model backbone geometry mismatch")
    if len(atoms) != 4 * len(residues):
        raise ValueError("Unexpected staged backbone atom count")
    args.output.mkdir(parents=True, exist_ok=False)
    patch_dir = args.output / "patch_example"
    patch_dir.mkdir()
    patch["source"]["pdb_path"] = "Data/raw/" + structure["structure_id"] + ".pdb"
    patch["source"]["path_basis"] = "repository root"
    (patch_dir / "manifest.json").write_text(json.dumps(patch, indent=2) + "\n", encoding="utf-8")
    for name in ("residues.csv", "original.fasta", "permuted.fasta"):
        shutil.copyfile(args.patch / name, patch_dir / name)
    report = {"status": "passed", "scope": "Development inputs and one same-machine rerun; not generalization or biological validation",
              "repeat": {"structures": [list(g) for g in sorted(repeated_groups)],
                         "generated_sequences_checked": sum(r["record_type"] == "sample" for r in repeated_rows),
                         "native_records_checked": sum(r["record_type"] == "native" for r in repeated_rows),
                         "sequences_identical": True, "maximum_absolute_score_difference": max_score_delta,
                         "primary_csv_sha256": digest(args.baseline / "candidates.csv"),
                         "repeat_csv_sha256": digest(args.repeat / "candidates.csv")},
              "patch_to_model_mapping": {"structure_id": structure["structure_id"], "chain": structure["chain"],
                          "residues_checked": len(residues), "backbone_atoms_checked": len(atoms),
                          "native_sequence_equal": True, "residue_identifiers_equal": True,
                          "backbone_coordinates_equal": True, "raw_input_sha256": structure["raw_sha256"],
                          "patch_original_manifest_sha256": digest(args.patch / "manifest.json")},
              "patch_invariants": patch["checks"]}
    (args.output / "handoff_checks.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

"""Package the existing 1UBQ seed-7 example unchanged for controlled scoring."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from patches.build_dataset import controls_for_pair, patch_contacts, sha256, write_json
from patches.starter import BACKBONE, geometry_fingerprint, parse_pdb


def prepare(example_path, pdb_path, output):
    example = json.loads(example_path.read_text(encoding="utf-8"))
    if output.exists():
        raise ValueError("Output exists; choose a new directory")
    if example.get("schema_version") != 1 or example.get("status") != "development_demo_only":
        raise ValueError("Expected the checked development example")
    if not example.get("checks") or not all(value is True for value in example["checks"].values()):
        raise ValueError("Existing example does not report passing checks")
    if example["source"]["chain"] != "A" or sha256(pdb_path) != example["source"]["sha256"]:
        raise ValueError("PDB source checksum or chain differs from the example")
    residues, mapping = parse_pdb(pdb_path.read_text(encoding="ascii"), "A")
    if mapping["requires_mapping_review"] or len(residues) != 76:
        raise ValueError("Expected the complete 76-residue 1UBQ chain A")
    original = example["sequences"]["original_observed_chain"]
    permuted = example["sequences"]["permuted_observed_chain"]
    if original != "".join(r.aa for r in residues) or len(permuted) != len(original):
        raise ValueError("Example/PDB native sequence or length mismatch")
    rows = example["residues"]
    if len(rows) != len(residues):
        raise ValueError("Example residue record count differs from input")
    indices = example["selection"]["patch_observed_sequence_indices"]
    if (indices != list(range(6, 14)) or example["selection"]["center_observed_sequence_index"] != 11
            or example["parameters"]["seed"] != 7):
        raise ValueError("Expected the original A:6-13 patch centered at A:11, seed 7")
    members = set(indices)
    geometry_hash = geometry_fingerprint(residues)
    if any(example["geometry"][name] != geometry_hash for name in ("original_sha256", "permuted_sha256")):
        raise ValueError("Example geometry hashes differ from the input PDB")
    for residue, row in zip(residues, rows):
        expected = (residue.index, residue.chain, residue.resseq, residue.icode, residue.aa, residue.altloc)
        observed = (row["observed_sequence_index"], row["chain"], row["resseq"], row["insertion_code"], row["original_aa"], row["selected_altloc"])
        if expected != observed or row["in_patch"] != (residue.index in members):
            raise ValueError("Example residue mapping or membership mismatch")
        if row["permuted_aa"] != permuted[residue.index - 1]:
            raise ValueError("Example residue assignments disagree with its full sequence")
        source = row["permuted_identity_source_index"]
        if type(source) is not int or not 1 <= source <= len(residues) or original[source - 1] != row["permuted_aa"]:
            raise ValueError("Invalid identity-source provenance")
        if residue.index in members and source not in members:
            raise ValueError("Permutation draws an identity from outside the patch")
        for atom, xyz in zip(BACKBONE, residue.coordinates):
            if row["backbone_coordinates"][atom] != list(xyz):
                raise ValueError("Example backbone coordinates differ from input")
    changed = sum(a != b for a, b in zip(original, permuted))
    patch = [r for r in residues if r.index in members]
    checks = {"patch_composition_equal": Counter(original[i - 1] for i in indices) == Counter(permuted[i - 1] for i in indices),
              "chain_composition_equal": Counter(original) == Counter(permuted),
              "backbone_geometry_unchanged": True, "patch_membership_unchanged": True,
              "outside_patch_unchanged": all(a == b for i, (a, b) in enumerate(zip(original, permuted), 1) if i not in members),
              "nonidentity_assignment": changed == 6, "chain_length_unchanged": len(original) == len(permuted)}
    if not all(checks.values()) or changed != example["selection"]["changed_identity_count"]:
        raise ValueError("The saved first-pair invariants did not reproduce")
    distances, contacts = patch_contacts(patch, radius=8.0)
    pair_id = "1UBQ_A_existing_seed7"
    pair = {"pair_id": pair_id, "structure_id": "1UBQ", "chain": "A", "patch_id": "1UBQ_A_existing_patch",
            "group": "ubiquitin", "split": "development", "original_sequence": original,
            "permuted_sequence": permuted, "patch_indices": indices, "patch_size": len(indices),
            "changed_identity_count": changed, "patch_contiguous_in_sequence": True,
            "permutation_seed": 7, "identity_source_indices": [row["permuted_identity_source_index"] for row in rows if row["in_patch"]],
            "original_sequence_sha256": hashlib.sha256(original.encode("ascii")).hexdigest(),
            "permuted_sequence_sha256": hashlib.sha256(permuted.encode("ascii")).hexdigest(),
            "geometry_original_sha256": geometry_hash, "geometry_permuted_sha256": geometry_hash,
            "checks": checks, "controls": controls_for_pair(original, permuted, indices, contacts)}
    output.mkdir(parents=True, exist_ok=False)
    (output / "inputs").mkdir()
    (output / "metadata").mkdir()
    copied = output / "inputs/1UBQ.pdb"
    copied.write_bytes(pdb_path.read_bytes())
    payload = {"schema_version": 1, "purpose": "patch_compatibility_pilot",
               "interpretation": "The previously shown seed-7 pair is preserved exactly; no new permutation, aggregation label or biological outcome.",
               "config": {"source_example_manifest_sha256": sha256(example_path),
                          "packer_sha256": sha256(Path(__file__)), "permutation_preserved_exactly": True,
                          "original_patch_parameters": example["parameters"],
                          "descriptor": {"hydrophobic_set": "AVILMFWY", "contact_distance_angstrom": 8.0,
                                         "exclude_observed_sequence_separation_at_most": 1,
                                         "delta_sign": "permuted minus original; no preferred biological direction"}},
               "structures": [{"structure_id": "1UBQ", "chain": "A", "pdb_path": "inputs/1UBQ.pdb",
                               "sha256": sha256(copied), "raw_sha256": sha256(copied), "group": "ubiquitin",
                               "split": "development", "expected_length": 76}], "pairs": [pair]}
    write_json(output / "pairs.json", payload)
    write_json(output / "patch_geometry.json", {"patch_indices": indices, "center_index": 11,
                                               "ca_pair_distances": distances, "descriptor_contacts": contacts,
                                               "geometry_sha256": geometry_hash})
    summary = {"status": "complete", "pair_count": 1, "pair_id": pair_id, "source_example_manifest_sha256": sha256(example_path),
               "pairs_sha256": sha256(output / "pairs.json"), "source_pdb_sha256": sha256(copied),
               "original_and_permuted_sequences_preserved_exactly": True, "changed_identity_count": changed,
               "residue_mappings_checked": len(residues), "backbone_atoms_checked": len(residues) * 4,
               "all_pair_invariants_passed": all(checks.values())}
    write_json(output / "metadata/summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example", type=Path, required=True)
    parser.add_argument("--pdb", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.example.resolve(), args.pdb.resolve(), args.output.resolve()), indent=2))


if __name__ == "__main__":
    main()

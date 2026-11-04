"""Build audited composition-matched patch pairs without scoring or model training."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baseline.run_baseline import stage_backbone
from patches.starter import BACKBONE, geometry_fingerprint, parse_pdb, squared_distance

ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
HYDROPHOBIC = frozenset("AVILMFWY")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with Path(path).open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def composition(sequence):
    counts = Counter(sequence)
    return {aa: counts[aa] for aa in ALPHABET}


def select_patches(residues, neighbor_radius=10.0, patch_radius=8.0,
                   min_patch_size=8, max_patches=3, max_jaccard=0.5):
    """Geometry-ranked selection; heterogeneity is a permutation-feasibility filter."""
    counts = {r.index: sum(s.index != r.index and squared_distance(r.ca, s.ca) <= neighbor_radius ** 2
                           for s in residues) for r in residues}
    ranked = sorted(residues, key=lambda r: (counts[r.index], r.index))
    threshold = counts[ranked[max(0, math.ceil(len(ranked) * 0.25) - 1)].index]
    selected, seen_memberships = [], set()
    diagnostics = {"bottom_quartile_count_threshold": threshold, "candidate_centers": 0,
                   "too_small": 0, "homogeneous": 0, "duplicate_memberships": 0,
                   "overlap_rejections": 0, "eligible_unique_memberships": 0}
    for center in ranked:
        if counts[center.index] > threshold:
            break
        diagnostics["candidate_centers"] += 1
        patch = [r for r in residues if squared_distance(center.ca, r.ca) <= patch_radius ** 2]
        membership = tuple(r.index for r in patch)
        if len(patch) < min_patch_size:
            diagnostics["too_small"] += 1
            continue
        if len({r.aa for r in patch}) < 2:
            diagnostics["homogeneous"] += 1
            continue
        if membership in seen_memberships:
            diagnostics["duplicate_memberships"] += 1
            continue
        seen_memberships.add(membership)
        diagnostics["eligible_unique_memberships"] += 1
        if len(selected) >= max_patches:
            continue
        members = set(membership)
        if any(len(members & set(previous["indices"])) / len(members | set(previous["indices"])) > max_jaccard
               for previous in selected):
            diagnostics["overlap_rejections"] += 1
            continue
        selected.append({"center": center, "residues": patch, "indices": list(membership)})
    diagnostics["selected_patch_count"] = len(selected)
    diagnostics["patch_shortfall"] = max_patches - len(selected)
    return selected, counts, diagnostics


def distinct_permutations(identities, count, seed, max_attempts=10000):
    """Reject unchanged/repeated assignments; bounded attempts never relax matching."""
    identities = tuple(identities)
    rng = random.Random(seed)
    seen = {identities}
    output = []
    attempts = 0
    while len(output) < count and attempts < max_attempts:
        attempts += 1
        source_indices = list(range(len(identities)))
        rng.shuffle(source_indices)
        assignment = tuple(identities[index] for index in source_indices)
        if assignment in seen:
            continue
        seen.add(assignment)
        output.append((assignment, source_indices))
    return output, attempts


def patch_contacts(patch, radius=8.0):
    distances, contacts = [], []
    for offset, left in enumerate(patch):
        for right in patch[offset + 1:]:
            distance_squared = squared_distance(left.ca, right.ca)
            distance = math.sqrt(distance_squared)
            nonadjacent = abs(left.index - right.index) > 1
            contact = nonadjacent and distance_squared <= radius ** 2
            distances.append({"left_index": left.index, "right_index": right.index,
                              "ca_distance_angstrom": distance,
                              "sequence_nonadjacent": nonadjacent, "descriptor_contact": contact})
            if contact:
                contacts.append((left.index, right.index))
    return distances, contacts


def controls_for_pair(original, permuted, patch_indices, contacts):
    native_patch = "".join(original[i - 1] for i in patch_indices)
    shuffled_patch = "".join(permuted[i - 1] for i in patch_indices)
    hydrophobic_counts = [sum(sequence[i - 1] in HYDROPHOBIC and sequence[j - 1] in HYDROPHOBIC
                              for i, j in contacts) for sequence in (original, permuted)]
    fractions = [count / len(contacts) if contacts else None for count in hydrophobic_counts]
    return {"composition_original": composition(native_patch),
            "composition_permuted": composition(shuffled_patch),
            "composition_tie": Counter(native_patch) == Counter(shuffled_patch),
            "geometry_tie": True, "contact_count": len(contacts),
            "hydrophobic_contacts_original": hydrophobic_counts[0],
            "hydrophobic_contacts_permuted": hydrophobic_counts[1],
            "hydrophobic_fraction_original": fractions[0],
            "hydrophobic_fraction_permuted": fractions[1],
            "hydrophobic_fraction_delta": fractions[1] - fractions[0] if contacts else None}


def load_inputs(manifest_path):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if manifest.get("schema_version") != 1 or manifest.get("purpose") not in ("development_only", "patch_compatibility_pilot"):
        raise ValueError("Unsupported structure manifest schema/purpose")
    if not manifest.get("structures"):
        raise ValueError("At least one input structure is required")
    seen = set()
    for item in manifest["structures"]:
        identifier, chain = item["structure_id"], item["chain"]
        if not re.fullmatch(r"[A-Za-z0-9_-]+", identifier) or not re.fullmatch(r"[A-Za-z0-9]", chain):
            raise ValueError("Unsupported structure/chain identifier")
        key = (identifier.casefold(), chain.casefold())
        if key in seen:
            raise ValueError("Duplicate structure/chain")
        seen.add(key)
        if not item.get("split") or not item.get("group"):
            raise ValueError("Every source requires frozen split and group metadata")
        source = (manifest_path.parent / item["pdb_path"]).resolve()
        if sha256(source) != item["sha256"]:
            raise ValueError("Input checksum mismatch: " + identifier)
        item["resolved_path"] = source
    return manifest


def pair_record(pair_id, patch_id, item, residues, patch, assignment, sources, contacts, seed):
    original = "".join(r.aa for r in residues)
    permuted_list = list(original)
    for residue, identity in zip(patch, assignment):
        permuted_list[residue.index - 1] = identity
    permuted = "".join(permuted_list)
    indices = [r.index for r in patch]
    members = set(indices)
    geometry_hash = geometry_fingerprint(residues)
    checks = {"patch_composition_equal": Counter(r.aa for r in patch) == Counter(assignment),
              "chain_composition_equal": Counter(original) == Counter(permuted),
              "backbone_geometry_unchanged": True, "patch_membership_unchanged": True,
              "outside_patch_unchanged": all(a == b for i, (a, b) in enumerate(zip(original, permuted), 1) if i not in members),
              "nonidentity_assignment": original != permuted, "chain_length_unchanged": len(original) == len(permuted)}
    if not all(checks.values()):
        raise AssertionError("Pair invariant failure")
    return {"pair_id": pair_id, "structure_id": item["structure_id"], "chain": item["chain"],
            "patch_id": patch_id, "group": item["group"], "split": item["split"],
            "original_sequence": original, "permuted_sequence": permuted, "patch_indices": indices,
            "patch_size": len(patch), "changed_identity_count": sum(a != b for a, b in zip(original, permuted)),
            "patch_contiguous_in_sequence": indices == list(range(indices[0], indices[-1] + 1)),
            "permutation_seed": seed, "identity_source_indices": [patch[i].index for i in sources],
            "original_sequence_sha256": hashlib.sha256(original.encode("ascii")).hexdigest(),
            "permuted_sequence_sha256": hashlib.sha256(permuted.encode("ascii")).hexdigest(),
            "geometry_original_sha256": geometry_hash, "geometry_permuted_sha256": geometry_hash,
            "checks": checks, "controls": controls_for_pair(original, permuted, indices, contacts)}


def build_dataset(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "inputs").mkdir()
    (output / "metadata").mkdir()
    summary = {"schema_version": 1, "status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
               "structure_reports": [], "exclusions": [], "pair_count": 0, "patch_count": 0}
    try:
        if not all(math.isfinite(v) and v > 0 for v in (args.neighbor_radius, args.patch_radius, args.contact_radius)):
            raise ValueError("Radii must be finite and positive")
        if args.min_patch_size < 2 or args.patches_per_structure < 1 or args.permutations_per_patch < 1:
            raise ValueError("Invalid requested patch/permutation counts")
        if not math.isfinite(args.max_jaccard) or not 0 <= args.max_jaccard <= 1:
            raise ValueError("Jaccard cap must be between zero and one")
        manifest_path = args.manifest.resolve()
        manifest = load_inputs(manifest_path)
        config = {key: value for key, value in vars(args).items() if key not in ("manifest", "output")}
        config.update({"source_manifest_sha256": sha256(manifest_path),
                       "builder_sha256": sha256(Path(__file__)),
                       "parser_sha256": sha256(Path(__file__).with_name("starter.py")),
                       "stager_sha256": sha256(Path(__file__).resolve().parents[1] / "baseline/run_baseline.py"),
                       "selection_rule": "Rank bottom-quartile neighbor-count centers by count, then observed sequence index; require sufficient heterogeneous membership; skip duplicate memberships; greedily retain patches only when Jaccard overlap with every retained patch is at most max_jaccard.",
                       "surface_proxy": "Other CA atoms within neighbor_radius; lower is surface-like, not SASA.",
                       "heterogeneity_rule": "At least two identity types is required only for a nonidentity permutation to exist; scores never inform selection.",
                       "descriptor": {"hydrophobic_set": "AVILMFWY", "contact_distance_angstrom": args.contact_radius,
                                      "exclude_observed_sequence_separation_at_most": 1,
                                      "definition": "Number of patch contacts with both endpoints hydrophobic divided by all eligible patch contacts; zero contacts yields null.",
                                      "delta_sign": "permuted minus original; no biological preferred direction"},
                       "geometry_control": "Both pair members share exactly one stored backbone geometry, membership set and contact graph.",
                       "max_shuffle_attempts_per_patch": 10000})
        pairs, structures, patch_records, structure_records = [], [], [], []
        summary["requested_structure_count"] = len(manifest["structures"])
        summary["requested_patch_count"] = len(manifest["structures"]) * args.patches_per_structure
        summary["requested_pair_count"] = summary["requested_patch_count"] * args.permutations_per_patch
        write_json(output / "metadata/config.json", config)
        for item in manifest["structures"]:
            stem = item["structure_id"] + "_" + item["chain"]
            staged = output / "inputs" / (stem + ".pdb")
            try:
                staged_info = stage_backbone(item["resolved_path"], item["chain"], staged)
                residues, diagnostics = parse_pdb(staged.read_text(encoding="ascii"), item["chain"])
                if diagnostics["requires_mapping_review"]:
                    raise ValueError("Unsupported gaps, insertion codes or incomplete backbone mapping")
                if item.get("expected_length", len(residues)) != len(residues):
                    raise ValueError("Expected sequence length does not match staged backbone")
                if "".join(r.aa for r in residues) != staged_info["native_sequence"]:
                    raise ValueError("Patch parser and baseline stager sequences differ")
                for residue, mapping in zip(residues, staged_info["residue_mapping"]):
                    if (residue.index, residue.chain, residue.resseq, residue.icode, residue.aa) != (mapping["position"], mapping["chain"], mapping["residue_number"], mapping["insertion_code"], mapping["aa"]):
                        raise ValueError("Patch parser and baseline residue mapping differ")
                selected, counts, selection_report = select_patches(residues, args.neighbor_radius, args.patch_radius,
                                      args.min_patch_size, args.patches_per_structure, args.max_jaccard)
            except (ValueError, OSError, UnicodeError) as error:
                summary["exclusions"].append({"structure_id": item["structure_id"], "chain": item["chain"], "reason": str(error)})
                continue
            selection_report.update({"structure_id": item["structure_id"], "chain": item["chain"],
                                     "group": item["group"], "split": item["split"], "pair_count": 0,
                                     "permutation_shortfall": 0})
            summary["structure_reports"].append(selection_report)
            if not selected:
                summary["exclusions"].append({"structure_id": item["structure_id"], "chain": item["chain"],
                                             "reason": "No eligible patches under the frozen selection rules"})
                continue
            structures.append({"structure_id": item["structure_id"], "chain": item["chain"],
                               "pdb_path": staged.relative_to(output).as_posix(), "sha256": sha256(staged),
                               "raw_sha256": item["sha256"], "split": item["split"], "group": item["group"],
                               "expected_length": len(residues)})
            structure_records.append({"structure_id": item["structure_id"], "chain": item["chain"],
                                      "raw_sha256": item["sha256"], "staged_sha256": sha256(staged),
                                      "geometry_sha256": geometry_fingerprint(residues), "mapping_diagnostics": diagnostics,
                                      "baseline_stager_mapping": staged_info["residue_mapping"],
                                      "residues": [{"observed_sequence_index": r.index, "chain": r.chain,
                                                    "resseq": r.resseq, "insertion_code": r.icode, "original_aa": r.aa,
                                                    "ca_neighbor_count": counts[r.index],
                                                    "backbone_coordinates": {atom: list(xyz) for atom, xyz in zip(BACKBONE, r.coordinates)}}
                                                   for r in residues]})
            for patch_number, selection in enumerate(selected, 1):
                patch_id = stem + "_patch_" + str(patch_number).zfill(2)
                patch = selection["residues"]
                distances, contacts = patch_contacts(patch, args.contact_radius)
                seed_bytes = (str(args.seed) + "|" + stem + "|" + ",".join(str(i) for i in selection["indices"])).encode("ascii")
                derived_seed = int.from_bytes(hashlib.sha256(seed_bytes).digest()[:8], "big")
                permutations, attempts = distinct_permutations([r.aa for r in patch], args.permutations_per_patch, derived_seed)
                patch_records.append({"patch_id": patch_id, "structure_id": item["structure_id"], "chain": item["chain"],
                                      "center_index": selection["center"].index,
                                      "center_residue_id": "{}:{}{}".format(item["chain"], selection["center"].resseq, selection["center"].icode),
                                      "patch_indices": selection["indices"], "size": len(patch),
                                      "permutation_seed": derived_seed, "shuffle_attempts": attempts,
                                      "requested_permutations": args.permutations_per_patch,
                                      "actual_permutations": len(permutations),
                                      "ca_pair_distances": distances,
                                      "descriptor_contacts": [list(edge) for edge in contacts]})
                selection_report["permutation_shortfall"] += args.permutations_per_patch - len(permutations)
                for variant_number, (assignment, sources) in enumerate(permutations, 1):
                    pair_id = patch_id + "_perm_" + str(variant_number).zfill(3)
                    pairs.append(pair_record(pair_id, patch_id, item, residues, patch, assignment, sources, contacts, derived_seed))
                    selection_report["pair_count"] += 1
            write_json(output / "metadata/summary.json", summary)
        payload = {"schema_version": 1, "purpose": "patch_compatibility_pilot", "config": config,
                   "interpretation": "Composition-matched arrangement controls, not aggregation labels. Multiple pairs per source protein are not independent biological replicates.",
                   "structures": structures, "pairs": pairs}
        write_json(output / "pairs.json", payload)
        write_json(output / "patches.json", {"schema_version": 1, "structures": structure_records, "patches": patch_records})
        csv_fields = ["pair_id", "structure_id", "chain", "group", "split", "patch_id", "patch_size", "changed_identity_count",
                      "patch_contiguous_in_sequence", "composition_tie", "geometry_tie", "contact_count",
                      "hydrophobic_contacts_original", "hydrophobic_contacts_permuted", "hydrophobic_fraction_original",
                      "hydrophobic_fraction_permuted", "hydrophobic_fraction_delta", "all_pair_invariants_passed"]
        with (output / "pair_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=csv_fields)
            writer.writeheader()
            for pair in pairs:
                row = {field: pair[field] for field in csv_fields if field in pair}
                row.update({field: pair["controls"][field] for field in csv_fields if field in pair["controls"]})
                row["all_pair_invariants_passed"] = all(pair["checks"].values())
                writer.writerow(row)
        summary.update({"structure_count": len(structures), "patch_count": len(patch_records), "pair_count": len(pairs),
                        "patch_shortfall": summary["requested_patch_count"] - len(patch_records),
                        "pair_shortfall": summary["requested_pair_count"] - len(pairs),
                        "zero_contact_pair_count": sum(pair["controls"]["contact_count"] == 0 for pair in pairs),
                        "all_pair_invariants_passed": all(all(pair["checks"].values()) for pair in pairs),
                        "output_sha256": {name: sha256(output / name) for name in ("pairs.json", "patches.json", "pair_manifest.csv")}})
        if not pairs:
            raise ValueError("No valid pairs under the frozen rules; inspect exclusions and selection reports")
        summary["status"] = "complete"
        return summary
    except BaseException as error:
        summary.update({"status": "failed", "error": {"type": type(error).__name__, "message": str(error)}})
        raise
    finally:
        summary["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(output / "metadata/summary.json", summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--neighbor-radius", type=float, default=10.0)
    parser.add_argument("--patch-radius", type=float, default=8.0)
    parser.add_argument("--contact-radius", type=float, default=8.0)
    parser.add_argument("--min-patch-size", type=int, default=8)
    parser.add_argument("--patches-per-structure", type=int, default=3)
    parser.add_argument("--permutations-per-patch", type=int, default=5)
    parser.add_argument("--max-jaccard", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    result = build_dataset(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

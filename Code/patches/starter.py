"""Build a reproducible illustrative patch pair; this is not a trained model.

Only observed identities and N/CA/C/O coordinates are used. Neighbor counts are
an exposure proxy, not SASA, and the output provides no aggregation labels.
Compatible with Python 3.9 and the standard library.
"""

import argparse
from collections import Counter
import csv
from dataclasses import dataclass
import hashlib
import io
import json
import math
from pathlib import Path
import random
from typing import Dict, List, Optional, Tuple


AA3_TO_1 = dict(zip(
    "ALA CYS ASP GLU PHE GLY HIS ILE LYS LEU MET ASN PRO GLN ARG SER THR VAL TRP TYR".split(),
    "ACDEFGHIKLMNPQRSTVWY",
))
BACKBONE = ("N", "CA", "C", "O")
Coordinate = Tuple[float, float, float]


@dataclass(frozen=True)
class Residue:
    index: int  # One-based position in the observed standard-residue FASTA.
    chain: str
    resseq: int
    icode: str
    name: str
    altloc: str
    coordinates: Tuple[Optional[Coordinate], ...]

    @property
    def aa(self) -> str:
        return AA3_TO_1[self.name]

    @property
    def complete(self) -> bool:
        return all(coordinate is not None for coordinate in self.coordinates)

    @property
    def ca(self) -> Optional[Coordinate]:
        return self.coordinates[1]


def parse_pdb(text: str, chain: str):
    """Read the first explicit model, or the sole implicit model.

    For each residue, consider each backbone altloc label with blank atoms as
    shared atoms. Choose the candidate with most complete backbone, then largest
    summed occupancy, then blank/A/lexical label. Duplicate atom records use
    largest occupancy, then smallest serial. No sidechain coordinates are read.
    """
    if len(chain) != 1:
        raise ValueError("--chain must be exactly one PDB chain character (a space for blank).")
    lines = text.splitlines()
    model_starts = [i for i, line in enumerate(lines) if line[:6].strip() == "MODEL"]
    first = model_starts[0] + 1 if model_starts else 0
    model_label = lines[first - 1][10:14].strip() if model_starts else None
    records = {}
    ignored = set()
    for line_number, line in enumerate(lines[first:], first + 1):
        record_type = line[:6].strip()
        if record_type in ("ENDMDL", "END", "MODEL"):
            break
        if record_type != "ATOM" or len(line) < 22 or line[21] != chain:
            continue
        name = line[17:20].strip()
        if name not in AA3_TO_1:
            ignored.add((line[22:26].strip(), line[26:27].strip(), name))
            continue
        try:
            resseq = int(line[22:26])
        except ValueError as exc:
            raise ValueError("Invalid residue number on PDB line {}.".format(line_number)) from exc
        icode = line[26:27].strip()
        key = (resseq, icode)
        if key not in records:
            records[key] = {"name": name, "atoms": {}}
        if records[key]["name"] != name:
            raise ValueError("Ambiguous residue identities at {}:{}{}.".format(chain, resseq, icode))
        atom_name = line[12:16].strip()
        if atom_name not in BACKBONE:
            continue
        try:
            xyz = tuple(float(line[start:start + 8]) for start in (30, 38, 46))
            occupancy = float(line[54:60].strip() or "0")
            serial = int(line[6:11])
        except ValueError as exc:
            raise ValueError("Invalid backbone atom on PDB line {}.".format(line_number)) from exc
        if not all(math.isfinite(value) for value in xyz + (occupancy,)):
            raise ValueError("Nonfinite backbone coordinate or occupancy on line {}.".format(line_number))
        altloc = line[16].strip()
        atom_key = (atom_name, altloc)
        candidate = (occupancy, serial, xyz)
        previous = records[key]["atoms"].get(atom_key)
        if previous is None or (-occupancy, serial, xyz) < (-previous[0], previous[1], previous[2]):
            records[key]["atoms"][atom_key] = candidate
    if not records:
        raise ValueError("No standard ATOM residues found in chain {!r} in the first model.".format(chain))
    residues = []
    for index, ((resseq, icode), item) in enumerate(records.items(), 1):
        atoms = item["atoms"]
        labels = {""} | {label for _, label in atoms}
        candidates = []
        for label in labels:
            chosen = [atoms.get((name, label), atoms.get((name, ""))) for name in BACKBONE]
            completeness = sum(atom is not None for atom in chosen)
            occupancy = sum(atom[0] for atom in chosen if atom is not None)
            label_order = (0 if label == "" else 1 if label == "A" else 2, label)
            candidates.append(((-completeness, -occupancy, label_order), label, chosen))
        _, altloc, chosen = min(candidates, key=lambda candidate: candidate[0])
        coordinates = tuple(atom[2] if atom is not None else None for atom in chosen)
        residues.append(Residue(index, chain, resseq, icode, item["name"], altloc, coordinates))
    gaps = []
    nonmonotonic = []
    for left, right in zip(residues, residues[1:]):
        if right.resseq > left.resseq + 1:
            gaps.append({"after_observed_index": left.index, "before_observed_index": right.index,
                         "missing_number_start": left.resseq + 1, "missing_number_end": right.resseq - 1})
        if right.resseq < left.resseq or (right.resseq == left.resseq and right.icode <= left.icode):
            nonmonotonic.append([left.index, right.index])
    diagnostics = {
        "model": model_label,
        "explicit_model_count": len(model_starts),
        "selected_model_policy": "first explicit model, otherwise implicit model",
        "sequence_basis": "observed standard ATOM residue order; no SEQRES or unresolved-residue reconstruction",
        "incomplete_backbone_indices": [r.index for r in residues if not r.complete],
        "insertion_code_indices": [r.index for r in residues if r.icode],
        "numbering_gaps": gaps,
        "nonmonotonic_numbering_pairs": nonmonotonic,
        "ignored_nonstandard_atom_residues": [
            {"resseq": seq, "insertion_code": code, "name": name} for seq, code, name in sorted(ignored)
        ],
        "heteroatom_policy": "HETATM records excluded, including modified amino acids",
    }
    diagnostics["requires_mapping_review"] = bool(
        gaps or nonmonotonic or ignored or diagnostics["incomplete_backbone_indices"]
        or diagnostics["insertion_code_indices"]
    )
    return residues, diagnostics


def squared_distance(left: Coordinate, right: Coordinate) -> float:
    return sum((a - b) ** 2 for a, b in zip(left, right))


def geometry_fingerprint(residues: List[Residue]) -> str:
    payload = [(r.index, r.chain, r.resseq, r.icode, r.coordinates) for r in residues]
    return hashlib.sha256(json.dumps(payload, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def build_example(pdb_path: Path, chain: str, seed: int = 7, neighbor_radius: float = 10.0,
                  patch_radius: float = 8.0, min_patch_size: int = 4) -> Dict:
    """Select one low-neighbor-count center and permute its identities."""
    if not all(math.isfinite(value) and value > 0 for value in (neighbor_radius, patch_radius)):
        raise ValueError("Radii must be finite and positive.")
    if min_patch_size < 2:
        raise ValueError("Minimum patch size must be at least two.")
    pdb_bytes = Path(pdb_path).read_bytes()
    residues, diagnostics = parse_pdb(pdb_bytes.decode("ascii"), chain)
    eligible = [residue for residue in residues if residue.complete]
    if len(eligible) < min_patch_size:
        raise ValueError("Too few complete backbone residues for the requested patch size.")
    counts = {
        r.index: sum(s.index != r.index and squared_distance(r.ca, s.ca) <= neighbor_radius ** 2
                     for s in eligible)
        for r in eligible
    }
    ranked = sorted(eligible, key=lambda r: (counts[r.index], r.index))
    # Use the lowest quartile of neighbor counts, including ties at its boundary.
    threshold = counts[ranked[max(0, math.ceil(len(ranked) * 0.25) - 1)].index]
    center = None
    patch = []
    for candidate in ranked:
        if counts[candidate.index] > threshold:
            break
        members = [r for r in eligible if squared_distance(candidate.ca, r.ca) <= patch_radius ** 2]
        if len(members) >= min_patch_size and len({r.aa for r in members}) > 1:
            center, patch = candidate, members
            break
    if center is None:
        raise ValueError("No heterogeneous patch of sufficient size among low-neighbor-count centers. "
                         "This demo does not relax its selection rule automatically.")
    original = [r.aa for r in patch]
    source_indices = list(range(len(patch)))
    rng = random.Random(seed)
    for _ in range(100):
        rng.shuffle(source_indices)
        if [original[i] for i in source_indices] != original:
            break
    else:
        for offset in range(1, len(patch)):
            source_indices = list(range(offset, len(patch))) + list(range(offset))
            if [original[i] for i in source_indices] != original:
                break
    assignments = {r.index: r.aa for r in residues}
    identity_sources = {r.index: r.index for r in residues}
    for target, source_index in zip(patch, source_indices):
        assignments[target.index] = patch[source_index].aa
        identity_sources[target.index] = patch[source_index].index
    patch_indices = [r.index for r in patch]
    patch_set = set(patch_indices)
    original_sequence = "".join(r.aa for r in residues)
    permuted_sequence = "".join(assignments[r.index] for r in residues)
    permuted_patch = [assignments[r.index] for r in patch]
    # Both assignments share immutable backbone coordinates and fixed membership.
    permuted_geometry = list(residues)
    original_geometry_hash = geometry_fingerprint(residues)
    permuted_geometry_hash = geometry_fingerprint(permuted_geometry)
    checks = {
        "patch_composition_equal": Counter(original) == Counter(permuted_patch),
        "chain_composition_equal": Counter(original_sequence) == Counter(permuted_sequence),
        "backbone_geometry_unchanged": original_geometry_hash == permuted_geometry_hash,
        "patch_membership_unchanged": patch_indices == [r.index for r in permuted_geometry if r.index in patch_set],
        "outside_patch_unchanged": all(assignments[r.index] == r.aa for r in residues if r.index not in patch_set),
        "nonidentity_assignment": original != permuted_patch,
        "chain_length_unchanged": len(original_sequence) == len(permuted_sequence),
    }
    if not all(checks.values()):
        raise AssertionError("Patch-pair invariants failed: {}".format(checks))
    rows = []
    for residue in residues:
        rows.append({
            "observed_sequence_index": residue.index,
            "chain": residue.chain,
            "resseq": residue.resseq,
            "insertion_code": residue.icode,
            "pdb_residue_id": "{}:{}{}".format(residue.chain or " ", residue.resseq, residue.icode),
            "selected_altloc": residue.altloc,
            "complete_backbone": residue.complete,
            "in_patch": residue.index in patch_set,
            "original_aa": residue.aa,
            "permuted_aa": assignments[residue.index],
            "permuted_identity_source_index": identity_sources[residue.index],
            "ca_neighbor_count": counts.get(residue.index),
            "backbone_coordinates": {
                name: list(xyz) if xyz is not None else None
                for name, xyz in zip(BACKBONE, residue.coordinates)
            },
        })
    return {
        "schema_version": 1,
        "status": "development_demo_only",
        "interpretation": "One illustrative composition-matched arrangement pair; no training, "
                          "aggregation label, SASA estimate, or biological performance claim.",
        "source": {"pdb_path": str(Path(pdb_path).resolve()), "sha256": hashlib.sha256(pdb_bytes).hexdigest(),
                   "chain": chain},
        "parameters": {"seed": seed, "neighbor_radius_angstrom": neighbor_radius,
                       "patch_radius_angstrom": patch_radius, "min_patch_size": min_patch_size},
        "selection": {
            "exposure_proxy": "Number of OTHER complete-backbone residue CAs within neighbor_radius; "
                              "lower counts are surface-like, not solvent-accessible surface area (SASA).",
            "center_rule": "Lowest neighbor count, then observed sequence index, among bottom-quartile "
                           "counts (ties included), with sufficient patch size and at least two identities.",
            "membership_rule": "All complete-backbone residues with CA within patch_radius of center CA.",
            "altloc_rule": "Coherent backbone label plus shared blank atoms; most complete, highest "
                           "summed occupancy, then blank/A/lexical label; duplicate atoms by occupancy/serial/coordinates.",
            "center_observed_sequence_index": center.index,
            "center_pdb_residue_id": "{}:{}{}".format(center.chain, center.resseq, center.icode),
            "bottom_quartile_count_threshold": threshold,
            "patch_observed_sequence_indices": patch_indices,
            "patch_size": len(patch),
            "changed_identity_count": sum(a != b for a, b in zip(original, permuted_patch)),
        },
        "mapping": diagnostics,
        "sequences": {"original_observed_chain": original_sequence,
                      "permuted_observed_chain": permuted_sequence},
        "patch_composition": {"original": dict(sorted(Counter(original).items())),
                              "permuted": dict(sorted(Counter(permuted_patch).items()))},
        "geometry": {"policy": "Backbone coordinates stored once and shared by both identity assignments; "
                               "native sidechains are not used or regenerated.",
                     "original_sha256": original_geometry_hash, "permuted_sha256": permuted_geometry_hash},
        "checks": checks,
        "residues": rows,
    }


def write_example(manifest: Dict, output: Path) -> None:
    """Write a new output directory, refusing to overwrite existing artifacts."""
    output = Path(output)
    if output.exists():
        raise ValueError("Output already exists; choose a new directory: {}".format(output))
    rows = []
    for residue in manifest["residues"]:
        row = {key: value for key, value in residue.items() if key != "backbone_coordinates"}
        for atom in BACKBONE:
            xyz = residue["backbone_coordinates"][atom]
            for dimension, value in zip("xyz", xyz if xyz is not None else (None, None, None)):
                row["{}_{}".format(atom, dimension)] = value
        rows.append(row)
    csv_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(csv_buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    payloads = {
        "manifest.json": json.dumps(manifest, indent=2, allow_nan=False) + "\n",
        "residues.csv": csv_buffer.getvalue(),
    }
    for label in ("original", "permuted"):
        sequence = manifest["sequences"][label + "_observed_chain"]
        payloads[label + ".fasta"] = ">{}_observed_chain|development_demo_only\n{}\n".format(label, sequence)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    for name, content in payloads.items():
        with (output / name).open("w", encoding="utf-8", newline="") as handle:
            handle.write(content)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdb", required=True, type=Path)
    parser.add_argument("--chain", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--neighbor-radius", type=float, default=10.0)
    parser.add_argument("--patch-radius", type=float, default=8.0)
    parser.add_argument("--min-patch-size", type=int, default=4)
    args = parser.parse_args()
    try:
        manifest = build_example(args.pdb, args.chain, args.seed, args.neighbor_radius,
                                 args.patch_radius, args.min_patch_size)
        write_example(manifest, args.output)
    except (ValueError, OSError, UnicodeError) as exc:
        parser.exit(2, "error: {}\n".format(exc))
    print(json.dumps({"output": str(args.output.resolve()), "status": manifest["status"],
                      "patch_size": manifest["selection"]["patch_size"],
                      "changed_identity_count": manifest["selection"]["changed_identity_count"],
                      "checks": manifest["checks"],
                      "requires_mapping_review": manifest["mapping"]["requires_mapping_review"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

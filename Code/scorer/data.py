"""Validate measured pairs against preassigned parent/homology splits and PDBs."""
import hashlib
import json
import math
import random
import tempfile
from collections import Counter
from pathlib import Path

from baseline.run_baseline import stage_backbone, sha256
from patches.starter import parse_pdb


def fixed_orders(length, count=8, seed=137, parent=""):
    if count < 1 or count > math.factorial(length):
        raise ValueError("Invalid order count")
    rng = random.Random(hashlib.sha256(f"stability_orders_v1|{seed}|{parent}".encode()).hexdigest())
    orders = []
    while len(orders) < count:
        order = list(range(length))
        rng.shuffle(order)
        if order not in orders:
            orders.append(order)
    return orders


def normalize_ddg(value, convention, units):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Missing/censored/nonfinite experimental label")
    if units not in ("kcal/mol", "kJ/mol"):
        raise ValueError("Explicit kcal/mol or kJ/mol units required")
    if convention not in ("positive_destabilizing", "positive_stabilizing"):
        raise ValueError("Explicit experimental ddG sign convention required")
    return value / (4.184 if units == "kJ/mol" else 1) * (1 if convention == "positive_destabilizing" else -1)


def load_dataset(path):
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if data.get("schema_version") != 1:
        raise ValueError("Expected schema_version=1")
    homology = data["homology"]
    if not homology.get("method") or not homology.get("parameters") or not homology.get("audit_source"):
        raise ValueError("Document externally computed close-homology groups before pairing")
    if not data.get("quality_policy") or not data.get("label_source"):
        raise ValueError("Experimental provenance and quality policy required")
    parents, clusters = {}, {}
    # All parent/cluster splits are validated before any pair is accepted.
    for parent in data["parents"]:
        pid, cluster, split = parent["parent_id"], parent["homology_group"], parent["split"]
        if pid in parents or not cluster or split not in ("train", "validation", "test"):
            raise ValueError("Invalid/duplicate parent or split")
        if cluster in clusters and clusters[cluster] != split:
            raise ValueError("Homology group crosses splits")
        clusters[cluster] = split
        parents[pid] = dict(parent)
    sequence_splits = {}
    for parent in parents.values():
        pdb = (path.parent / parent["pdb_path"]).resolve()
        if sha256(pdb) != parent["pdb_sha256"]:
            raise ValueError("Backbone checksum mismatch")
        with tempfile.TemporaryDirectory() as tmp:
            staged = Path(tmp) / "backbone.pdb"
            mapping = stage_backbone(pdb, parent["chain"], staged)
            residues, diagnostics = parse_pdb(staged.read_text(encoding="utf-8"), parent["chain"])
        if diagnostics["numbering_gaps"] or diagnostics["insertion_code_indices"]:
            raise ValueError("Numbering gaps/insertion codes require an explicit audited mapping; unsupported here")
        if mapping["native_sequence"] != parent["reference_sequence"]:
            raise ValueError("Reference does not exactly align to complete PDB chain")
        parent["mapping"] = mapping["residue_mapping"]
        parent["coordinates"] = [list(r.coordinates) for r in residues]
        seq = parent["reference_sequence"]
        if seq in sequence_splits and sequence_splits[seq] != parent["split"]:
            raise ValueError("Identical reference crosses splits")
        sequence_splits[seq] = parent["split"]
    seen, pairs = set(), []
    for raw in data["pairs"]:
        row = dict(raw)
        parent = parents[row["parent_id"]]
        ref, var = row["reference_sequence"], row["variant_sequence"]
        measured_swap = row.get("pair_kind") == "measured_swap"
        valid_reference = ref == parent["reference_sequence"] or (
            measured_swap and len(ref) == len(parent["reference_sequence"]) and Counter(ref) == Counter(var)
            and isinstance(row.get("measurement_id"), dict)
            and row["measurement_id"].get("reference") and row["measurement_id"].get("variant"))
        if not valid_reference or len(var) != len(ref) or not var or set(var + ref) - set("ACDEFGHIKLMNPQRSTVWY"):
            raise ValueError("Unsupported sequence/alignment; fixed-backbone substitutions only")
        if not row.get("measurement_id") or row.get("label_kind") != "experimental_ddg":
            raise ValueError("Only traceable experimental ddG labels allowed")
        key = (row["parent_id"], ref, var)
        if row["pair_id"] in seen or key in seen:
            raise ValueError("Duplicate pair; consolidate measured replicates upstream")
        seen.update((row["pair_id"], key))
        for seq in (ref, var):
            if seq in sequence_splits and sequence_splits[seq] != parent["split"]:
                raise ValueError("Identical sequence crosses splits")
            sequence_splits[seq] = parent["split"]
        row.update(target=normalize_ddg(row["ddg"], data["sign_convention"], data["units"]),
                   split=parent["split"], is_swap=ref != var and Counter(ref) == Counter(var))
        pairs.append(row)
    if not pairs:
        raise ValueError("No experimental pairs")
    return data, parents, pairs


def backbone_features(parent, device):
    import torch
    length = len(parent["reference_sequence"])
    # Reuse tied_featurize's single-chain sequential-index convention.
    ones = torch.ones(1, length, device=device)
    return {"X": torch.tensor([parent["coordinates"]], dtype=torch.float32, device=device),
            "mask": ones, "chain_M": ones, "chain_encoding_all": ones.long(),
            "residue_idx": torch.arange(length, device=device)[None]}

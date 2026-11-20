"""Prepare audited MegaScale measurements, parent splits and labelled pairs; no training.

Run from the repository root. Requires RapidFuzz (C++ global edit-distance audit).
All raw tables stay inside their ZIPs; only verified aligned structures are staged.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import random
import re
import sqlite3
import sys
import tempfile
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baseline.run_baseline import sha256, stage_backbone
from patches.starter import parse_pdb

CANONICAL = set("ACDEFGHIKLMNPQRSTVWY")
MUTATION = re.compile(r"([ACDEFGHIKLMNPQRSTVWY])(\d+)([ACDEFGHIKLMNPQRSTVWY])")
MEMBER = "Processed_K50_dG_datasets/Tsuboyama2023_Dataset2_Dataset3_20230416.csv"
POLICY = {
    "version": "megascale_supervised_v1",
    "source_filter": "require finite numeric dG_ML AND ddG_ML; never recover filtered labels from raw deltaG",
    "sequence_filter": "canonical aa_seq; one/two substitutions verified against WT sequence and mutation name",
    "replicates": "arithmetic mean dG_ML per WT_name/aa_seq; retain every row/name and replicate spread",
    "reference": "unique WT sequence with at least one accepted measured WT row",
    "target": "mean measured dG_ML(reference) - mean measured dG_ML(variant)",
    "units": "kcal/mol", "sign_convention": "positive_destabilizing",
    "uncertainty": "retain source intervals; no significance-based target filtering or calibrated error claims",
    "structures": "released AlphaFold models, exact full-chain sequence and complete N/CA/C/O; no gaps or insertions",
}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def finite(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (ValueError, TypeError):
        return None


def changes(reference, sequence):
    if len(reference) != len(sequence) or set(sequence) - CANONICAL:
        return None
    return [(i + 1, a, b) for i, (a, b) in enumerate(zip(reference, sequence)) if a != b]


def valid_mutation_name(name, parent, reference, sequence):
    observed = changes(reference, sequence)
    if observed is None or not 1 <= len(observed) <= 2 or not name.startswith(parent):
        return False
    suffix = name[len(parent):]
    scan = re.fullmatch(r"_dmutv\d+_(\d+[ACDEFGHIKLMNPQRSTVWY](?::\d+[ACDEFGHIKLMNPQRSTVWY])+)_([^_]+)", suffix)
    if scan:
        # The scan identifier is metadata, not a mutation. Check its WT identities too.
        sites = re.findall(r"(\d+)([ACDEFGHIKLMNPQRSTVWY])", scan[1])
        if any(not 1 <= int(position) <= len(reference) or reference[int(position) - 1] != aa for position, aa in sites):
            return False
        suffix = scan[2]
    network = re.fullmatch(r"_hnet\d+_([^_]+)", suffix)
    if network:
        suffix = network[1]
    # Full suffix match forbids ins/del annotations and undeclared changes.
    tokens = list(MUTATION.finditer(suffix))
    if any(not 1 <= int(m[2]) <= len(reference) or reference[int(m[2]) - 1] != m[1]
           or sequence[int(m[2]) - 1] != m[3] for m in tokens):
        return False
    remainder = MUTATION.sub("", suffix).replace("_", "").replace(":", "")
    declared = [(int(m[2]), m[1], m[3]) for m in tokens if m[1] != m[3]]
    return not remainder and sorted(declared) == observed


def stage1(raw, output, connection):
    """Stream filtered labels to SQLite, then consolidate actual repeated measurements."""
    connection.executescript("""
        CREATE TABLE raw(parent TEXT, sequence TEXT, name TEXT, row_number INTEGER,
                         dg REAL, source_ddg REAL, interval_low REAL, interval_high REAL, is_wt INTEGER);
        CREATE TABLE measurements(parent TEXT, sequence TEXT, dg REAL, n INTEGER,
                         minimum REAL, maximum REAL, provenance TEXT, PRIMARY KEY(parent,sequence));
    """)
    counters = Counter()
    with zipfile.ZipFile(raw / "Processed_K50_dG_datasets.zip") as archive:
        with archive.open(MEMBER) as stream:
            reader = csv.DictReader(io.TextIOWrapper(stream, encoding="utf-8-sig"))
            batch = []
            for number, row in enumerate(reader, 2):
                counters["source_rows"] += 1
                dg, ddg = finite(row["dG_ML"]), finite(row["ddG_ML"])
                if dg is None or ddg is None:
                    counters["excluded_filtered_censored_or_nonfinite"] += 1
                    continue
                sequence, parent, name = row["aa_seq"], row["WT_name"], row["name"]
                if not parent or not sequence or set(sequence) - CANONICAL:
                    counters["excluded_noncanonical_or_missing_parent"] += 1
                    continue
                batch.append((parent, sequence, name, number, dg, ddg,
                              finite(row["deltaG_95CI_low"]), finite(row["deltaG_95CI_high"]),
                              int(row["mut_type"].lower() == "wt")))
                if len(batch) >= 10000:
                    connection.executemany("INSERT INTO raw VALUES (?,?,?,?,?,?,?,?,?)", batch)
                    batch.clear()
            connection.executemany("INSERT INTO raw VALUES (?,?,?,?,?,?,?,?,?)", batch)
    connection.execute("CREATE INDEX raw_parent_sequence ON raw(parent,sequence)")
    connection.commit()
    references, rejected = {}, []
    for parent, in connection.execute("SELECT DISTINCT parent FROM raw ORDER BY parent"):
        sequences = [r[0] for r in connection.execute("SELECT DISTINCT sequence FROM raw WHERE parent=? AND is_wt=1", (parent,))]
        if len(sequences) != 1:
            rejected.append({"parent_id": parent, "reason": "missing_or_conflicting_measured_WT"})
            continue
        reference = sequences[0]
        references[parent] = reference
        groups = connection.execute("SELECT sequence FROM raw WHERE parent=? GROUP BY sequence ORDER BY sequence", (parent,)).fetchall()
        for sequence, in groups:
            records = connection.execute("SELECT name,row_number,dg,source_ddg,interval_low,interval_high,is_wt FROM raw WHERE parent=? AND sequence=? ORDER BY row_number", (parent, sequence)).fetchall()
            accepted = [r for r in records if (r[6] and sequence == reference) or
                        (not r[6] and valid_mutation_name(r[0], parent, reference, sequence))]
            counters["excluded_mutation_annotation_or_not_single_double"] += len(records) - len(accepted)
            if not accepted:
                continue
            values = [r[2] for r in accepted]
            provenance = [{"name": r[0], "csv_row": r[1], "dG_ML": r[2], "ddG_ML": r[3],
                           "interval_low": r[4], "interval_high": r[5]} for r in accepted]
            connection.execute("INSERT INTO measurements VALUES (?,?,?,?,?,?,?)",
                               (parent, sequence, sum(values) / len(values), len(values), min(values), max(values), json.dumps(provenance)))
            counters["accepted_source_rows"] += len(accepted)
            counters["consolidated_sequences"] += 1
    connection.commit()
    counters["parents_with_measured_reference"] = len(references)
    write_json(output / "stage1_labels.json", {"policy": POLICY, "counts": dict(counters), "rejected_parents": rejected})
    print("stage1", json.dumps(dict(counters)), flush=True)
    return references


def map_structures(raw, output, references):
    by_sequence = defaultdict(list)
    for parent, sequence in references.items():
        by_sequence[sequence].append(parent)
    matched, issues = {}, []
    structures = output / "structures"
    structures.mkdir()
    with zipfile.ZipFile(raw / "AlphaFold_model_PDBs.zip") as archive, tempfile.TemporaryDirectory() as temporary:
        source, staged = Path(temporary) / "input.pdb", Path(temporary) / "staged.pdb"
        for member in sorted(archive.namelist()):
            if not member.endswith(".pdb") or member.startswith("__MACOSX/"):
                continue
            text = archive.read(member).decode("utf-8")
            source.write_text(text, encoding="utf-8")
            chains = sorted({line[21:22] for line in text.splitlines() if line.startswith("ATOM  ")})
            for chain in chains:
                try:
                    metadata = stage_backbone(source, chain, staged)
                    _, diagnostics = parse_pdb(staged.read_text(), chain)
                    if diagnostics["numbering_gaps"] or diagnostics["insertion_code_indices"]:
                        raise ValueError("numbering gaps or insertion codes")
                except ValueError as error:
                    issues.append({"member": member, "chain": chain, "reason": str(error)})
                    continue
                sequence = metadata["native_sequence"]
                for parent in by_sequence.get(sequence, []):
                    if parent in matched:
                        continue
                    identifier = hashlib.sha256(parent.encode()).hexdigest()[:20]
                    destination = structures / (identifier + ".pdb")
                    destination.write_bytes(staged.read_bytes())
                    matched[parent] = {"parent_id": parent, "reference_sequence": sequence,
                                       "pdb_path": "structures/" + destination.name,
                                       "pdb_sha256": sha256(destination), "chain": chain,
                                       "structure_source": member, "structure_type": "released_AlphaFold_prediction",
                                       "residue_mapping": metadata["residue_mapping"]}
    for parent in sorted(set(references) - set(matched)):
        issues.append({"parent_id": parent, "reason": "no_exact_complete_reference_backbone"})
    return matched, issues


def group_parents(parents, connection, threshold=0.4, seed=7):
    """Exhaustive global edit-similarity linkage, plus any exact measured-sequence overlaps.

    similarity = 1 - Levenshtein_distance/max(lengths); all pairs, no heuristic screen.
    Conservative sequence-similarity leakage guard, not a remote-homology claim.
    """
    import rapidfuzz
    from rapidfuzz.distance import Levenshtein
    names = sorted(parents)
    roots = {name: name for name in names}
    def find(name):
        while roots[name] != name:
            roots[name] = roots[roots[name]]
            name = roots[name]
        return name
    def union(a, b):
        a, b = find(a), find(b)
        roots[max(a, b)] = min(a, b)
    links = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            similarity = Levenshtein.normalized_similarity(parents[a]["reference_sequence"], parents[b]["reference_sequence"])
            if similarity >= threshold:
                union(a, b)
                links.append({"a": a, "b": b, "global_edit_similarity": similarity})
    # Do this before split assignment and pair generation, including variants.
    owners, overlap_links = {}, []
    for name in names:
        for sequence, in connection.execute("SELECT sequence FROM measurements WHERE parent=?", (name,)):
            if sequence in owners and find(name) != find(owners[sequence]):
                union(name, owners[sequence])
                overlap_links.append({"a": name, "b": owners[sequence], "sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest()})
            owners.setdefault(sequence, name)
    groups = defaultdict(list)
    for name in names:
        groups[find(name)].append(name)
    if len(groups) < 3:
        raise ValueError("Fewer than three independent groups: cannot create three honest partitions")
    held = {"train": 0, "validation": 0, "test": 0}
    targets = {s: len(names) * f for s, f in (("train", .7), ("validation", .15), ("test", .15))}
    rng = random.Random(seed)
    ordered = sorted(groups.items(), key=lambda item: (-len(item[1]), item[0]))
    for index, (root, members) in enumerate(ordered):
        empty = [s for s in held if not held[s]]
        candidates = empty if len(ordered) - index == len(empty) else list(held)
        gap = max(targets[s] - held[s] for s in candidates)
        split = rng.choice(sorted(s for s in candidates if targets[s] - held[s] == gap))
        for name in members:
            parents[name].update(homology_group=root, split=split)
        held[split] += len(members)
    return {"method": "exhaustive global Levenshtein similarity connected components plus exact measured-sequence overlaps",
            "parameters": {"minimum_similarity": threshold, "normalization": "1-distance/max_length", "seed": seed,
                           "split_fractions": [0.7, 0.15, 0.15], "rapidfuzz_version": rapidfuzz.__version__},
            "audit_source": "stage2_splits.json",
            "limitations": "sequence-similarity guard; remote homology below threshold is not certified absent",
            "groups": dict(groups), "links": links, "measured_sequence_overlap_links": overlap_links,
            "parent_counts": held, "all_parent_pairs_checked": len(names) * (len(names) - 1) // 2}


def measurement(connection, parent, sequence):
    row = connection.execute("SELECT dg,n,minimum,maximum,provenance FROM measurements WHERE parent=? AND sequence=?", (parent, sequence)).fetchone()
    return {"sequence": sequence, "dG_ML": row[0], "replicates": row[1], "range": [row[2], row[3]], "source_rows": json.loads(row[4])}


def pair_record(parent, reference, variant, kind):
    identity = "|".join((parent, reference["sequence"], variant["sequence"]))
    return {"pair_id": hashlib.sha256(identity.encode()).hexdigest(), "parent_id": parent,
            "reference_sequence": reference["sequence"], "variant_sequence": variant["sequence"],
            "pair_kind": kind, "label_kind": "experimental_ddg",
            "measurement_id": {"reference": [r["csv_row"] for r in reference["source_rows"]],
                               "variant": [r["csv_row"] for r in variant["source_rows"]]},
            "ddg": reference["dG_ML"] - variant["dG_ML"],
            "reference_dG_ML": reference["dG_ML"], "variant_dG_ML": variant["dG_ML"]}


def stage3(output, parents, homology, connection):
    pairs, counts = [], Counter()
    for name, parent in sorted(parents.items()):
        reference = parent["reference_sequence"]
        wt = measurement(connection, name, reference)
        records = {s: measurement(connection, name, s) for s, in connection.execute("SELECT sequence FROM measurements WHERE parent=?", (name,))}
        for sequence, record in sorted(records.items()):
            if sequence == reference:
                continue
            pairs.append(pair_record(name, wt, record, "reference_variant"))
            counts[parent["split"] + "_reference_variant"] += 1
        # Each sequence has at most two substitutions. Reverse just those two sites;
        # emit only if the entire reverse sequence is experimentally present.
        emitted = set()
        for sequence, record in sorted(records.items()):
            changed = changes(reference, sequence)
            if len(changed) != 2:
                continue
            i, j = changed[0][0] - 1, changed[1][0] - 1
            if sequence[i] == sequence[j]:
                continue
            reverse = list(sequence)
            reverse[i], reverse[j] = reverse[j], reverse[i]
            reverse = "".join(reverse)
            if reverse == reference or reverse not in records:
                continue  # WT-to-swap pairs already emitted above; unordered pair once.
            key = tuple(sorted((sequence, reverse)))
            if key in emitted:
                continue
            emitted.add(key)
            pairs.append(pair_record(name, records[key[0]], records[key[1]], "measured_swap"))
            counts[parent["split"] + "_measured_swap"] += 1
    manifest = {"schema_version": 1, "label_source": "Tsuboyama2023 Zenodo 7992926 v2_230420; " + MEMBER,
                "quality_policy": POLICY, "units": "kcal/mol", "sign_convention": "positive_destabilizing",
                "homology": {k: homology[k] for k in ("method", "parameters", "audit_source", "limitations")},
                "parents": list(parents.values()), "pairs": pairs}
    write_json(output / "stability_pairs.json", manifest)
    counts["total_pairs"] = len(pairs)
    counts["composition_preserving_pairs"] = sum(Counter(p["reference_sequence"]) == Counter(p["variant_sequence"]) for p in pairs)
    write_json(output / "stage3_manifest.json", {"counts": dict(counts), "manifest_sha256": sha256(output / "stability_pairs.json")})
    print("stage3", json.dumps(dict(counts)), flush=True)
    return dict(counts)


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source_manifest = json.loads(Path("Data/megascale_manifest.json").read_text())
    checksums = {}
    for name in ("Processed_K50_dG_datasets.zip", "AlphaFold_model_PDBs.zip"):
        expected = next(f["sha256"] for f in source_manifest["files"] if f["filename"] == name)
        actual = sha256(args.raw / name)
        if expected != actual:
            raise ValueError("Raw archive SHA256 mismatch: " + name)
        checksums[name] = actual
    write_json(output / "source_checksums.json", checksums)
    connection = sqlite3.connect(output / "measurements.sqlite")
    try:
        references = stage1(args.raw, output, connection)
        parents, issues = map_structures(args.raw, output, references)
        print("stage2 aligned", len(parents), "of", len(references), flush=True)
        homology = group_parents(parents, connection, args.similarity, args.seed)
        write_json(output / "stage2_splits.json", {**homology, "parents": list(parents.values()), "mapping_exclusions": issues})
        counts = stage3(output, parents, homology, connection)
    finally:
        connection.close()
    from scorer.data import load_dataset
    _, verified_parents, verified_pairs = load_dataset(output / "stability_pairs.json")
    summary = {"status": "prepared_and_loader_validated_not_trained", "parents": len(verified_parents),
               "groups": len(homology["groups"]), "parent_counts": homology["parent_counts"],
               "counts": counts, "validated_pairs": len(verified_pairs), "source_checksums": checksums,
               "homology_limitations": homology["limitations"]}
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def audit_existing(output):
    """Explain eligible mutation counts and exclusions without altering prepared labels."""
    connection = sqlite3.connect(output / "measurements.sqlite")
    references = dict(connection.execute("SELECT parent,sequence FROM raw WHERE is_wt=1 GROUP BY parent"))
    counts, examples = Counter(), []
    for parent, sequence, name in connection.execute("SELECT parent,sequence,name FROM raw WHERE is_wt=0"):
        delta = changes(references[parent], sequence)
        counts["filtered_source_substitutions_" + str(None if delta is None else len(delta))] += 1
        if delta is not None and len(delta) in (1, 2) and "ins" not in name and "del" not in name and not valid_mutation_name(name, parent, references[parent], sequence) and len(examples) < 12:
            examples.append({"parent": parent, "name": name, "changes": delta,
                             "annotation_valid": valid_mutation_name(name, parent, references[parent], sequence)})
    for parent, sequence in connection.execute("SELECT parent,sequence FROM measurements"):
        counts["accepted_substitutions_" + str(len(changes(references[parent], sequence)))] += 1
    connection.close()
    audit = {"counts": dict(counts), "double_mutant_examples": examples}
    write_json(output / "mutation_audit.json", audit)
    print(json.dumps(audit, indent=2), flush=True)


def rebuild_pairs(output):
    """Regenerate only measured pair definitions, preserving frozen parent splits."""
    homology = json.loads((output / "stage2_splits.json").read_text())
    parents = {p["parent_id"]: p for p in homology["parents"]}
    connection = sqlite3.connect(output / "measurements.sqlite")
    try:
        counts = stage3(output, parents, homology, connection)
    finally:
        connection.close()
    from scorer.data import load_dataset
    _, verified_parents, verified_pairs = load_dataset(output / "stability_pairs.json")
    summary = json.loads((output / "summary.json").read_text())
    summary.update(counts=counts, validated_pairs=len(verified_pairs), parents=len(verified_parents))
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("Data/raw/megascale"))
    parser.add_argument("--output", type=Path, default=Path("Data/processed/vani_training_v2"))
    parser.add_argument("--similarity", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--audit-existing", action="store_true")
    parser.add_argument("--rebuild-pairs", action="store_true")
    args = parser.parse_args()
    if not 0 < args.similarity <= 1:
        parser.error("similarity must be in (0,1]")
    if args.rebuild_pairs:
        rebuild_pairs(args.output)
    elif args.audit_existing:
        audit_existing(args.output)
    else:
        run(args)

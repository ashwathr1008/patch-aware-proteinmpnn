"""Validate controlled patch-pair scores and summarize by source protein."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

TIE_TOLERANCE = 1e-6
ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
SCORE_FIELDS = "pair_id structure_id chain split group patch_id model_type model_name order_index order_sha256 length patch_size changed_identity_count presentation_first original_full_nll permuted_full_nll full_nll_gap original_patch_nll permuted_patch_nll patch_nll_gap".split()
PAIR_FIELDS = "pair_id structure_id chain split group patch_id model_type model_name length patch_size changed_identity_count order_count original_full_nll_mean permuted_full_nll_mean full_nll_gap_mean full_nll_gap_min full_nll_gap_max full_nll_gap_std native_preference native_preference_across_orders order_sign_disagreement original_patch_nll_mean permuted_patch_nll_mean patch_nll_gap_mean hydrophobic_fraction_delta".split()
PROTEIN_FIELDS = "structure_id chain split group model_type model_name pair_count patch_count orders_per_pair full_nll_gap_mean full_nll_gap_min full_nll_gap_max native_preferred_count tie_count permuted_preferred_count native_preference_fraction order_sign_disagreement_pair_count mean_order_gap_range patch_nll_gap_mean".split()
CONTROL_FIELDS = "pair_id structure_id chain split group patch_id patch_size changed_identity_count patch_contiguous_in_sequence composition_tie geometry_tie contact_count hydrophobic_contacts_original hydrophobic_contacts_permuted hydrophobic_fraction_original hydrophobic_fraction_permuted hydrophobic_fraction_delta".split()
GROUP_FIELDS = "group split model_type model_name protein_count pair_count patch_count equal_protein_mean_full_nll_gap equal_protein_native_preference_fraction".split()
SPLIT_FIELDS = "split model_type model_name protein_count source_group_count pair_count patch_count equal_protein_mean_full_nll_gap protein_mean_gap_min protein_mean_gap_max equal_protein_native_preference_fraction equal_group_mean_full_nll_gap equal_group_native_preference_fraction".split()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite(value, name):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Nonfinite " + name)
    return number


def subset(data, names):
    return {key: data[key] for key in names if key in data}


def write_csv(path, fields, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def preference(gap):
    return 1.0 if gap > TIE_TOLERANCE else 0.0 if gap < -TIE_TOLERANCE else 0.5


def validate_pairs(dataset):
    if dataset.get("schema_version") != 1 or dataset.get("purpose") != "patch_compatibility_pilot":
        raise ValueError("Expected a schema-1 patch compatibility dataset")
    structures = {(s["structure_id"], s["chain"]): s for s in dataset["structures"]}
    if len(structures) != len(dataset["structures"]):
        raise ValueError("Duplicate source structure")
    pairs, controls = {}, []
    for pair in dataset["pairs"]:
        identifier = pair["pair_id"]
        if identifier in pairs:
            raise ValueError("Duplicate pair identifier")
        key = (pair["structure_id"], pair["chain"])
        if key not in structures:
            raise ValueError("Pair references an unknown structure")
        structure = structures[key]
        original, permuted = pair["original_sequence"], pair["permuted_sequence"]
        if not original or len(original) != len(permuted) or not set(original + permuted) <= set(ALPHABET):
            raise ValueError("Invalid canonical pair sequences")
        indices = pair["patch_indices"]
        if not indices or len(set(indices)) != len(indices) or any(type(i) is not int or not 1 <= i <= len(original) for i in indices):
            raise ValueError("Invalid one-based patch indices")
        members = set(indices)
        changed = sum(a != b for a, b in zip(original, permuted))
        if not changed or any(a != b for i, (a, b) in enumerate(zip(original, permuted), 1) if i not in members):
            raise ValueError("Pair must change only patch identities")
        counts = [Counter(sequence[i - 1] for i in indices) for sequence in (original, permuted)]
        if counts[0] != counts[1] or Counter(original) != Counter(permuted):
            raise ValueError("Composition control failed")
        raw = pair.get("controls", {})
        if raw.get("composition_tie") is not True or raw.get("geometry_tie") is not True:
            raise ValueError("Pair must explicitly report composition and shared-geometry ties")
        if pair.get("checks") and any(value is not True for value in pair["checks"].values()):
            raise ValueError("Pair reports a failed invariant")
        for label, count in zip(("original", "permuted"), counts):
            declared = raw.get("composition_" + label)
            if declared is not None and any(declared.get(aa, 0) != count[aa] for aa in ALPHABET):
                raise ValueError("Declared composition counts differ from sequences")
        for field, expected in (("patch_size", len(indices)), ("changed_identity_count", changed)):
            if field in pair and pair[field] != expected:
                raise ValueError("Declared " + field + " is incorrect")
        control = {"pair_id": identifier, "structure_id": key[0], "chain": key[1],
                   "split": structure["split"], "group": structure["group"], "patch_id": pair["patch_id"],
                   "patch_size": len(indices), "changed_identity_count": changed,
                   "patch_contiguous_in_sequence": max(indices) - min(indices) + 1 == len(indices),
                   "composition_tie": True, "geometry_tie": True}
        for field in CONTROL_FIELDS[11:]:
            control[field] = raw.get(field)
        contact_count = raw.get("contact_count")
        if contact_count is not None:
            if type(contact_count) is not int or contact_count < 0:
                raise ValueError("Invalid spatial contact count")
            for label in ("original", "permuted"):
                contacts, fraction = raw["hydrophobic_contacts_" + label], raw["hydrophobic_fraction_" + label]
                if type(contacts) is not int or not 0 <= contacts <= contact_count:
                    raise ValueError("Invalid hydrophobic contact count")
                if contact_count:
                    if not math.isclose(finite(fraction, "hydrophobic fraction"), contacts / contact_count, rel_tol=0, abs_tol=1e-12):
                        raise ValueError("Incorrect hydrophobic fraction")
                elif fraction is not None:
                    raise ValueError("Zero-contact patch must have undefined fraction")
            delta = raw.get("hydrophobic_fraction_delta")
            if contact_count:
                expected = raw["hydrophobic_fraction_permuted"] - raw["hydrophobic_fraction_original"]
                if not math.isclose(finite(delta, "hydrophobic delta"), expected, rel_tol=0, abs_tol=1e-12):
                    raise ValueError("Incorrect descriptor delta")
            elif delta is not None:
                raise ValueError("Zero-contact patch must have undefined descriptor delta")
        elif any(raw.get(k) is not None for k in CONTROL_FIELDS[12:]):
            raise ValueError("Descriptor values require a contact count")
        pairs[identifier] = {**pair, "split": structure["split"], "group": structure["group"],
                             "length": len(original), "patch_size": len(indices), "changed_identity_count": changed}
        controls.append(control)
    if not pairs:
        raise ValueError("Dataset has no pairs")
    return structures, pairs, controls


def summarize_rows(rows, dataset, models, orders_per_pair, model_name="v_48_020"):
    """Preserve pair, order and source protein as distinct units."""
    _, pairs, controls = validate_pairs(dataset)
    if not models or len(set(models)) != len(models) or any(m not in ("vanilla", "soluble") for m in models):
        raise ValueError("Unsupported model selection")
    if type(orders_per_pair) is not int or orders_per_pair < 1:
        raise ValueError("Orders per pair must be positive")
    expected = {(pair_id, model, order) for pair_id in pairs for model in models for order in range(1, orders_per_pair + 1)}
    seen, grouped, shared_orders, checked = set(), defaultdict(list), {}, []
    numeric_fields = SCORE_FIELDS[14:]
    for raw in rows:
        if not set(SCORE_FIELDS) <= set(raw):
            raise ValueError("Missing per-order score fields")
        row = dict(raw)
        row["order_index"] = int(row["order_index"])
        key = (row["pair_id"], row["model_type"], row["order_index"])
        if key in seen or key not in expected:
            raise ValueError("Duplicate or unexpected pair/model/order")
        seen.add(key)
        pair = pairs[row["pair_id"]]
        for field in ("structure_id", "chain", "split", "group", "patch_id"):
            if row[field] != pair[field]:
                raise ValueError("Score metadata differs from pair: " + field)
        for field in ("length", "patch_size", "changed_identity_count"):
            row[field] = int(row[field])
            if row[field] != pair[field]:
                raise ValueError("Score count differs from pair: " + field)
        if row["model_name"] != model_name:
            raise ValueError("Unexpected checkpoint name")
        order_hash = row["order_sha256"]
        if len(order_hash) != 64 or any(c not in "0123456789abcdef" for c in order_hash):
            raise ValueError("Invalid order hash")
        order_key = (row["structure_id"], row["chain"], row["order_index"])
        if shared_orders.setdefault(order_key, order_hash) != order_hash:
            raise ValueError("Decoding order differs across pairs or checkpoints")
        if row["presentation_first"] not in ("original", "permuted"):
            raise ValueError("Unknown presentation order")
        for field in numeric_fields:
            row[field] = finite(row[field], field)
            if "gap" not in field and row[field] < 0:
                raise ValueError("Negative NLL")
        for region in ("full", "patch"):
            difference = row["permuted_" + region + "_nll"] - row["original_" + region + "_nll"]
            if not math.isclose(row[region + "_nll_gap"], difference, rel_tol=1e-6, abs_tol=1e-8):
                raise ValueError("Incorrect signed NLL gap")
        checked.append(row)
        grouped[key[:2]].append(row)
    if seen != expected:
        raise ValueError("Incomplete pair/model/order coverage")
    pair_summary = []
    for (pair_id, model), values in sorted(grouped.items()):
        pair = pairs[pair_id]
        gaps = [r["full_nll_gap"] for r in values]
        summary = subset(pair, ("pair_id", "structure_id", "chain", "split", "group", "patch_id", "length", "patch_size", "changed_identity_count"))
        summary.update({"model_type": model, "model_name": model_name, "order_count": orders_per_pair,
                        "full_nll_gap_mean": statistics.mean(gaps), "full_nll_gap_min": min(gaps),
                        "full_nll_gap_max": max(gaps), "full_nll_gap_std": statistics.pstdev(gaps),
                        "native_preference": preference(statistics.mean(gaps)),
                        "native_preference_across_orders": statistics.mean(preference(g) for g in gaps),
                        "order_sign_disagreement": min(gaps) < -TIE_TOLERANCE and max(gaps) > TIE_TOLERANCE,
                        "hydrophobic_fraction_delta": pair["controls"].get("hydrophobic_fraction_delta")})
        for region in ("full", "patch"):
            for label in ("original", "permuted"):
                field = label + "_" + region + "_nll"
                summary[field + "_mean"] = statistics.mean(r[field] for r in values)
        summary["patch_nll_gap_mean"] = statistics.mean(r["patch_nll_gap"] for r in values)
        pair_summary.append(summary)
    protein_groups = defaultdict(list)
    for row in pair_summary:
        protein_groups[(row["structure_id"], row["chain"], row["model_type"])].append(row)
    protein_summary = []
    for key, values in sorted(protein_groups.items()):
        gaps = [r["full_nll_gap_mean"] for r in values]
        summary = subset(values[0], ("structure_id", "chain", "split", "group", "model_type", "model_name"))
        summary.update({"pair_count": len(values), "patch_count": len({r["patch_id"] for r in values}),
                        "orders_per_pair": orders_per_pair, "full_nll_gap_mean": statistics.mean(gaps),
                        "full_nll_gap_min": min(gaps), "full_nll_gap_max": max(gaps),
                        "native_preferred_count": sum(preference(g) == 1 for g in gaps),
                        "tie_count": sum(preference(g) == 0.5 for g in gaps),
                        "permuted_preferred_count": sum(preference(g) == 0 for g in gaps),
                        "native_preference_fraction": statistics.mean(preference(g) for g in gaps),
                        "order_sign_disagreement_pair_count": sum(r["order_sign_disagreement"] for r in values),
                        "mean_order_gap_range": statistics.mean(r["full_nll_gap_max"] - r["full_nll_gap_min"] for r in values),
                        "patch_nll_gap_mean": statistics.mean(r["patch_nll_gap_mean"] for r in values)})
        protein_summary.append(summary)
    group_groups = defaultdict(list)
    for row in protein_summary:
        group_groups[(row["group"], row["model_type"])].append(row)
    group_summary = []
    for key, values in sorted(group_groups.items()):
        if len({r["split"] for r in values}) != 1:
            raise ValueError("A source group crosses development and diagnostic splits")
        group_summary.append({"group": key[0], "split": values[0]["split"], "model_type": key[1], "model_name": model_name,
                              "protein_count": len(values), "pair_count": sum(r["pair_count"] for r in values),
                              "patch_count": sum(r["patch_count"] for r in values),
                              "equal_protein_mean_full_nll_gap": statistics.mean(r["full_nll_gap_mean"] for r in values),
                              "equal_protein_native_preference_fraction": statistics.mean(r["native_preference_fraction"] for r in values)})
    split_groups = defaultdict(list)
    for row in protein_summary:
        split_groups[(row["split"], row["model_type"])].append(row)
    split_summary = []
    for key, values in sorted(split_groups.items()):
        group_values = [r for r in group_summary if (r["split"], r["model_type"]) == key]
        split_summary.append({"split": key[0], "model_type": key[1], "model_name": model_name,
            "protein_count": len(values), "source_group_count": len({r["group"] for r in values}),
            "pair_count": sum(r["pair_count"] for r in values), "patch_count": sum(r["patch_count"] for r in values),
            "equal_protein_mean_full_nll_gap": statistics.mean(r["full_nll_gap_mean"] for r in values),
            "protein_mean_gap_min": min(r["full_nll_gap_mean"] for r in values),
            "protein_mean_gap_max": max(r["full_nll_gap_mean"] for r in values),
            "equal_protein_native_preference_fraction": statistics.mean(r["native_preference_fraction"] for r in values),
            "equal_group_mean_full_nll_gap": statistics.mean(r["equal_protein_mean_full_nll_gap"] for r in group_values),
            "equal_group_native_preference_fraction": statistics.mean(r["equal_protein_native_preference_fraction"] for r in group_values)})
    checked.sort(key=lambda r: (r["pair_id"], r["model_type"], r["order_index"]))
    return checked, pair_summary, protein_summary, group_summary, split_summary, controls


def collect_run(run_dir, pairs_path):
    run_dir, pairs_path = Path(run_dir), Path(pairs_path)
    summary = read_json(run_dir / "metadata/summary.json")
    config = read_json(run_dir / "metadata/config.json")
    if summary.get("purpose") != "patch_compatibility_pilot" or summary.get("status") != "complete":
        raise ValueError("Analysis requires a completed patch compatibility run")
    if sha256(pairs_path) != config.get("pairs_sha256"):
        raise ValueError("Input pairs checksum differs from scoring configuration")
    for name in ("pair_scores", "pair_summary"):
        if sha256(run_dir / (name + ".csv")) != summary.get(name + "_sha256"):
            raise ValueError(name + " checksum mismatch")
    if config.get("backbone_noise") != 0 or config.get("tie_tolerance") != TIE_TOLERANCE:
        raise ValueError("Unexpected scoring noise or tie tolerance")
    dataset = read_json(pairs_path)
    with (run_dir / "pair_scores.csv").open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    tables = summarize_rows(rows, dataset, config["models"], config["orders"], config["model_name"])
    checked, pair_table, protein_table, group_table, split_table, controls = tables
    used = {(r["structure_id"], r["chain"]) for r in pair_table}
    expected_rows = len(dataset["pairs"]) * len(config["models"]) * config["orders"]
    for field in ("expected_score_rows", "completed_score_rows"):
        if summary.get(field) != expected_rows or len(rows) != expected_rows:
            raise ValueError("Scoring summary has incomplete row totals")
    if (summary.get("pair_count") != len(dataset["pairs"]) or summary.get("structure_count") != len(used)
            or summary.get("order_count") != config["orders"] or summary.get("model_types") != config["models"]):
        raise ValueError("Scoring summary disagrees with input coverage")
    expected_audits = {(sid, chain, model, order) for sid, chain in used for model in config["models"]
                       for order in range(1, config["orders"] + 1)}
    found_audits = set()
    for audit in summary.get("audits", []):
        key = tuple(audit[k] for k in ("structure_id", "chain", "model_type", "order_index"))
        if key in found_audits or key not in expected_audits or audit.get("status") != "passed":
            raise ValueError("Invalid null/reversed-presentation audit")
        found_audits.add(key)
        for field in ("identical_sequence_max_abs_difference", "reversed_presentation_max_abs_difference"):
            difference = finite(audit[field], field)
            if not 0 <= difference <= TIE_TOLERANCE:
                raise ValueError("Null/reversed-presentation audit exceeded tolerance")
        derived = audit.get("explicit_vs_rank_derived_max_abs_difference")
        if audit["order_index"] == 1:
            if derived is None or not 0 <= finite(derived, "derived-order audit") <= TIE_TOLERANCE:
                raise ValueError("Explicit/rank-derived order audit missing or failed")
    if found_audits != expected_audits:
        raise ValueError("Missing null/reversed-presentation audit")
    calls = summary.get("model_forward_calls")
    observed = summary.get("order_audit", {}).get("verified_forward_calls")
    if type(calls) is not int or calls < 2 * expected_rows or observed != calls:
        raise ValueError("Actual decoding-order audit does not cover every model call")
    order_records = read_json(run_dir / "metadata/orders.json")
    order_map, seen_structures = {}, set()
    lengths = {(r["structure_id"], r["chain"]): r["length"] for r in pair_table}
    for structure in order_records:
        key = (structure["structure_id"], structure["chain"])
        if key in seen_structures or key not in used:
            raise ValueError("Duplicate or unknown structure in order metadata")
        seen_structures.add(key)
        for order in structure["orders"]:
            order_key = key + (order["order_index"],)
            if order_key in order_map or sorted(order["decoding_order_one_based"]) != list(range(1, lengths[key] + 1)):
                raise ValueError("Invalid stored decoding permutation")
            actual_hash = hashlib.sha256(json.dumps(order["decoding_order_one_based"], separators=(",", ":")).encode("ascii")).hexdigest()
            if actual_hash != order["order_sha256"]:
                raise ValueError("Stored decoding permutation hash mismatch")
            order_map[order_key] = order["order_sha256"]
    expected_orders = {(sid, chain, order) for sid, chain in used for order in range(1, config["orders"] + 1)}
    if set(order_map) != expected_orders:
        raise ValueError("Stored decoding-order coverage is incomplete")
    for row in checked:
        if order_map[(row["structure_id"], row["chain"], row["order_index"])] != row["order_sha256"]:
            raise ValueError("Scored decoding order differs from stored order")
    record = portable_record(run_dir, summary, config, controls, used, calls)
    return tables, record, order_records


def portable_record(run_dir, summary, config, controls, used, calls):
    environment = read_json(run_dir / "metadata/environment.json")
    structures = read_json(run_dir / "metadata/structures.json")
    return {
        "schema_version": 1, "purpose": "patch_compatibility_pilot", "source_run_status": "complete",
        "started_utc": summary["started_utc"], "finished_utc": summary["finished_utc"],
        "counts": {"source_protein_chains": len(used), "source_groups": len({r["group"] for r in controls}),
                   "patches": len({(r["structure_id"], r["chain"], r["patch_id"]) for r in controls}),
                   "pairs": len(controls), "checkpoints": len(config["models"]),
                   "orders_per_pair": config["orders"], "per_order_score_rows": summary["completed_score_rows"],
                   "model_forward_calls_including_audits": calls,
                   "composition_ties": sum(r["composition_tie"] for r in controls),
                   "shared_geometry_ties": sum(r["geometry_tie"] for r in controls),
                   "pairs_with_defined_spatial_descriptor": sum(r["hydrophobic_fraction_delta"] is not None for r in controls)},
        "config": subset(config, ("pairs_sha256", "upstream_commit", "model_name", "models", "orders",
                                   "order_seed", "tie_tolerance", "backbone_noise", "device", "script_sha256", "baseline_helper_sha256", "upstream_files",
                                   "score_definition", "patch_score_definition", "order_average_definition", "outside_patch_policy", "inference_temperature")),
        "environment": subset(environment, ("python", "platform", "torch", "numpy", "device_requested",
                                             "torch_cuda_version", "gpu", "gpu_compute_capability", "preflight")),
        "structures": [subset(s, ("structure_id", "chain", "length", "split", "group", "raw_sha256", "staged_sha256", "preprocessing")) for s in structures],
        "elapsed_seconds": finite(summary["elapsed_seconds"], "elapsed seconds"),
        "audits": [subset(a, ("structure_id", "chain", "model_type", "order_index", "identical_sequence_max_abs_difference",
                                "reversed_presentation_max_abs_difference", "explicit_vs_rank_derived_max_abs_difference", "status")) for a in summary["audits"]],
        "order_audit": subset(summary["order_audit"], ("method", "verified_forward_calls")),
        "source_file_sha256": {name: sha256(run_dir / name) for name in (
            "pair_scores.csv", "pair_summary.csv", "metadata/summary.json", "metadata/config.json",
            "metadata/orders.json", "metadata/structures.json", "metadata/environment.json")},
        "analysis_script_sha256": sha256(Path(__file__)),
        "definitions": {
            "primary_gap": "permuted full-chain mean NLL minus original full-chain mean NLL; positive means original preferred",
            "pair_summary": "Arithmetic average over the same explicit decoding orders for both pair members",
            "native_preference_fields": "Legacy names mean original-input preference, including designed scaffolds; not necessarily biological native sequence",
            "native_preference_fraction": "Fraction of pairs with positive order-averaged gap; ties within 1e-6 count as half",
            "protein_summary": "Each source protein/chain summarizes its own pairs; overlapping patches are not independent proteins",
            "group_summary": "Equal-weight average of member protein summaries within predeclared conservative source groups",
            "split_summary": "Descriptive equal-protein and equal-group means within split and checkpoint; neither is pair-pooled accuracy",
            "order_sign_disagreement": "At least one order gap is below -1e-6 and another above +1e-6",
            "order_range": "Observed min/max across fixed decoding orders; not a confidence interval",
            "patch_terms": "Mean conditional NLL terms at patch positions under full sequence context; not an isolated patch likelihood",
            "spatial_descriptor": "Fraction of within-patch nonadjacent CA contacts joining two AVILMFWY identities; delta is permuted minus original",
        },
        "limitations": [
            "This is a compatibility diagnostic on selected proteins, not a solubility or aggregation measurement.",
            "Original preference does not establish that a permuted protein is nonfunctional or fails to fold.",
            "No statistical independence is claimed for overlapping patches, variants or decoding orders.",
            "Fixed diagnostic inputs do not establish exclusion from ProteinMPNN training.",
            "A changed hydrophobic descriptor has no validated favorable direction in this experiment.",
            "No inferential confidence intervals, fitted predictor or biological improvement claims are reported.",
        ],
    }


def plot_results(pair_table, protein_table, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    models = sorted({r["model_type"] for r in pair_table}, reverse=True)
    proteins = sorted({(r["structure_id"], r["chain"], r["split"]) for r in pair_table}, key=lambda k: (k[2], k[0], k[1]))
    colors = {key[:2]: plt.get_cmap("tab10")(i % 10) for i, key in enumerate(proteins)}
    model_colors = {"vanilla": "#0072B2", "soluble": "#D55E00"}
    artifacts = []

    def save(figure, stem, title):
        figure.savefig(output / (stem + ".png"), dpi=220, metadata={"Software": "Matplotlib " + matplotlib.__version__, "Title": title})
        figure.savefig(output / (stem + ".pdf"), metadata={"Title": title, "Creator": "summarize_pairs.py", "CreationDate": None, "ModDate": None})
        plt.close(figure)
        artifacts.extend([stem + ".png", stem + ".pdf"])

    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42}):
        figure, axes = plt.subplots(len(models), 1, figsize=(max(7.5, len(proteins) * 1.25), 2.8 * len(models) + 1), squeeze=False, sharex=True)
        for axis, model in zip(axes[:, 0], models):
            axis.axhline(0, color="#555555", linestyle="--", linewidth=1)
            for i, (sid, chain, split) in enumerate(proteins):
                selected = sorted([r for r in pair_table if (r["structure_id"], r["chain"], r["model_type"]) == (sid, chain, model)], key=lambda r: r["pair_id"])
                values = [r["full_nll_gap_mean"] for r in selected]
                shifts = [0] if len(values) == 1 else [0.35 * (j / (len(values) - 1) - 0.5) for j in range(len(values))]
                axis.scatter([i + shift for shift in shifts], values, s=25, color=model_colors[model], alpha=0.8, zorder=3)
                axis.plot([i - 0.2, i + 0.2], [statistics.mean(values)] * 2, color="#222222", linewidth=2.2)
                record = next(r for r in protein_table if (r["structure_id"], r["chain"], r["model_type"]) == (sid, chain, model))
                label = "{} pairs; {:.0%} preferred".format(record["pair_count"], record["native_preference_fraction"])
                axis.text(i, 0.97, label, transform=axis.get_xaxis_transform(), ha="center", va="top", fontsize=7)
            axis.margins(y=0.22)
            axis.set_title(model.capitalize() + " checkpoint", loc="left", fontsize=10)
            axis.set_ylabel("Full-chain NLL gap\n(permuted − original)")
            axis.grid(axis="y", alpha=0.2)
        axes[-1, 0].set_xticks(range(len(proteins)), [sid + ":" + chain + "\n" + split for sid, chain, split in proteins])
        figure.suptitle("Controlled arrangement comparison by source protein", fontsize=13)
        figure.text(0.5, 0.025, "Each point: one pair, averaged over shared orders. Bar: protein mean. Positive gap: original preferred.\nPreference counts ties as half; overlapping patches are dependent. No aggregation or folding claim.", ha="center", fontsize=8)
        figure.tight_layout(rect=(0, 0.085, 1, 0.96))
        save(figure, "paired_gaps", "Controlled patch-pair NLL gaps")

        pair_keys = sorted({(r["structure_id"], r["chain"], r["pair_id"]) for r in pair_table}, key=lambda k: (next(p[2] for p in proteins if p[:2] == k[:2]), k))
        lookup = {(r["pair_id"], r["model_type"]): r for r in pair_table}
        figure, axes = plt.subplots(len(models), 1, figsize=(max(8, len(proteins) * 1.25), 2.6 * len(models) + 1), squeeze=False, sharex=True)
        for axis, model in zip(axes[:, 0], models):
            axis.axhline(0, color="#555555", linestyle="--", linewidth=1)
            for i, key in enumerate(pair_keys):
                row = lookup[(key[2], model)]
                color = colors[key[:2]]
                axis.plot([i, i], [row["full_nll_gap_min"], row["full_nll_gap_max"]], color=color, alpha=0.8, linewidth=1)
                axis.scatter([i], [row["full_nll_gap_mean"]], color=[color], s=12, zorder=3)
            axis.set_ylabel("NLL gap")
            axis.set_title(model.capitalize() + " checkpoint", loc="left", fontsize=10)
            axis.grid(axis="y", alpha=0.2)
        tick_positions, labels = [], []
        for sid, chain, split in proteins:
            positions = [i for i, key in enumerate(pair_keys) if key[:2] == (sid, chain)]
            tick_positions.append(statistics.mean(positions))
            labels.append(sid + ":" + chain)
        axes[-1, 0].set_xticks(tick_positions, labels)
        figure.suptitle("Sensitivity to the shared decoding order", fontsize=13)
        figure.text(0.5, 0.025, "All pairs shown in identifier order. Line: observed min–max across orders; dot: mean.\nThese ranges are not confidence intervals. Positive gap: original preferred.", ha="center", fontsize=8)
        figure.tight_layout(rect=(0, 0.085, 1, 0.96))
        save(figure, "order_sensitivity", "Shared decoding-order variation")

        descriptor_rows = [r for r in pair_table if r["hydrophobic_fraction_delta"] is not None]
        if descriptor_rows:
            figure, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4.4), squeeze=False, sharey=True)
            for axis, model in zip(axes[0], models):
                axis.axhline(0, color="#999999", linestyle="--", linewidth=0.8)
                axis.axvline(0, color="#999999", linestyle="--", linewidth=0.8)
                for sid, chain, _ in proteins:
                    selected = [r for r in descriptor_rows if (r["structure_id"], r["chain"], r["model_type"]) == (sid, chain, model)]
                    axis.scatter([r["hydrophobic_fraction_delta"] for r in selected], [r["full_nll_gap_mean"] for r in selected],
                                 color=[colors[(sid, chain)]], s=27, alpha=0.75, label=sid + ":" + chain)
                axis.set_title(model.capitalize())
                axis.set_xlabel("Hydrophobic-contact fraction change\n(permuted − original)")
                axis.grid(alpha=0.15)
            axes[0, 0].set_ylabel("Order-averaged full-chain NLL gap")
            handles = [Line2D([], [], color=colors[k[:2]], marker="o", linestyle="none", label=k[0] + ":" + k[1]) for k in proteins]
            figure.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.91), ncol=min(7, len(proteins)), fontsize=8)
            figure.suptitle("Spatial descriptor and model response", fontsize=13)
            figure.text(0.5, 0.025, "Descriptive comparison only: neither descriptor direction is a validated solubility or aggregation label.\nRepeated permutations/overlapping patches share a source; points may overlap. No fitted trend.", ha="center", fontsize=8)
            figure.tight_layout(rect=(0, 0.09, 1, 0.86))
            save(figure, "spatial_descriptor", "Spatial descriptor versus controlled NLL gap")
    return {"artifacts": artifacts, "matplotlib": matplotlib.__version__, "png_dpi": 220,
            "pair_point_order": "identifier order; deterministic jitter", "intervals": "none; order ranges are observed min/max"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Analysis output must be new")
    tables, record, orders = collect_run(args.run_dir, args.pairs)
    args.output.mkdir(parents=True, exist_ok=False)
    record["analysis_status"] = "running"
    try:
        filenames = ("pair_scores.csv", "pair_summary.csv", "protein_summary.csv", "group_summary.csv", "split_summary.csv", "pair_controls.csv")
        fields = (SCORE_FIELDS, PAIR_FIELDS, PROTEIN_FIELDS, GROUP_FIELDS, SPLIT_FIELDS, CONTROL_FIELDS)
        for name, columns, rows in zip(filenames, fields, tables):
            write_csv(args.output / name, columns, rows)
        # Store the exact audited permutations; no local paths or hostnames are included.
        portable_orders = [{"structure_id": s["structure_id"], "chain": s["chain"], "orders": [subset(o, ("order_index", "order_sha256", "decoding_order_one_based")) for o in s["orders"]]} for s in orders]
        write_json(args.output / "decoding_orders.json", portable_orders)
        record["plots"] = plot_results(tables[1], tables[2], args.output)
        artifacts = list(filenames) + ["decoding_orders.json"] + record["plots"]["artifacts"]
        record["artifacts_sha256"] = {name: sha256(args.output / name) for name in artifacts}
        record["analysis_status"] = "complete"
    except BaseException as error:
        record["analysis_status"] = "failed"
        record["analysis_error_type"] = type(error).__name__
        raise
    finally:
        write_json(args.output / "run_record.json", record)
    print("Exported {} pairs across {} source protein chains and {} groups to {}".format(record["counts"]["pairs"], record["counts"]["source_protein_chains"], record["counts"]["source_groups"], args.output))


if __name__ == "__main__":
    main()

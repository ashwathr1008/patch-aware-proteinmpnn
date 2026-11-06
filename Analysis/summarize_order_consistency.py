"""Measure decoding-order inconsistency and the outside-patch joint component of each gap.

Two quantities are derived from an existing pair_scores.csv without any new forward pass.

Order inconsistency. For an autoregressive model whose conditionals come from a single
consistent joint distribution, the chain total -sum_i log p(y_i | y_before_i) is identical
for every decoding order, because a joint does not depend on its factorization. Any spread
of the chain total across shared orders is therefore amortization error in the network, not
a property of the sequence. Its scale is only interpretable next to the between-arrangement
gaps the experiment reports, so both are reported in the same units.

Outside-patch joint component. A first-order model scores position i as -log p(aa_i | backbone)
with no sequence context, so for a composition-matched pair whose identities differ only inside
the patch, the first-order gap is exactly zero at every outside-patch position. Whatever gap
appears outside the patch cannot be first-order burial preference and is attributable to
sequence coupling. It is a zero-compute lower bound on the magnitude of the joint component,
not the full decomposition, which needs the unconditional-probability control.

Scores in pair_scores.csv are per-residue means. Chain and region totals are reconstructed
as mean * count so that the full chain, the patch and the outside-patch remainder add up.
"""
import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import statistics

TIE_TOLERANCE = 1e-6
LABELS = ("original", "permuted")
REQUIRED_COLUMNS = frozenset(
    "pair_id structure_id chain split group patch_id model_type model_name order_index "
    "order_sha256 length patch_size original_full_nll permuted_full_nll full_nll_gap "
    "original_patch_nll permuted_patch_nll patch_nll_gap".split()
)
SEQUENCE_FIELDS = (
    "pair_id structure_id chain split group patch_id model_type model_name label length "
    "patch_size order_count chain_total_nll_mean chain_total_nll_sd chain_total_nll_range "
    "chain_total_nll_min chain_total_nll_max per_residue_nll_sd"
).split()
PAIR_FIELDS = (
    "pair_id structure_id chain split group patch_id model_type model_name length patch_size "
    "order_count order_sd_original order_sd_permuted order_sd_mean order_sd_max "
    "full_gap_total_mean full_gap_total_abs patch_gap_total_mean outside_gap_total_mean "
    "outside_share_of_full_gap inconsistency_ratio outside_gap_sign outside_gap_consistent_orders"
).split()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, fields, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def finite(value, name):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Nonfinite " + name)
    return number


def sign(value, tolerance=TIE_TOLERANCE):
    return 1 if value > tolerance else -1 if value < -tolerance else 0


def read_scores(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or ())
        if missing:
            raise ValueError("pair_scores.csv is missing columns: " + ", ".join(sorted(missing)))
        rows = list(reader)
    if not rows:
        raise ValueError("pair_scores.csv contains no score rows")
    return rows


def region_totals(row):
    """Chain, patch and outside-patch totals in nats per chain, from per-residue means."""
    length, patch_size = int(row["length"]), int(row["patch_size"])
    if length < 2 or not 2 <= patch_size < length:
        raise ValueError("A patch must hold at least two positions and leave an outside region")
    totals = {}
    for label in LABELS:
        chain = finite(row[label + "_full_nll"], label + "_full_nll") * length
        patch = finite(row[label + "_patch_nll"], label + "_patch_nll") * patch_size
        totals[label] = {"chain": chain, "patch": patch, "outside": chain - patch}
    for region, column in (("chain", "full_nll_gap"), ("patch", "patch_nll_gap")):
        scale = length if region == "chain" else patch_size
        stated = finite(row[column], column) * scale
        derived = totals["permuted"][region] - totals["original"][region]
        if abs(stated - derived) > 1e-6 * max(1.0, abs(stated)):
            raise ValueError("Stated and derived " + region + " gaps disagree for " + row["pair_id"])
    return totals


def collect(rows):
    """Group rows by pair and checkpoint, requiring one row per distinct shared order."""
    groups = defaultdict(dict)
    for row in rows:
        key = (row["pair_id"], row["model_type"])
        index = int(row["order_index"])
        if index in groups[key]:
            raise ValueError("Duplicate order_index for " + row["pair_id"] + "/" + row["model_type"])
        groups[key][index] = {"row": row, "totals": region_totals(row)}
    counts, digests = set(), defaultdict(dict)
    for key, members in groups.items():
        if sorted(members) != list(range(1, len(members) + 1)):
            raise ValueError("Order indices are not a complete 1..n run for " + key[0])
        if len(members) < 2:
            raise ValueError("Order inconsistency needs at least two shared orders for " + key[0])
        counts.add(len(members))
        structure = (members[1]["row"]["structure_id"], members[1]["row"]["chain"])
        for index, member in members.items():
            recorded = digests[structure].setdefault(index, member["row"]["order_sha256"])
            if recorded != member["row"]["order_sha256"]:
                raise ValueError("Orders are not shared within structure " + structure[0])
    if len(counts) != 1:
        raise ValueError("Every pair and checkpoint must use the same number of orders")
    return groups


def summarize_sequences(groups):
    rows = []
    for key in sorted(groups):
        members = groups[key]
        first = members[1]["row"]
        length = int(first["length"])
        for label in LABELS:
            totals = [members[i]["totals"][label]["chain"] for i in sorted(members)]
            deviation = statistics.pstdev(totals)
            rows.append({
                **{field: first[field] for field in
                   "pair_id structure_id chain split group patch_id model_type model_name".split()},
                "label": label, "length": length, "patch_size": int(first["patch_size"]),
                "order_count": len(totals), "chain_total_nll_mean": statistics.fmean(totals),
                "chain_total_nll_sd": deviation, "chain_total_nll_range": max(totals) - min(totals),
                "chain_total_nll_min": min(totals), "chain_total_nll_max": max(totals),
                "per_residue_nll_sd": deviation / length,
            })
    return rows


def summarize_pairs(groups):
    rows = []
    for key in sorted(groups):
        members = groups[key]
        first = members[1]["row"]
        indices = sorted(members)
        deviations = {
            label: statistics.pstdev([members[i]["totals"][label]["chain"] for i in indices])
            for label in LABELS
        }
        gaps = {
            region: [members[i]["totals"]["permuted"][region] - members[i]["totals"]["original"][region]
                     for i in indices]
            for region in ("chain", "patch", "outside")
        }
        full_mean = statistics.fmean(gaps["chain"])
        outside_mean = statistics.fmean(gaps["outside"])
        deviation_mean = statistics.fmean(deviations.values())
        rows.append({
            **{field: first[field] for field in
               "pair_id structure_id chain split group patch_id model_type model_name".split()},
            "length": int(first["length"]), "patch_size": int(first["patch_size"]),
            "order_count": len(indices),
            "order_sd_original": deviations["original"], "order_sd_permuted": deviations["permuted"],
            "order_sd_mean": deviation_mean, "order_sd_max": max(deviations.values()),
            "full_gap_total_mean": full_mean, "full_gap_total_abs": abs(full_mean),
            "patch_gap_total_mean": statistics.fmean(gaps["patch"]),
            "outside_gap_total_mean": outside_mean,
            # Signed share, so a negative value flags an outside region that opposes the full gap.
            "outside_share_of_full_gap": outside_mean / full_mean if abs(full_mean) > TIE_TOLERANCE else None,
            # Order spread measured against the effect it has to resolve; >= 1 means the
            # amortization error is as large as the quantity being reported.
            "inconsistency_ratio": deviation_mean / abs(full_mean) if abs(full_mean) > TIE_TOLERANCE else None,
            "outside_gap_sign": sign(outside_mean),
            "outside_gap_consistent_orders": sum(1 for g in gaps["outside"] if sign(g) == sign(outside_mean))
            if sign(outside_mean) else 0,
        })
    return rows


def distribution(values, name):
    ordered = sorted(values)
    if not ordered:
        raise ValueError("No values to summarize for " + name)
    return {
        "count": len(ordered), "mean": statistics.fmean(ordered), "median": statistics.median(ordered),
        "min": ordered[0], "max": ordered[-1],
        "p25": statistics.quantiles(ordered, n=4)[0] if len(ordered) > 1 else ordered[0],
        "p75": statistics.quantiles(ordered, n=4)[2] if len(ordered) > 1 else ordered[0],
    }


def sign_test(values):
    """Two-sided exact binomial sign test against p=0.5, ties excluded."""
    positive = sum(1 for v in values if sign(v) > 0)
    negative = sum(1 for v in values if sign(v) < 0)
    ties, trials = len(values) - positive - negative, positive + negative
    result = {"positive": positive, "negative": negative, "ties": ties, "nonties": trials}
    if not trials:
        return {**result, "p_value": None}
    extreme = min(positive, negative)
    tail = sum(math.comb(trials, k) for k in range(extreme + 1)) / (2 ** trials)
    return {**result, "p_value": min(1.0, 2 * tail)}


def build_summary(sequence_rows, pair_rows, sources):
    by_model = defaultdict(list)
    for row in pair_rows:
        by_model[row["model_type"]].append(row)
    checkpoints = {}
    for model_type in sorted(by_model):
        group = by_model[model_type]
        ratios = [r["inconsistency_ratio"] for r in group if r["inconsistency_ratio"] is not None]
        shares = [r["outside_share_of_full_gap"] for r in group
                  if r["outside_share_of_full_gap"] is not None]
        # Both lists are empty exactly when every pair's gap is within tie tolerance of zero,
        # which is the degenerate case the ratio exists to diagnose. Report it rather than
        # raising, so the diagnostic still reaches the reader.
        checkpoints[model_type] = {
            "pair_count": len(group),
            "pairs_with_measurable_gap": len(ratios),
            "order_sd_nats_per_chain": distribution([r["order_sd_mean"] for r in group], "order_sd"),
            "absolute_full_gap_nats_per_chain": distribution([r["full_gap_total_abs"] for r in group], "gap"),
            "inconsistency_ratio": distribution(ratios, "ratio") if ratios else None,
            "pairs_with_order_sd_above_gap": sum(1 for r in ratios if r >= 1.0),
            "outside_gap_nats_per_chain": distribution([r["outside_gap_total_mean"] for r in group], "outside"),
            "outside_share_of_full_gap_median": statistics.median(shares) if shares else None,
            "full_gap_sign_test": sign_test([r["full_gap_total_mean"] for r in group]),
            "outside_gap_sign_test": sign_test([r["outside_gap_total_mean"] for r in group]),
            "patch_gap_sign_test": sign_test([r["patch_gap_total_mean"] for r in group]),
        }
    return {
        "schema_version": 1,
        "purpose": "order_consistency_and_outside_patch_joint_component",
        "sources": sources,
        "units": "nats per chain unless a field name says otherwise",
        "sequence_row_count": len(sequence_rows),
        "pair_row_count": len(pair_rows),
        "order_count": sequence_rows[0]["order_count"],
        "theory": {
            "consistent_joint_prediction": "Zero spread of chain-total NLL across decoding orders",
            "observed_spread_meaning": "Amortization error: the network's conditionals are not "
                                       "mutually consistent with one joint distribution",
            "outside_patch_first_order_gap": "Exactly zero, because identities outside the patch are "
                                             "unchanged and unconditional probabilities ignore sequence context",
            "outside_gap_interpretation": "Attributable to sequence coupling; a lower bound on the "
                                          "magnitude of the joint component, not the full decomposition",
        },
        "checkpoints": checkpoints,
    }


def figure(pair_rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = sorted({row["model_type"] for row in pair_rows})
    figure_, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for model_type in models:
        group = [r for r in pair_rows if r["model_type"] == model_type]
        axes[0].scatter([r["order_sd_mean"] for r in group], [r["full_gap_total_abs"] for r in group],
                        s=22, alpha=0.75, label=model_type)
        axes[1].scatter([r["full_gap_total_mean"] for r in group], [r["outside_gap_total_mean"] for r in group],
                        s=22, alpha=0.75, label=model_type)
    limit = max([r["order_sd_mean"] for r in pair_rows] + [r["full_gap_total_abs"] for r in pair_rows]) * 1.08
    axes[0].plot([0, limit], [0, limit], color="0.4", linestyle="--", linewidth=1,
                 label="equal magnitude")
    axes[0].set_xlim(0, limit)
    axes[0].set_ylim(0, limit)
    axes[0].set_xlabel("Decoding-order SD of chain total (nats)")
    axes[0].set_ylabel("|mean arrangement gap| (nats)")
    axes[0].set_title("Amortization spread vs. measured effect")
    axes[1].axhline(0, color="0.4", linewidth=1)
    axes[1].axvline(0, color="0.4", linewidth=1)
    axes[1].set_xlabel("Full-chain gap (nats)")
    axes[1].set_ylabel("Outside-patch gap (nats)")
    axes[1].set_title("Outside-patch component is pure coupling")
    for axis in axes:
        axis.legend(frameon=False, fontsize=8)
        axis.spines[["top", "right"]].set_visible(False)
    figure_.tight_layout()
    for suffix in (".png", ".pdf"):
        figure_.savefig(path.with_suffix(suffix), dpi=200)
    plt.close(figure_)


def run(args):
    scores_path = args.scores.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = read_scores(scores_path)
    groups = collect(rows)
    sequence_rows = summarize_sequences(groups)
    pair_rows = summarize_pairs(groups)
    summary = build_summary(sequence_rows, pair_rows, {"pair_scores": scores_path.name,
                                                       "score_row_count": len(rows)})
    write_csv(output / "order_consistency_sequence.csv", SEQUENCE_FIELDS, sequence_rows)
    write_csv(output / "order_consistency_pair.csv", PAIR_FIELDS, pair_rows)
    write_json(output / "order_consistency_summary.json", summary)
    if not args.no_figure:
        figure(pair_rows, output / "order_consistency")
    for model_type, block in summary["checkpoints"].items():
        ratio = block["inconsistency_ratio"]
        print("{}: order SD median {:.4f} nats, |gap| median {:.4f} nats, ratio median {}, "
              "SD >= gap in {}/{} pairs".format(
                  model_type, block["order_sd_nats_per_chain"]["median"],
                  block["absolute_full_gap_nats_per_chain"]["median"],
                  "{:.3f}".format(ratio["median"]) if ratio else "n/a (no measurable gap)",
                  block["pairs_with_order_sd_above_gap"],
                  block["pair_count"]), flush=True)
        outside = block["outside_gap_sign_test"]
        print("    outside-patch gap: {} positive, {} negative, {} ties, p = {}".format(
            outside["positive"], outside["negative"], outside["ties"],
            "n/a" if outside["p_value"] is None else "{:.3g}".format(outside["p_value"])), flush=True)
    print("Complete: " + str(output / "order_consistency_summary.json"), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scores", type=Path, required=True, help="pair_scores.csv from score_pairs.py")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-figure", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()

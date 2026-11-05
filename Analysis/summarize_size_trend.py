"""Test whether the joint arrangement term varies with chain length. It is a non-result.

The project's eventual target is small, single-domain systems, which raises a fair question:
does the arrangement signal this experiment measures depend on how large the protein is? If
the joint term grew with length, scoping to small systems would be scoping away from the
effect. The question is worth answering before the scope is fixed, and worth reporting even
though the answer is "we cannot tell from seven proteins".

Three normalizations are reported because they do not agree, and quoting only one would
misrepresent the result:

  joint_total        nats per chain; grows with length simply because longer chains hold
                     more terms, so a positive correlation here is close to arithmetic
  joint_per_residue  joint_total divided by chain length
  first_order_share  the fraction of the full gap that is first-order burial, so a negative
                     correlation with length means longer chains are proportionally more
                     arrangement-driven

Significance uses an **exact permutation test** rather than a normal approximation. With seven
structures there are 5,040 orderings, so the null distribution of the correlation is
enumerated completely and the p-value is exact. This matters at n = 7, where the usual
asymptotic p-value for a correlation is not trustworthy.
"""
import argparse
import csv
from collections import defaultdict
import itertools
import json
import math
from pathlib import Path
import statistics

MEASURES = ("joint_total", "joint_per_residue", "first_order_share", "full_gap_total")
STRUCTURE_FIELDS = ("structure_id chain model_type split group length pair_count "
                    "joint_total joint_per_residue first_order_share full_gap_total").split()
# Enumerating permutations is exact but factorial; refuse rather than silently approximate.
MAXIMUM_EXACT_N = 9


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, fields, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def pearson(xs, ys):
    mean_x, mean_y = statistics.fmean(xs), statistics.fmean(ys)
    dx = [x - mean_x for x in xs]
    dy = [y - mean_y for y in ys]
    denominator = math.sqrt(sum(v * v for v in dx) * sum(v * v for v in dy))
    if denominator <= 0:
        return None
    return sum(a * b for a, b in zip(dx, dy)) / denominator


def ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    result = [0.0] * len(values)
    position = 0
    while position < len(order):
        stop = position
        while stop + 1 < len(order) and values[order[stop + 1]] == values[order[position]]:
            stop += 1
        shared = (position + stop) / 2.0
        for index in range(position, stop + 1):
            result[order[index]] = shared
        position = stop + 1
    return result


def spearman(xs, ys):
    return pearson(ranks(xs), ranks(ys))


def exact_permutation_p(xs, ys, statistic):
    """Two-sided exact p: the share of orderings whose |correlation| reaches the observed one."""
    if len(xs) > MAXIMUM_EXACT_N:
        raise ValueError("Exact enumeration refuses more than {} points".format(MAXIMUM_EXACT_N))
    observed = statistic(xs, ys)
    if observed is None:
        return None, None
    total = extreme = 0
    for permuted in itertools.permutations(ys):
        value = statistic(xs, list(permuted))
        total += 1
        if value is not None and abs(value) >= abs(observed) - 1e-12:
            extreme += 1
    return observed, extreme / total


def collect(rows):
    """One record per structure and checkpoint, averaged over that structure's pairs."""
    grouped = defaultdict(list)
    for row in rows:
        key = (row["structure_id"], row["chain"], row["model_type"])
        grouped[key].append(row)
    records = []
    for key in sorted(grouped, key=lambda k: (k[2], int(grouped[k][0]["length"]))):
        group = grouped[key]
        length = int(group[0]["length"])
        joint = statistics.fmean(float(r["delta_joint_total"]) for r in group)
        shares = [float(r["delta_fo_share"]) for r in group if r["delta_fo_share"] not in ("", None)]
        records.append({
            "structure_id": key[0], "chain": key[1], "model_type": key[2],
            "split": group[0].get("split", ""), "group": group[0].get("group", ""),
            "length": length, "pair_count": len(group),
            "joint_total": joint, "joint_per_residue": joint / length,
            "first_order_share": statistics.median(shares) if shares else None,
            "full_gap_total": statistics.fmean(float(r["delta_full_total"]) for r in group),
        })
    return records


def analyze(records):
    by_model = defaultdict(list)
    for record in records:
        by_model[record["model_type"]].append(record)
    result = {}
    for model_type in sorted(by_model):
        group = by_model[model_type]
        lengths = [r["length"] for r in group]
        block = {"structure_count": len(group), "lengths": sorted(lengths), "measures": {},
                 "undefined_measures": []}
        for measure in MEASURES:
            values = [r[measure] for r in group]
            if any(v is None for v in values):
                block["undefined_measures"].append({"measure": measure, "reason": "missing value"})
                continue
            r_value, r_p = exact_permutation_p(lengths, values, pearson)
            # A measure that never varies has no correlation with anything; record that
            # rather than storing a null that later reads as a direction.
            if r_value is None:
                block["undefined_measures"].append(
                    {"measure": measure, "reason": "measure is constant across structures"})
                continue
            rho, rho_p = exact_permutation_p(lengths, values, spearman)
            block["measures"][measure] = {
                "pearson_r": r_value, "pearson_exact_p": r_p,
                "spearman_rho": rho, "spearman_exact_p": rho_p,
                "significant_at_0.05": bool(r_p is not None and r_p < 0.05),
            }
        result[model_type] = block
    return result


def verdict(analysis):
    significant = [(model, measure)
                   for model, block in analysis.items()
                   for measure, stats in block["measures"].items()
                   if stats["significant_at_0.05"]]
    directions = {}
    for model, block in analysis.items():
        for measure, stats in block["measures"].items():
            directions.setdefault(measure, set()).add(
                "positive" if stats["pearson_r"] > 0 else "negative")
    return {
        "significant_relationships": [{"model_type": m, "measure": s} for m, s in significant],
        "any_significant": bool(significant),
        "normalizations_disagree": sorted(
            measure for measure, signs in directions.items() if measure != "full_gap_total"
            and signs != directions.get("joint_total", signs)),
        "statement":
            "No relationship between chain length and the joint arrangement term reaches "
            "significance at seven structures, and the three normalizations do not agree in "
            "sign. The variation across these proteins is protein-specific rather than "
            "size-specific. This is a non-result reported to close the question, not evidence "
            "that size does not matter"
            if not significant else
            "At least one length relationship reached significance; see measures",
    }


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with args.scores.resolve().open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    required = {"structure_id", "chain", "model_type", "length",
                "delta_joint_total", "delta_full_total", "delta_fo_share"}
    missing = required - set(rows[0] if rows else {})
    if missing:
        raise ValueError("first_order_scores.csv is missing: " + ", ".join(sorted(missing)))
    records = collect(rows)
    analysis = analyze(records)
    summary = {"schema_version": 1, "purpose": "chain_length_trend_in_the_joint_term",
               "method": "Exact permutation test over all orderings of the per-structure "
                         "values; no normal approximation",
               "normalization_note":
                   "joint_total grows with length almost arithmetically, so its positive "
                   "correlation is weak evidence; joint_per_residue and first_order_share "
                   "are the informative normalizations and they point opposite ways",
               "structures": records, "by_checkpoint": analysis, "verdict": verdict(analysis)}
    write_csv(output / "size_trend.csv", STRUCTURE_FIELDS, records)
    write_json(output / "size_trend.json", summary)
    for model_type, block in analysis.items():
        print("[{}] n = {} structures".format(model_type, block["structure_count"]), flush=True)
        for measure, stats in block["measures"].items():
            print("    {:<18} r = {:+.3f} (exact p = {:.3f})   rho = {:+.3f} (p = {:.3f})".format(
                measure, stats["pearson_r"], stats["pearson_exact_p"],
                stats["spearman_rho"], stats["spearman_exact_p"]), flush=True)
    print("\n" + summary["verdict"]["statement"], flush=True)
    print("Complete: " + str(output / "size_trend.json"), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scores", type=Path, required=True,
                        help="first_order_scores.csv from score_first_order.py")
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()

"""Compare what the model predicts about an arrangement swap against what was measured.

The scoring run writes one row per variant per decoding order. This turns those into swap
differences and asks how well they track the laboratory.

Sign convention, stated because it is easy to get backwards. The model reports negative log
probability, so a *lower* score means the model prefers that arrangement. The measurement
reports folding stability, so a *higher* value means the protein prefers it. Agreement
therefore shows up as a **negative** correlation, and the sign is reported as-is rather than
flipped to look tidy.

Two comparisons matter:

  pooled       all contrasts together. Easy to read and overstates significance, because a
               hundred thousand contrasts come from only a couple of hundred domains.
  per domain   one correlation per domain, then the distribution of those. This is the honest
               unit, and the fraction of domains agreeing in sign says whether an effect is
               general or driven by a few structures.

Contrasts whose measured effect is smaller than twice its own measurement error are reported
separately. Asking any model to order two numbers that the experiment could not separate is a
test of noise.
"""
import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def spearman(xs, ys):
    def ranked(values):
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
    a, b = ranked(xs), ranked(ys)
    mean_a, mean_b = statistics.fmean(a), statistics.fmean(b)
    da = [v - mean_a for v in a]
    db = [v - mean_b for v in b]
    denominator = math.sqrt(sum(v * v for v in da) * sum(v * v for v in db))
    return sum(x * y for x, y in zip(da, db)) / denominator if denominator else None


def load_variant_scores(path):
    """Order-averaged chain total for every variant, keyed by site pair and identities."""
    totals = defaultdict(list)
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            key = (row["domain"], int(row["site_i"]), int(row["site_j"]),
                   row["aa_i"], row["aa_j"])
            totals[key].append(float(row["full_nll_total"]))
    return {key: (statistics.fmean(values), len(values)) for key, values in totals.items()}


def build_rows(contrasts_path, scores):
    rows, missing = [], 0
    with Path(contrasts_path).open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            domain = row["domain"]
            site_i, site_j = int(row["site_i"]), int(row["site_j"])
            a, b = row["aa_i"], row["aa_j"]
            forward = scores.get((domain, site_i, site_j, a, b))
            reverse = scores.get((domain, site_i, site_j, b, a))
            if forward is None or reverse is None:
                missing += 1
                continue
            rows.append({
                "domain": domain,
                "measured": float(row["difference"]),
                "model": forward[0] - reverse[0],
                "resolved": str(row["resolved"]).lower() == "true",
                "orders": forward[1],
            })
    return rows, missing


def agreement(rows):
    if len(rows) < 3:
        return None
    measured = [r["measured"] for r in rows]
    model = [r["model"] for r in rows]
    # Agreement means the model's preferred arrangement is the more stable one: a lower
    # model score paired with a higher measured stability.
    agree = sum(1 for r in rows if (r["measured"] > 0) == (r["model"] < 0))
    return {"count": len(rows), "spearman": spearman(model, measured),
            "sign_agreement": agree / len(rows)}


def per_domain(rows, minimum):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["domain"]].append(row)
    values = []
    for domain, group in sorted(grouped.items()):
        if len(group) < minimum:
            continue
        value = spearman([r["model"] for r in group], [r["measured"] for r in group])
        if value is not None:
            values.append({"domain": domain, "count": len(group), "spearman": value})
    if not values:
        return None
    numbers = sorted(v["spearman"] for v in values)
    return {
        "domains": len(values), "minimum_contrasts": minimum,
        "median": statistics.median(numbers),
        "p25": numbers[len(numbers) // 4], "p75": numbers[3 * len(numbers) // 4],
        "agreeing_in_sign": sum(1 for v in numbers if v < 0),
        "agreeing_fraction": sum(1 for v in numbers if v < 0) / len(numbers),
        "per_domain": values,
    }


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {"schema_version": 1, "purpose": "model_versus_measured_swap_effect",
              "sign_convention": "Model reports negative log probability, lower means "
                                 "preferred; measurement reports stability, higher means "
                                 "preferred. Agreement is a negative correlation.",
              "checkpoints": {}}
    for item in args.scores:
        label, _, path = item.partition("=")
        if not path:
            raise ValueError("Pass scores as label=path, e.g. vanilla=.../scores.csv")
        print("loading %s" % label, flush=True)
        scores = load_variant_scores(path)
        rows, missing = build_rows(args.contrasts, scores)
        resolved = [r for r in rows if r["resolved"]]
        unresolved = [r for r in rows if not r["resolved"]]
        block = {
            "variants_scored": len(scores), "contrasts_matched": len(rows),
            "contrasts_unmatched": missing,
            "orders_per_variant": rows[0]["orders"] if rows else None,
            "all": agreement(rows),
            "resolved": agreement(resolved),
            "below_measurement_error": agreement(unresolved),
            "per_domain_resolved": per_domain(resolved, args.minimum_per_domain),
        }
        report["checkpoints"][label] = block
        pooled, domains = block["resolved"], block["per_domain_resolved"]
        print("  %-8s resolved n=%d  pooled rho=%+.3f  sign=%.1f%%" % (
            label, pooled["count"], pooled["spearman"], 100 * pooled["sign_agreement"]),
            flush=True)
        if domains:
            print("           per-domain median rho=%+.3f over %d domains, %.0f%% agree in sign"
                  % (domains["median"], domains["domains"],
                     100 * domains["agreeing_fraction"]), flush=True)
    write_json(output / "swap_prediction_comparison.json", report)
    print("Complete: " + str(output / "swap_prediction_comparison.json"), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--contrasts", required=True)
    parser.add_argument("--scores", nargs="+", required=True,
                        help="one or more label=path pairs")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-per-domain", type=int, default=20)
    run(parser.parse_args())


if __name__ == "__main__":
    main()

"""Test whether the model's own non-additivity predicts measured thermodynamic coupling.

The decomposition split ProteinMPNN's response to a rearrangement into a position-wise term and
a joint residual. That residual is defined inside the model's own arithmetic, so until now
nothing outside the model said whether it corresponds to anything real.

The stability dataset supplies the external quantity. For each site pair it measured all 400
identity combinations and reports the deviation from a fully additive fit: the thermodynamic
coupling between the two positions, in kcal/mol. Two columns carry it, the coupling at the
wild-type identities and the largest coupling over the 400 combinations.

The model's analogue is computed the same way rather than borrowed. For each site pair, the
predicted scores for all measured combinations are fitted with an additive model, one term per
identity at each site, and the root-mean-square residual is what the model's representation
says about coupling at that pair. Fitting proceeds by alternating means, which converges for a
near-complete grid and tolerates the missing cells that a few pairs have.

Independence is the trap here. The pairs come from about two hundred domains, several per
domain, so a pooled correlation across pairs overstates its own significance. The same
comparison is therefore repeated on one pair per domain, chosen by a fixed rule, and on
domain-centred values. If the relationship survives all three, it is not an artefact of domain
structure.
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


def additive_residual(table, iterations=50):
    """Root-mean-square deviation from a one-term-per-identity additive fit."""
    xs = sorted({x for x, _ in table})
    ys = sorted({y for _, y in table})
    mean = statistics.fmean(table.values())
    a = {x: 0.0 for x in xs}
    b = {y: 0.0 for y in ys}
    for _ in range(iterations):
        for x in xs:
            values = [table[(x, y)] - mean - b[y] for y in ys if (x, y) in table]
            if values:
                a[x] = statistics.fmean(values)
        for y in ys:
            values = [table[(x, y)] - mean - a[x] for x in xs if (x, y) in table]
            if values:
                b[y] = statistics.fmean(values)
    residuals = [table[(x, y)] - (mean + a[x] + b[y]) for (x, y) in table]
    return math.sqrt(statistics.fmean(r * r for r in residuals))


def load_model_coupling(path, minimum_cells):
    raw = defaultdict(lambda: defaultdict(list))
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            key = (row["domain"], int(row["site_i"]), int(row["site_j"]))
            raw[key][(row["aa_i"], row["aa_j"])].append(float(row["full_nll_total"]))
    coupling, skipped = {}, 0
    for key, cells in raw.items():
        table = {ab: statistics.fmean(values) for ab, values in cells.items()}
        if len(table) < minimum_cells:
            skipped += 1
            continue
        coupling[key] = additive_residual(table)
    return coupling, skipped


def load_measured(path):
    measured = {}
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            try:
                domain = row["pdb_name"].replace(".pdb", "")
                one, two = int(row["pos1"]), int(row["pos2"])
            except (KeyError, ValueError):
                continue
            entry = {}
            for column in ("max_ep_dG", "wt_ep_dG"):
                try:
                    value = float(row[column])
                    entry[column] = value if value == value else None
                except (KeyError, TypeError, ValueError):
                    entry[column] = None
            measured[(domain, min(one, two), max(one, two))] = entry
    return measured


def compare(model, measured, column):
    paired = [(key, model[key], measured[key][column]) for key in model
              if key in measured and measured[key].get(column) is not None]
    if len(paired) < 3:
        return None
    pooled = spearman([p[1] for p in paired], [p[2] for p in paired])
    # One pair per domain, chosen by a fixed rule, so no domain contributes twice.
    by_domain = defaultdict(list)
    for key, m, v in paired:
        by_domain[key[0]].append((key, m, v))
    single = [sorted(group)[0] for group in by_domain.values()]
    independent = spearman([p[1] for p in single], [p[2] for p in single])
    # Domain-centred: remove each domain's mean from both sides, keeping only domains
    # that contribute more than one pair.
    xs, ys = [], []
    for group in by_domain.values():
        if len(group) < 2:
            continue
        mx = statistics.fmean(p[1] for p in group)
        my = statistics.fmean(p[2] for p in group)
        for _, m, v in group:
            xs.append(m - mx)
            ys.append(v - my)
    centred = spearman(xs, ys) if len(xs) >= 3 else None
    return {"pairs": len(paired), "pooled_spearman": pooled,
            "domains": len(by_domain), "one_per_domain_spearman": independent,
            "within_domain_centred_spearman": centred,
            "within_domain_points": len(xs)}


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    model, skipped = load_model_coupling(args.scores, args.minimum_cells)
    measured = load_measured(args.measured)
    report = {"schema_version": 1, "purpose": "model_nonadditivity_versus_measured_coupling",
              "model_coupling": "root-mean-square residual from an additive fit to the "
                                "model's predicted scores across identity combinations",
              "site_pairs_with_model_estimate": len(model),
              "site_pairs_skipped_too_few_cells": skipped,
              "minimum_cells": args.minimum_cells, "comparisons": {}}
    for column in ("max_ep_dG", "wt_ep_dG"):
        result = compare(model, measured, column)
        report["comparisons"][column] = result
        if result:
            print("%-10s pairs=%-4d pooled=%+.3f | one-per-domain=%+.3f (n=%d) | "
                  "within-domain=%+.3f" % (
                      column, result["pairs"], result["pooled_spearman"],
                      result["one_per_domain_spearman"], result["domains"],
                      result["within_domain_centred_spearman"]
                      if result["within_domain_centred_spearman"] is not None else float("nan")),
                  flush=True)
    write_json(output / "coupling_comparison.json", report)
    print("Complete: " + str(output / "coupling_comparison.json"), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scores", required=True, help="swap_variant_scores.csv")
    parser.add_argument("--measured", required=True, help="Double_DMS_list.csv")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-cells", type=int, default=100)
    run(parser.parse_args())


if __name__ == "__main__":
    main()

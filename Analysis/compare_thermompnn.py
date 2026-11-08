"""Compare ThermoMPNN-D against frozen ProteinMPNN on exactly the same contrasts.

The published baseline cannot be compared against the number already in hand. ThermoMPNN-D
scores only genuine double mutants, so it covers 81.3% of the measured contrasts, while the
frozen model's -0.497 per-domain median was computed over all of them. Putting those two
numbers side by side would credit or penalise a model for the contrasts it never saw.

So this restricts both models to the intersection and recomputes the frozen baseline there.
The headline comparison is paired across domains, as the baseline comparison was, because
pooling tens of thousands of contrasts from 199 domains overstates significance.

Both models agree with the measurement by *negative* correlation: ProteinMPNN reports negative
log probability and ThermoMPNN reports ddG, both lower when the arrangement is preferred, while
measured stability is higher when the arrangement is more stable.

A caveat this script cannot resolve, and which must accompany any use of its output:
ThermoMPNN-D was trained on the same mega-scale dataset these contrasts come from, and the
split file its config names is not distributed with the model. For any domain it trained on,
its score here reflects memorisation as much as generalisation, so this is an upper bound on
its performance until that split is obtained.
"""

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare_swap_predictions as frozen_loader

MIN_CONTRASTS = 20


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_thermompnn(path):
    """Resolved contrasts only, keyed exactly as the frozen loader keys its variants."""
    rows = {}
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if row["resolved"] != "True":
                continue
            key = (row["domain"], int(row["site_i"]), int(row["site_j"]),
                   row["aa_i"], row["aa_j"])
            rows[key] = (float(row["model_difference"]), float(row["measured_difference"]))
    return rows


def per_domain(paired, spearmanr):
    out = {}
    for domain, values in paired.items():
        if len(values) >= MIN_CONTRASTS:
            model = [v[0] for v in values]
            measured = [v[1] for v in values]
            rho = spearmanr(model, measured).statistic
            if rho == rho:
                out[domain] = rho
    return out


def summarize(label, pooled_rho, domains):
    values = sorted(domains.values())
    return {"label": label, "pooled_spearman": pooled_rho, "domains": len(domains),
            "median": statistics.median(values) if values else None,
            "p25": values[len(values) // 4] if values else None,
            "p75": values[(3 * len(values)) // 4] if values else None}


def run(args):
    from scipy.stats import spearmanr, binomtest

    thermo = read_thermompnn(args.thermompnn)
    variants = frozen_loader.load_variant_scores(args.frozen)

    thermo_by_domain, frozen_by_domain = defaultdict(list), defaultdict(list)
    pooled_t, pooled_f, measured_all = [], [], []
    missing_frozen = 0

    for key, (model_difference, measured) in sorted(thermo.items()):
        domain, site_i, site_j, aa_i, aa_j = key
        forward = variants.get((domain, site_i, site_j, aa_i, aa_j))
        reverse = variants.get((domain, site_i, site_j, aa_j, aa_i))
        if forward is None or reverse is None:
            missing_frozen += 1
            continue
        frozen_difference = forward[0] - reverse[0]
        thermo_by_domain[domain].append((model_difference, measured))
        frozen_by_domain[domain].append((frozen_difference, measured))
        pooled_t.append(model_difference)
        pooled_f.append(frozen_difference)
        measured_all.append(measured)

    if not measured_all:
        raise SystemExit("No contrasts matched between the two models")

    thermo_domains = per_domain(thermo_by_domain, spearmanr)
    frozen_domains = per_domain(frozen_by_domain, spearmanr)
    shared = sorted(set(thermo_domains) & set(frozen_domains))

    # Paired: a model is stronger where its correlation is more negative.
    better_frozen = sum(1 for d in shared if frozen_domains[d] < thermo_domains[d])
    nonties = sum(1 for d in shared if frozen_domains[d] != thermo_domains[d])
    test = binomtest(better_frozen, nonties, 0.5) if nonties else None

    summary = {
        "schema_version": 1,
        "purpose": "thermompnn_d_versus_frozen_proteinmpnn_on_identical_contrasts",
        "sign_convention": ("Both models report lower values for the arrangement they prefer; "
                            "measured stability is higher for the more stable arrangement, so "
                            "agreement is a negative correlation."),
        "training_overlap_caveat": ("ThermoMPNN-D was trained on this dataset and its split "
                                    "file is not distributed. Treat its score as an upper "
                                    "bound until domain membership is known."),
        "contrasts_compared": len(measured_all),
        "contrasts_without_frozen_scores": missing_frozen,
        "minimum_contrasts_per_domain": MIN_CONTRASTS,
        "thermompnn_d": summarize("ThermoMPNN-D", spearmanr(pooled_t, measured_all).statistic,
                                  thermo_domains),
        "frozen_proteinmpnn": summarize("frozen ProteinMPNN, full model",
                                        spearmanr(pooled_f, measured_all).statistic,
                                        frozen_domains),
        "paired_sign_test": {
            "domains": len(shared), "nonties": nonties,
            "frozen_stronger": better_frozen,
            "thermompnn_stronger": nonties - better_frozen,
            "p_value": test.pvalue if test else None,
        },
        "per_domain": [{"domain": d, "thermompnn_d": thermo_domains[d],
                        "frozen": frozen_domains[d]} for d in shared],
    }

    args.output.resolve().mkdir(parents=True, exist_ok=True)
    write_json(args.output / "thermompnn_comparison.json", summary)

    print("contrasts compared        : %d" % len(measured_all))
    for block in ("thermompnn_d", "frozen_proteinmpnn"):
        entry = summary[block]
        print("%-32s pooled %.4f   per-domain median %.4f  (%d domains)"
              % (entry["label"], entry["pooled_spearman"], entry["median"], entry["domains"]))
    paired = summary["paired_sign_test"]
    print("frozen stronger in %d of %d domains, ThermoMPNN-D in %d, p = %.3g"
          % (paired["frozen_stronger"], paired["nonties"],
             paired["thermompnn_stronger"], paired["p_value"]))
    print("Complete: " + str(args.output / "thermompnn_comparison.json"))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--thermompnn", required=True,
                        help="thermompnn_pair_predictions.csv")
    parser.add_argument("--frozen", required=True,
                        help="swap_variant_scores.csv from the full autoregressive run")
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()

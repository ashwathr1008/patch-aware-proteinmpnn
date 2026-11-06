"""Extract composition-matched arrangement contrasts with measured labels.

The project's matched-pair design rearranges residue identities within a patch while holding
composition and backbone fixed, then asks what a model does with the change. Until now those
rearranged sequences carried no measured label, because nobody has synthesised them: a
permuted patch exists only in the manifest.

The mega-scale stability dataset breaks that, for pairs of sites. For each selected site pair
it measured all 400 identity combinations. Inside those 400 sit the swaps:

    variant A   X at site i,  Y at site j
    variant B   Y at site i,  X at site j

Same two residues, same backbone, different arrangement, both measured. The difference in
folding stability between them is an arrangement effect that someone recorded in a laboratory.
A two-residue swap is a permutation of a two-residue patch, so these are the project's own
experiment carried out on real protein.

The sites were chosen as interacting residues, which this repository verifies rather than
assumes; see test_site_pair_geometry.py.

Each contrast is emitted once per unordered identity pair, with both stabilities, their
difference, and a combined uncertainty, so that contrasts smaller than their own measurement
error can be filtered rather than silently trusted.
"""
import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

# A mutation code such as L32Q, bounded so it cannot match inside a longer token.
MUTATION = re.compile(r"(?<![A-Za-z0-9])([ACDEFGHIKLMNPQRSTVWY])(\d+)"
                      r"([ACDEFGHIKLMNPQRSTVWY])(?![A-Za-z0-9])")
FIELDS = ("domain site_i site_j aa_i aa_j stability_forward stability_reverse "
          "difference uncertainty resolved").split()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def parse_variants(path, stability_column, interval_column):
    """Map each (domain, site pair) to the stabilities of its identity combinations."""
    table = defaultdict(dict)
    rows = skipped = 0
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            rows += 1
            name = row.get("name_original") or ""
            domain = name.split(".pdb")[0]
            mutations = sorted((int(m.group(2)), m.group(3))
                               for m in MUTATION.finditer(name.split("_")[-1]))
            if len(mutations) != 2 or mutations[0][0] == mutations[1][0]:
                skipped += 1
                continue
            try:
                stability = float(row[stability_column])
                interval = float(row[interval_column])
            except (KeyError, TypeError, ValueError):
                skipped += 1
                continue
            if not (math.isfinite(stability) and math.isfinite(interval)):
                skipped += 1
                continue
            (site_i, aa_i), (site_j, aa_j) = mutations
            table[(domain, site_i, site_j)][(aa_i, aa_j)] = (stability, interval)
    return table, rows, skipped


def swap_contrasts(table):
    """One record per unordered identity pair that was measured in both arrangements."""
    contrasts = []
    for (domain, site_i, site_j), combinations in sorted(table.items()):
        emitted = set()
        for (aa_i, aa_j), (forward, forward_interval) in sorted(combinations.items()):
            if aa_i == aa_j:
                continue
            mirror = combinations.get((aa_j, aa_i))
            if mirror is None:
                continue
            key = tuple(sorted((aa_i, aa_j)))
            if key in emitted:
                continue
            emitted.add(key)
            reverse, reverse_interval = mirror
            # Half-widths combined in quadrature; the interval columns are full widths.
            uncertainty = math.sqrt(forward_interval ** 2 + reverse_interval ** 2) / 2
            difference = forward - reverse
            contrasts.append({
                "domain": domain, "site_i": site_i, "site_j": site_j,
                "aa_i": key[0], "aa_j": key[1],
                "stability_forward": forward, "stability_reverse": reverse,
                "difference": difference, "uncertainty": uncertainty,
                # Resolved means the effect is larger than twice its own measurement error.
                "resolved": abs(difference) > 2 * uncertainty,
            })
    return contrasts


def summarize(contrasts, table, rows, skipped):
    sizes = [len(v) for v in table.values()]
    magnitudes = sorted(abs(c["difference"]) for c in contrasts)
    uncertainties = [c["uncertainty"] for c in contrasts]
    resolved = sum(1 for c in contrasts if c["resolved"])

    def quantile(values, q):
        return values[min(len(values) - 1, int(q * len(values)))] if values else None

    by_domain = defaultdict(int)
    for c in contrasts:
        by_domain[c["domain"]] += 1
    return {
        "schema_version": 1, "purpose": "composition_matched_arrangement_contrasts",
        "definition": "Two variants of one site pair carrying the same two residue identities "
                      "in opposite arrangements, both measured. The difference in folding "
                      "stability is a measured arrangement effect at fixed composition.",
        "source_rows": rows, "rows_skipped": skipped,
        "site_pairs": len(table), "domains": len(by_domain),
        "combinations_per_site_pair": {
            "median": statistics.median(sizes) if sizes else None,
            "min": min(sizes) if sizes else None, "max": max(sizes) if sizes else None},
        "contrasts": len(contrasts),
        "contrasts_resolved": resolved,
        "resolved_fraction": resolved / len(contrasts) if contrasts else None,
        "magnitude_kcal_per_mol": {
            "median": statistics.median(magnitudes) if magnitudes else None,
            "p75": quantile(magnitudes, 0.75), "p90": quantile(magnitudes, 0.90),
            "max": magnitudes[-1] if magnitudes else None},
        "uncertainty_kcal_per_mol_median": statistics.median(uncertainties)
        if uncertainties else None,
        "contrasts_per_domain_median": statistics.median(by_domain.values())
        if by_domain else None,
    }


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    table, rows, skipped = parse_variants(args.variants, args.stability_column,
                                          args.interval_column)
    contrasts = swap_contrasts(table)
    summary = summarize(contrasts, table, rows, skipped)
    with (output / "swap_contrasts.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(contrasts)
    write_json(output / "swap_contrasts_summary.json", summary)
    print("parsed {} rows into {} site pairs across {} domains".format(
        rows, summary["site_pairs"], summary["domains"]), flush=True)
    print("combinations per site pair: median {} of 400".format(
        summary["combinations_per_site_pair"]["median"]), flush=True)
    print("swap contrasts            : {}".format(summary["contrasts"]), flush=True)
    print("  magnitude median {:.2f} kcal/mol, uncertainty median {:.2f}".format(
        summary["magnitude_kcal_per_mol"]["median"],
        summary["uncertainty_kcal_per_mol_median"]), flush=True)
    print("  resolved above twice their own error: {} ({:.0%})".format(
        summary["contrasts_resolved"], summary["resolved_fraction"]), flush=True)
    print("Complete: " + str(output / "swap_contrasts_summary.json"), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variants", required=True, help="double-mutant CSV from the dataset")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stability-column", default="deltaG")
    parser.add_argument("--interval-column", default="deltaG_95CI")
    run(parser.parse_args())


if __name__ == "__main__":
    main()

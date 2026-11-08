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


NAME_COLUMNS = ("name_original", "name")
CONFLICT_TOLERANCE = 1e-9
DUPLICATE_POLICIES = ("drop", "average-if-consistent")


def consistent(values):
    """Do repeated measurements of one variant agree within their own stated error?

    The test mirrors the `resolved` rule used for contrasts: a difference counts as real only
    when it exceeds twice the combined uncertainty. So two replicates are consistent exactly
    when their difference would *not* be called resolved. The interval columns are full 95%
    widths, hence the halving.
    """
    for index, (s_a, i_a) in enumerate(values):
        for s_b, i_b in values[index + 1:]:
            combined = math.sqrt((i_a / 2) ** 2 + (i_b / 2) ** 2)
            if abs(s_a - s_b) > 2 * combined:
                return False
    return True


def combine(values):
    """Mean of consistent replicates, with the uncertainty of that mean.

    For half-widths h_k the mean's half-width is sqrt(sum h_k^2) / N, so the full width is
    sqrt(sum i_k^2) / N. Averaging genuinely reduces the error rather than inheriting one
    replicate's, which is the reason to average rather than pick.
    """
    count = len(values)
    stability = statistics.fmean(v[0] for v in values)
    width = math.sqrt(sum(v[1] ** 2 for v in values)) / count
    return (stability, width)


def domain_of(name):
    """Bare domain name, from whatever precedes `.pdb`.

    Order matters here. A pipe appears in two unrelated roles: as a chain prefix before the
    domain, `EA|run2_0325_0005.pdb`, and as a separator between the two substitutions,
    `1GL5.pdb_dmutv5_10F:53Y_F10Q|Y53Q`. Splitting on the pipe first turns the second form's
    domain into `Y53Q`, collapsing every domain onto mutation strings. So the `.pdb` split
    comes first, and only the remainder is checked for a chain prefix.
    """
    base = name.split(".pdb")[0]
    return base.rsplit("|", 1)[-1] if "|" in base else base


def parse_variants(path, stability_column, interval_column,
                   duplicate_policy="drop"):
    """Map each (domain, site pair) to the stabilities of its identity combinations.

    **Repeated variant keys are refused, not overwritten.** An earlier version assigned
    straight into the table, so when the same domain, site pair and identity pair appeared
    twice with different measurements, whichever row came last silently won and the other
    measurement vanished. A review found real instances of this. There is no principled way to
    pick between two measurements of the same variant here, so every observation is collected
    and a key carrying more than one distinct value is dropped and counted. Exact repeats are
    collapsed, since those carry no disagreement.
    """
    observations = defaultdict(lambda: defaultdict(list))
    rows = skipped = 0
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        available = [c for c in NAME_COLUMNS if c in (reader.fieldnames or [])]
        if not available:
            raise SystemExit("No variant-name column found; expected one of %s"
                             % (NAME_COLUMNS,))
        name_column = available[0]
        for row in reader:
            rows += 1
            name = row.get(name_column) or ""
            domain = domain_of(name)
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
            observations[(domain, site_i, site_j)][(aa_i, aa_j)].append((stability, interval))

    table, conflicts, merged = defaultdict(dict), [], 0
    for pair_key, combinations in observations.items():
        for identity_key, values in combinations.items():
            distinct = []
            for value in values:
                if not any(abs(value[0] - seen[0]) <= CONFLICT_TOLERANCE
                           and abs(value[1] - seen[1]) <= CONFLICT_TOLERANCE
                           for seen in distinct):
                    distinct.append(value)
            if len(distinct) == 1:
                table[pair_key][identity_key] = distinct[0]
                continue
            record = {"domain": pair_key[0], "site_i": pair_key[1], "site_j": pair_key[2],
                      "aa_i": identity_key[0], "aa_j": identity_key[1],
                      "observations": len(values),
                      "distinct_values": [list(v) for v in distinct]}
            if duplicate_policy == "average-if-consistent" and consistent(distinct):
                record["resolution"] = "averaged"
                table[pair_key][identity_key] = combine(distinct)
                merged += 1
            else:
                record["resolution"] = "dropped"
                conflicts.append(record)
    return table, rows, skipped, conflicts, merged


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
    table, rows, skipped, conflicts, merged = parse_variants(
        args.variants, args.stability_column, args.interval_column, args.duplicate_policy)
    contrasts = swap_contrasts(table)
    summary = summarize(contrasts, table, rows, skipped)
    # Repeated variant keys carrying different measurements, dropped rather than overwritten.
    summary["duplicate_policy"] = args.duplicate_policy
    summary["conflicting_variant_keys_dropped"] = len(conflicts)
    summary["duplicate_keys_averaged"] = merged
    summary["conflicts"] = conflicts[:500]
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
    print("duplicate keys averaged: {}, dropped: {}".format(merged, len(conflicts)), flush=True)
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
    parser.add_argument("--duplicate-policy", default="average-if-consistent",
                        choices=DUPLICATE_POLICIES,
                        help="repeated measurements of one variant: drop them, or average "
                             "those agreeing within their own error and drop the rest")
    run(parser.parse_args())


if __name__ == "__main__":
    main()

"""Measure whether the double-mutant site pairs are spatially clustered, rather than assuming it.

The mega-scale stability dataset selected its double-mutant site pairs with a contact-detection
program, so the pairs should be in contact. "Should be" is not a measurement, and the project
depends on it: a two-site swap is only a patch-arrangement contrast if the two sites actually
sit together in space. If the pairs turn out to be arbitrary, the swap idea collapses into a
pair of unrelated point mutations.

The test compares each selected pair's Cbeta-Cbeta distance against the distribution of every
possible pair in the same domain. Using the same domain as its own null controls for domain
size and shape: a small domain has every residue close to every other, and a global null would
make its pairs look selected when they are not.

Three things are reported:

  contact fraction     share of selected pairs within the contact cutoff, against the share
                       expected if pairs had been drawn at random from the same domains
  sequence separation  how far apart the two sites are in sequence. Pairs that are close in
                       space but distant in sequence are the informative ones, since a
                       sequence-local pair could be explained by backbone geometry alone
  pair clustering      for domains carrying several pairs, whether those pairs sit near each
                       other, which would make them patch-like rather than scattered

Glycine has no Cbeta, so its alpha carbon stands in. Distances come from the structure files
supplied with the dataset.
"""
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
import random
import statistics

CONTACT_CUTOFF = 8.0
LOCAL_SEQUENCE_SEPARATION = 4


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def parse_structure(path):
    """Residue number to representative side-chain coordinate, from a PDB file."""
    beta, alpha = {}, {}
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        if line[:4] != "ATOM":
            continue
        name = line[12:16].strip()
        if name not in ("CB", "CA"):
            continue
        try:
            number = int(line[22:26])
            point = tuple(float(line[30 + 8 * i:38 + 8 * i]) for i in range(3))
        except ValueError:
            continue
        (beta if name == "CB" else alpha)[number] = point
    # Glycine contributes no CB, so its CA stands in.
    return {n: beta.get(n, alpha.get(n)) for n in set(beta) | set(alpha)}


def distance(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def all_pair_distances(coordinates):
    numbers = sorted(coordinates)
    return [(a, b, distance(coordinates[a], coordinates[b]))
            for i, a in enumerate(numbers) for b in numbers[i + 1:]]


def analyze(pairs_by_domain, structures, cutoff, seed=0):
    """Compare selected pairs against every possible pair in the same domain."""
    rng = random.Random(seed)
    selected, null, separations, per_domain = [], [], [], []
    missing = []
    for domain, sites in sorted(pairs_by_domain.items()):
        path = structures.get(domain)
        if path is None:
            missing.append(domain)
            continue
        coordinates = parse_structure(path)
        if len(coordinates) < 3:
            missing.append(domain)
            continue
        every = all_pair_distances(coordinates)
        if not every:
            continue
        domain_selected = []
        for i, j in sites:
            if i not in coordinates or j not in coordinates:
                continue
            d = distance(coordinates[i], coordinates[j])
            domain_selected.append((i, j, d))
            selected.append(d)
            separations.append(abs(i - j))
        if not domain_selected:
            continue
        # The null is every possible pair in this same domain, so domain size and shape
        # cannot make scattered pairs look selected.
        null.extend(d for _, _, d in every)
        centres = []
        for i, j, _ in domain_selected:
            centres.append(((coordinates[i][0] + coordinates[j][0]) / 2,
                            (coordinates[i][1] + coordinates[j][1]) / 2,
                            (coordinates[i][2] + coordinates[j][2]) / 2))
        spread = None
        if len(centres) > 1:
            spread = statistics.fmean(
                distance(centres[a], centres[b])
                for a in range(len(centres)) for b in range(a + 1, len(centres)))
        per_domain.append({"domain": domain, "pairs": len(domain_selected),
                           "residues": len(coordinates),
                           "median_pair_distance": statistics.median(
                               [d for _, _, d in domain_selected]),
                           "mean_distance_between_pair_centres": spread})
    def share(values):
        return sum(1 for v in values if v <= cutoff) / len(values) if values else None
    result = {
        "contact_cutoff_angstrom": cutoff,
        "selected_pairs_measured": len(selected),
        "domains_measured": len(per_domain),
        "domains_without_structure": missing[:20],
        "domains_without_structure_count": len(missing),
        "selected": {
            "median_distance": statistics.median(selected) if selected else None,
            "mean_distance": statistics.fmean(selected) if selected else None,
            "contact_fraction": share(selected),
        },
        "all_possible_pairs_same_domains": {
            "count": len(null),
            "median_distance": statistics.median(null) if null else None,
            "contact_fraction": share(null),
        },
        "sequence_separation": {
            "median": statistics.median(separations) if separations else None,
            "fraction_local": (sum(1 for s in separations if s <= LOCAL_SEQUENCE_SEPARATION)
                               / len(separations)) if separations else None,
            "local_threshold": LOCAL_SEQUENCE_SEPARATION,
        },
        "per_domain": per_domain,
    }
    if selected and null:
        result["enrichment"] = {
            "contact_fraction_ratio": (result["selected"]["contact_fraction"]
                                       / result["all_possible_pairs_same_domains"]["contact_fraction"])
            if result["all_possible_pairs_same_domains"]["contact_fraction"] else None,
            "median_distance_difference": (result["all_possible_pairs_same_domains"]["median_distance"]
                                           - result["selected"]["median_distance"]),
        }
    return result


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    pairs_by_domain = defaultdict(set)
    with Path(args.pairs).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            domain = row[args.domain_column].strip()
            try:
                i, j = int(float(row[args.site1_column])), int(float(row[args.site2_column]))
            except (ValueError, KeyError):
                continue
            if i != j:
                pairs_by_domain[domain].add((min(i, j), max(i, j)))
    structures = {}
    for path in Path(args.structures).rglob("*.pdb"):
        structures[path.stem] = path
        structures.setdefault(path.stem.split(".")[0], path)
    result = analyze(pairs_by_domain, structures, args.cutoff)
    result["distinct_site_pairs"] = sum(len(v) for v in pairs_by_domain.values())
    result["domains_with_pairs"] = len(pairs_by_domain)
    write_json(output / "site_pair_geometry.json", result)
    s, n = result["selected"], result["all_possible_pairs_same_domains"]
    print("site pairs parsed : {} across {} domains".format(
        result["distinct_site_pairs"], result["domains_with_pairs"]), flush=True)
    print("measured          : {} pairs across {} domains ({} domains had no structure)".format(
        result["selected_pairs_measured"], result["domains_measured"],
        result["domains_without_structure_count"]), flush=True)
    if s["median_distance"] is not None:
        print("selected pairs    : median {:.1f} A, {:.1%} within {:.0f} A".format(
            s["median_distance"], s["contact_fraction"], args.cutoff), flush=True)
        print("all possible pairs: median {:.1f} A, {:.1%} within {:.0f} A".format(
            n["median_distance"], n["contact_fraction"], args.cutoff), flush=True)
        if result.get("enrichment"):
            print("enrichment        : {:.1f}x more likely to be a contact".format(
                result["enrichment"]["contact_fraction_ratio"]), flush=True)
        sep = result["sequence_separation"]
        print("sequence gap      : median {}, {:.1%} are sequence-local (<= {})".format(
            sep["median"], sep["fraction_local"], sep["local_threshold"]), flush=True)
    print("Complete: " + str(output / "site_pair_geometry.json"), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pairs", required=True, help="CSV listing the double-mutant site pairs")
    parser.add_argument("--structures", required=True, help="directory of PDB files")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--domain-column", default="name")
    parser.add_argument("--site1-column", default="site1")
    parser.add_argument("--site2-column", default="site2")
    parser.add_argument("--cutoff", type=float, default=CONTACT_CUTOFF)
    run(parser.parse_args())


if __name__ == "__main__":
    main()

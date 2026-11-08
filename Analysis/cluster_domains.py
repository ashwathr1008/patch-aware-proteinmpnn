"""Cluster the 208 scored domains, and say how much independent evidence they carry.

A count of contrasts is not a count of evidence. The 81,769 measured contrasts sit on 208
domains, those domains are not independent of each other, and a split that divides a family
puts relatives on both sides and invalidates the evaluation. So the unit of independence is the
homology cluster, and this measures how many there are before any training happens.

Two further corrections are reported, because the cluster count alone still overstates things.

Clusters are unequal. A cluster carrying nine thousand contrasts and one carrying forty are not
equal evidence, so Kish's effective sample size, (sum w)^2 / sum w^2 over per-cluster contrast
counts, is reported alongside the raw count. It answers "how many equally-weighted clusters
would carry this much information", which is the number that belongs in a power calculation.

ThermoMPNN-D comparability cuts across clusters. Only the domains it never trained on give an
unbiased comparison, and those domains fall into their own, smaller set of clusters. That count
bounds any claim of beating it, so it is computed here rather than assumed to be 81.

Sequences come from the structures actually scored, not from a reference database, so what is
clustered is exactly what the model saw.
"""

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Code"))
from homology import cluster as homology

AA3 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
       "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
       "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}


def read_sequence(path):
    """One-letter sequence in residue order, from the CA records."""
    residues = []
    for line in Path(path).read_text(errors="replace").splitlines():
        if line[:4] == "ATOM" and line[12:16].strip() == "CA":
            name = line[17:20].strip()
            if name in AA3:
                residues.append((int(line[22:26]), AA3[name]))
    residues.sort()
    return "".join(letter for _, letter in residues)


def kish(weights):
    """Effective number of equally-weighted units carrying the same information."""
    weights = [w for w in weights if w > 0]
    if not weights:
        return 0.0
    return sum(weights) ** 2 / sum(w * w for w in weights)


def describe(label, clusters, contrasts_by_domain):
    sizes = sorted((len(m) for m in clusters.values()), reverse=True)
    per_cluster = [sum(contrasts_by_domain.get(d, 0) for d in members)
                   for members in clusters.values()]
    return {
        "label": label,
        "domains": sum(sizes),
        "clusters": len(clusters),
        "singleton_clusters": sum(1 for s in sizes if s == 1),
        "largest_cluster": sizes[0] if sizes else 0,
        "cluster_sizes_top10": sizes[:10],
        "contrasts": sum(per_cluster),
        "contrasts_per_cluster_median": statistics.median(per_cluster) if per_cluster else 0,
        "effective_clusters_kish": round(kish(per_cluster), 1),
    }


def run(args):
    structures = Path(args.structures)
    contrasts_by_domain, resolved_by_domain = defaultdict(int), defaultdict(int)
    with Path(args.contrasts).open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            contrasts_by_domain[row["domain"]] += 1
            if row["resolved"] == "True":
                resolved_by_domain[row["domain"]] += 1

    sequences = {}
    for domain in sorted(contrasts_by_domain):
        path = structures / (domain + ".pdb")
        if not path.exists():
            raise SystemExit("Missing structure for " + domain)
        sequence = read_sequence(path)
        if not sequence:
            raise SystemExit("Empty sequence for " + domain)
        sequences[domain] = sequence

    print("clustering %d domains (%d pairs, nulls on the top %.0f%% by raw score)"
          % (len(sequences), len(sequences) * (len(sequences) - 1) // 2,
             100 * (1 - args.candidate_quantile)), flush=True)

    result = homology.cluster(
        sequences, z_threshold=args.z_threshold,
        identity_threshold=args.identity_threshold, shuffles=args.shuffles,
        seed=args.seed, candidate_quantile=args.candidate_quantile)
    clusters = result["clusters"]
    membership = {d: root for root, members in clusters.items() for d in members}

    blocks = [describe("all domains", clusters, contrasts_by_domain),
              describe("all domains, resolved contrasts", clusters, resolved_by_domain)]

    # The ThermoMPNN-comparable subset has its own cluster count, which bounds that comparison.
    comparable = None
    if args.membership:
        groups = json.loads(Path(args.membership).read_text())["membership"]
        for name in ("clean", "contaminated"):
            subset = {root: [d for d in members if d in set(groups[name])]
                      for root, members in clusters.items()}
            subset = {r: m for r, m in subset.items() if m}
            blocks.append(describe("%s (ThermoMPNN %s)" % (name, name), subset,
                                   resolved_by_domain))
            if name == "clean":
                comparable = len(subset)

    splits = None
    if args.splits:
        fractions = {}
        for piece in args.splits.split(","):
            key, value = piece.split(":")
            fractions[key.strip()] = float(value)
        # assign_splits returns {split: [domain, ...]}; whole clusters move together and
        # verify_split refuses any assignment that divided one.
        assignment = homology.assign_splits(clusters, fractions, seed=args.seed)
        homology.verify_split(assignment, clusters)
        counts = {}
        for split, domains in sorted(assignment.items()):
            counts[split] = {
                "clusters": len({membership[d] for d in domains}),
                "domains": len(domains),
                "resolved_contrasts": sum(resolved_by_domain.get(d, 0) for d in domains),
            }
        splits = {"fractions": fractions, "seed": args.seed, "counts": counts,
                  "assignment": assignment}

    related = result["related_pairs"]
    summary = {
        "schema_version": 1,
        "purpose": "homology_clusters_and_independent_evidence_for_the_scored_domains",
        "settings": result["settings"],
        "related_pair_count": len(related),
        "blocks": blocks,
        "thermompnn_comparable_clusters": comparable,
        "splits": splits,
        "clusters": clusters,
        "membership": membership,
        "related_pairs": related,
    }
    args.output.resolve().mkdir(parents=True, exist_ok=True)
    (args.output / "domain_clusters.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    print()
    for block in blocks:
        print("%-38s domains %3d  clusters %3d  singletons %3d  largest %2d"
              % (block["label"], block["domains"], block["clusters"],
                 block["singleton_clusters"], block["largest_cluster"]))
        print("%-38s contrasts %6d  effective clusters (Kish) %s"
              % ("", block["contrasts"], block["effective_clusters_kish"]))
    print()
    print("related pairs found: %d" % len(related))
    if splits:
        print("\nsplit assignment (clusters kept whole):")
        for name, c in splits["counts"].items():
            print("  %-12s clusters %3d  domains %3d  resolved contrasts %6d"
                  % (name, c["clusters"], c["domains"], c["resolved_contrasts"]))
    print("\nComplete: " + str(args.output / "domain_clusters.json"))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--structures", required=True)
    parser.add_argument("--contrasts", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--membership", help="thermompnn_split_membership.json")
    parser.add_argument("--splits", help="e.g. 'train:0.7,val:0.15,test:0.15'")
    parser.add_argument("--z-threshold", type=float, default=homology.DEFAULT_Z)
    parser.add_argument("--identity-threshold", type=float, default=homology.DEFAULT_IDENTITY)
    parser.add_argument("--shuffles", type=int, default=homology.DEFAULT_SHUFFLES)
    parser.add_argument("--candidate-quantile", type=float,
                        default=homology.DEFAULT_CANDIDATE_QUANTILE)
    parser.add_argument("--seed", type=int, default=0)
    run(parser.parse_args())


if __name__ == "__main__":
    main()

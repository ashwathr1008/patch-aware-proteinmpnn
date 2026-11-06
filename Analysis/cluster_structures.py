"""Cluster a pair dataset's source structures by homology and check the recorded groups.

The dataset carries a `group` label per structure, used to avoid treating related proteins as
independent evidence. Those labels were assigned by hand as a convenience, and the repository
says so: they do not implement homology clustering. This script replaces the assumption with a
measurement, and reports where the recorded labels and the measured clusters disagree.

Disagreement runs in two directions and they are not equally serious:

  over-merged   two unrelated proteins share a label. Costs statistical power, because the
                analysis treats independent evidence as one source. Safe but wasteful.
  under-merged  two related proteins carry different labels. Invalidates any split built on
                those labels, because patches from one family land on both sides.

Only the second is a correctness problem, so the two are counted separately rather than summed
into one disagreement figure.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Code" / "homology"))
import cluster as homology


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_structures(path):
    """One sequence per structure and chain, with its recorded group and split."""
    dataset = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    recorded, sequences = {}, {}
    for item in dataset["structures"]:
        key = item["structure_id"] + "_" + item["chain"]
        recorded[key] = {"group": item.get("group", ""), "split": item.get("split", "")}
    for pair in dataset["pairs"]:
        key = pair["structure_id"] + "_" + pair["chain"]
        sequence = pair["original_sequence"]
        if key in sequences and sequences[key] != sequence:
            raise ValueError("Conflicting original sequences for " + key)
        sequences[key] = sequence
    missing = set(recorded) - set(sequences)
    if missing:
        raise ValueError("No sequence for " + ", ".join(sorted(missing)))
    return sequences, recorded


def compare(clusters, recorded):
    """Count the two kinds of disagreement between measured clusters and recorded labels."""
    measured = {}
    for root, members in clusters.items():
        for member in members:
            measured[member] = root
    by_label = defaultdict(list)
    for name, info in recorded.items():
        by_label[info["group"]].append(name)
    over_merged = []
    for label, members in sorted(by_label.items()):
        roots = {measured[m] for m in members}
        if len(roots) > 1:
            over_merged.append({"recorded_group": label, "members": sorted(members),
                                "measured_clusters": sorted(roots)})
    under_merged = []
    for root, members in sorted(clusters.items()):
        labels = {recorded[m]["group"] for m in members}
        if len(labels) > 1:
            under_merged.append({"measured_cluster": root, "members": sorted(members),
                                 "recorded_groups": sorted(labels)})
    return {
        "recorded_group_count": len(by_label),
        "measured_cluster_count": len(clusters),
        "over_merged": over_merged,
        "under_merged": under_merged,
        "leakage_risk": bool(under_merged),
        "independent_sources_recorded": len(by_label),
        "independent_sources_measured": len(clusters),
    }


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    sequences, recorded = load_structures(args.pairs.resolve())
    result = homology.cluster(
        sequences, z_threshold=args.z_threshold, identity_threshold=args.identity_threshold,
        shuffles=args.shuffles, seed=args.seed, candidate_quantile=args.candidate_quantile)
    verdict = compare(result["clusters"], recorded)
    summary = {
        "schema_version": 1, "purpose": "homology_clustering_of_source_structures",
        "method": "Smith-Waterman score against a composition-preserving shuffled null; a pair "
                  "is related only when its Z reaches the threshold and its end-to-end "
                  "identity clears the floor. Single-linkage merging.",
        "why_not_identity_alone":
            "Percent identity is unreliable on short domains. Among these structures unrelated "
            "pairs reach 39% identity, so a 30% threshold would chain them into one cluster.",
        "settings": result["settings"],
        "sequences": {k: len(v) for k, v in sorted(sequences.items())},
        "clusters": result["clusters"],
        "related_pairs": result["related_pairs"],
        "comparison_with_recorded_groups": verdict,
        "evidence": result["evidence"],
    }
    if args.splits:
        fractions = {}
        for item in args.splits.split(","):
            name, _, value = item.partition(":")
            fractions[name.strip()] = float(value)
        assignment = homology.assign_splits(result["clusters"], fractions, seed=args.seed)
        homology.verify_split(assignment, result["clusters"])
        summary["proposed_split"] = assignment
    write_json(output / "homology_clusters.json", summary)

    print("{} structures -> {} clusters (recorded groups: {})".format(
        len(sequences), result["cluster_count"], verdict["recorded_group_count"]), flush=True)
    if result["related_pairs"]:
        for pair in result["related_pairs"]:
            print("  related: {} / {}".format(pair["a"], pair["b"]), flush=True)
    else:
        print("  no pair reaches the relatedness threshold", flush=True)
    for entry in verdict["over_merged"]:
        print("  over-merged: '{}' holds {} unrelated proteins ({})".format(
            entry["recorded_group"], len(entry["members"]), ", ".join(entry["members"])),
            flush=True)
    for entry in verdict["under_merged"]:
        print("  LEAKAGE RISK: related proteins carry different group labels: {}".format(
            ", ".join(entry["members"])), flush=True)
    if not verdict["under_merged"]:
        print("  no leakage risk: no related pair is split across recorded groups", flush=True)
    print("Complete: " + str(output / "homology_clusters.json"), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--z-threshold", type=float, default=homology.DEFAULT_Z)
    parser.add_argument("--identity-threshold", type=float, default=homology.DEFAULT_IDENTITY)
    parser.add_argument("--shuffles", type=int, default=homology.DEFAULT_SHUFFLES)
    parser.add_argument("--candidate-quantile", type=float,
                        default=homology.DEFAULT_CANDIDATE_QUANTILE)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--splits", help="e.g. 'development:0.6,holdout:0.4'")
    run(parser.parse_args())


if __name__ == "__main__":
    main()

"""Join Vani's v3 cohort to the HSSP domain clusters, and audit the split for leakage.

Two things make this audit worth writing down rather than eyeballing.

The first is that the join is not the obvious one. The v3 manifest names parents
`1E0L.pdb`; the HSSP cluster file names domains `1E0L`. Exact-string overlap between
the two sets is *zero*. Stripping the suffix recovers 140 of 148. The eight that remain are
genuinely outside the clustered set: six PDB codes (1WR4, 1ZHC, 2K9D, 2LCL, 2M9E, 5OAO)
and two de novo designs (EHEE_rd2_1106, HHH_rd4_0395). Case is not the issue, and the
designs are not the issue either -- seven of the cohort's nine designed parents do match,
because the HSSP set contains nineteen of them. An unadjusted join
therefore matches nothing -- and, far worse, a zero-overlap join finds no family that
crosses a partition boundary and so reports a clean pass. The failure mode and the
success look identical. So this script asserts the expected match count and refuses to
report anything if the count is wrong, rather than asserting the absence of leakage.

The second is that both clusterings independently produce 84 groups. v3's groups come
from global Levenshtein similarity over measured parents at threshold 0.4; the HSSP
clusters come from structural/profile relationships over 208 domains. The agreement is
a coincidence, and it would let a reader believe the two partitions are the same object.
This script reports the cross-tabulation so the coincidence cannot pass unexamined.

The question the audit answers: if we adopt the HSSP families as the CV unit over the
v3 cohort, does any HSSP family have v3 members sitting in more than one of Vani's
inherited train/val/test partitions? Where it does, the inherited split leaks and must
be replaced rather than reused.
"""

import argparse
import json
import pathlib
from collections import Counter, defaultdict

EXPECTED_MATCHES = 140
EXPECTED_V3_PARENTS = 148
EXPECTED_HSSP_DOMAINS = 208


def domain_code(parent_id):
    """`1E0L.pdb` -> `1E0L`. The only normalisation the join needs, and the whole trap."""
    return parent_id[:-4] if parent_id.lower().endswith(".pdb") else parent_id


def v3_partition(groups, parent_ids):
    """Map each v3 parent to its v3 similarity group representative.

    `groups` maps a representative parent to its members. A parent absent from every
    member list is its own singleton group, which is how the manifest stores them.
    """
    assignment = {}
    for representative, members in groups.items():
        for member in members:
            assignment[member] = representative
    for parent in parent_ids:
        assignment.setdefault(parent, parent)
    return assignment


def load_v3(path):
    stage2 = json.loads(pathlib.Path(path).read_text())
    parents = [record["parent_id"] for record in stage2["parents"]]
    split_of = {record["parent_id"]: record["split"] for record in stage2["parents"]}
    return parents, stage2["groups"], split_of


def load_hssp(path):
    clusters = json.loads(pathlib.Path(path).read_text())
    assignment = clusters["splits"]["assignment"]
    split_of = {}
    for split, domains in assignment.items():
        for domain in domains:
            split_of[domain] = split
    return clusters["membership"], split_of, clusters["clusters"]


def audit(v3_stage2, hssp_clusters, strict=True):
    parents, v3_groups, v3_split = load_v3(v3_stage2)
    membership, hssp_split, hssp_clusters_by_rep = load_hssp(hssp_clusters)

    report = {
        "v3_parents": len(parents),
        "hssp_domains": len(membership),
        "exact_string_overlap": len(set(parents) & set(membership)),
    }

    matched, unmatched = {}, []
    for parent in parents:
        code = domain_code(parent)
        if code in membership:
            matched[parent] = code
        else:
            unmatched.append(parent)
    report["matched"] = len(matched)
    report["unmatched"] = sorted(unmatched)

    if strict and len(matched) != EXPECTED_MATCHES:
        raise SystemExit(
            "Join produced %d matches, expected %d. The audit is void: a join that "
            "matches too little reports an absence of leakage that is an artefact of "
            "the join, not a property of the split. Fix the join, do not relax the "
            "check." % (len(matched), EXPECTED_MATCHES))

    # Which HSSP families does the v3 cohort actually touch, and how much of each?
    family_members = defaultdict(list)
    for parent, code in matched.items():
        family_members[membership[code]].append(parent)
    report["hssp_families_touched"] = len(family_members)
    report["hssp_families_total"] = len(hssp_clusters_by_rep)

    coverage = []
    for family, members in sorted(family_members.items()):
        whole = hssp_clusters_by_rep[family]
        coverage.append({"family": family, "v3_members": len(members),
                         "family_size": len(whole),
                         "covered": round(len(members) / len(whole), 3)})
    report["family_coverage"] = sorted(coverage, key=lambda r: -r["v3_members"])

    # The leakage question, against Vani's inherited v3 similarity groups.
    v3_group_of = v3_partition(v3_groups, parents)
    report["v3_groups"] = len(set(v3_group_of[p] for p in parents))

    crossing = []
    for family, members in sorted(family_members.items()):
        groups = sorted(set(v3_group_of[p] for p in members))
        if len(groups) > 1:
            crossing.append({"family": family, "v3_groups": groups,
                             "members": sorted(members)})
    report["families_crossing_v3_groups"] = crossing
    report["families_crossing_count"] = len(crossing)

    # The decisive test. Vani's v3 parents each carry an inherited train/val/test label.
    # If one HSSP family has members on both sides of that boundary, then training on
    # her train split and reporting on her test split evaluates a model on a family it
    # was trained on, and the inherited split cannot be reused as-is.
    split_crossing = []
    for family, members in sorted(family_members.items()):
        by_split = defaultdict(list)
        for parent in members:
            by_split[v3_split[parent]].append(parent)
        if len(by_split) > 1:
            split_crossing.append({
                "family": family,
                "splits": {split: sorted(names) for split, names in sorted(by_split.items())},
                "members": len(members)})
    report["families_crossing_v3_train_test"] = split_crossing
    report["families_crossing_v3_train_test_count"] = len(split_crossing)
    report["parents_in_leaking_families"] = sum(r["members"] for r in split_crossing)

    # What the leakage costs the reported number: a test parent whose HSSP family also
    # contains a train parent is not held out in any meaningful sense.
    train_families = {family for family, members in family_members.items()
                      if any(v3_split[p] == "train" for p in members)}
    contaminated = sorted(parent for parent, code in matched.items()
                          if v3_split[parent] == "test"
                          and membership[code] in train_families)
    test_parents = [p for p in parents if v3_split[p] == "test"]
    report["test_parents"] = len(test_parents)
    report["test_parents_sharing_a_family_with_train"] = contaminated
    report["test_contamination_fraction"] = (
        round(len(contaminated) / len(test_parents), 3) if test_parents else None)

    # And against the HSSP file's own committed train/val/test assignment, which is the
    # split we would adopt instead. A family is one cluster and the file assigns whole
    # clusters, so by construction it cannot cross -- this verifies that rather than
    # assuming it.
    hssp_crossing = []
    for family, members in sorted(family_members.items()):
        splits = sorted(set(hssp_split.get(domain_code(p), "UNASSIGNED") for p in members))
        if len(splits) > 1:
            hssp_crossing.append({"family": family, "splits": splits})
    report["families_crossing_hssp_splits"] = hssp_crossing

    # Cross-tabulation of the two 84-group partitions, to defuse the coincidence.
    pairs = Counter((v3_group_of[p], membership[code]) for p, code in matched.items())
    v3_to_hssp = defaultdict(set)
    hssp_to_v3 = defaultdict(set)
    for (group, family) in pairs:
        v3_to_hssp[group].add(family)
        hssp_to_v3[family].add(group)
    report["partition_comparison"] = {
        "v3_groups_over_matched_parents": len(v3_to_hssp),
        "hssp_families_over_matched_parents": len(hssp_to_v3),
        "v3_groups_spanning_multiple_hssp_families":
            sum(1 for v in v3_to_hssp.values() if len(v) > 1),
        "hssp_families_spanning_multiple_v3_groups":
            sum(1 for v in hssp_to_v3.values() if len(v) > 1),
        "identical_partitions": all(len(v) == 1 for v in v3_to_hssp.values())
        and all(len(v) == 1 for v in hssp_to_v3.values()),
    }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v3-stage2", required=True,
                        help="vani_training_v3/stage2_splits.json")
    parser.add_argument("--hssp-clusters", required=True,
                        help="Analysis/homology_domains/hssp/domain_clusters.json")
    parser.add_argument("--output", help="write the report as JSON")
    parser.add_argument("--allow-unexpected-match-count", action="store_true",
                        help="only for investigating a changed cohort")
    arguments = parser.parse_args()

    report = audit(arguments.v3_stage2, arguments.hssp_clusters,
                   strict=not arguments.allow_unexpected_match_count)
    text = json.dumps(report, indent=2)
    if arguments.output:
        pathlib.Path(arguments.output).write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()

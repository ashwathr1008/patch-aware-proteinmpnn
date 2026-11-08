"""Split the ThermoMPNN-D comparison by whether it trained on each domain.

ThermoMPNN-D was trained on the same mega-scale dataset our contrasts come from, so its score
against ours is only a baseline for domains it did not train on. Its configuration names a
`mega_splits.pkl` that is absent from the ThermoMPNN-D repository, but the original ThermoMPNN
repository ships it, and it labels splits at the domain level, which is the granularity needed.

    https://github.com/Kuhlman-Lab/ThermoMPNN/raw/main/dataset_splits/mega_splits.pkl
    sha256 9e06230fe8febd8f07f8fab374f153328d4bdde20b78de15ead93b39e61ca1eb

Not vendored here: it is third-party data, 16 KB, and fetched per run. The checksum above is
what this script verifies, so a changed upstream file is noticed rather than silently used.

The pickle references numpy array reconstruction and nothing else, which was checked with
`pickletools.dis` before it was ever loaded. Loading a pickle executes whatever it names, so it
is read through an unpickler that refuses every global except those three, rather than trusted
because of where it came from.

Two domain groups are clean. Those in ThermoMPNN's own test split, and those absent from all of
its splits, which it therefore never saw. Validation-split domains count as contaminated: they
drove model selection even if no gradient came from them.

**A confound to state alongside the result.** Clean and contaminated domains are not a
controlled contrast. The splits were built by homology clustering, so the two groups differ in
composition, and frozen ProteinMPNN -- which has no exposure to ThermoMPNN's splits at all --
also scores worse on the clean group. Some of the difference between groups is domain
difficulty rather than contamination. The paired comparison *within* the clean group is the
part that remains valid, because both models see the same domains there.
"""

import argparse
import csv
import hashlib
import json
import pickle
import statistics
from pathlib import Path

EXPECTED_SHA256 = "9e06230fe8febd8f07f8fab374f153328d4bdde20b78de15ead93b39e61ca1eb"
ALLOWED_GLOBALS = {("numpy", "ndarray"), ("numpy.core.multiarray", "_reconstruct"),
                   ("numpy", "dtype"), ("ndarray", "dtype")}
MIN_DOMAINS = 5


class RestrictedUnpickler(pickle.Unpickler):
    """Refuses every global but numpy array reconstruction."""

    def find_class(self, module, name):
        if (module, name) not in ALLOWED_GLOBALS:
            raise pickle.UnpicklingError("Refused to load %s.%s from the splits file"
                                         % (module, name))
        return super().find_class(module, name)


def load_splits(path, require_checksum=True):
    data = Path(path).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if require_checksum and digest != EXPECTED_SHA256:
        raise SystemExit("Splits checksum mismatch.\n  expected %s\n  found    %s"
                         % (EXPECTED_SHA256, digest))
    with Path(path).open("rb") as handle:
        return RestrictedUnpickler(handle).load(), digest


def domain_names(entries):
    """Bare domain names. The dataset prefixes some re-released entries with `v2_`."""
    out = set()
    for raw in entries:
        stem = str(raw)
        if stem.endswith(".pdb"):
            stem = stem[:-4]
        out.add(stem)
        if stem.startswith("v2_"):
            out.add(stem[3:])
    return out


def classify(splits, domains):
    train = domain_names(splits["train"])
    val = domain_names(splits["val"])
    test = domain_names(splits["test"])
    known = train | val | test
    groups = {"in_train": [], "in_val": [], "in_test": [], "unseen": []}
    for domain in domains:
        if domain in train:
            groups["in_train"].append(domain)
        elif domain in val:
            groups["in_val"].append(domain)
        elif domain in test:
            groups["in_test"].append(domain)
        elif domain not in known:
            groups["unseen"].append(domain)
    groups["clean"] = sorted(set(groups["in_test"]) | set(groups["unseen"]))
    groups["contaminated"] = sorted(set(groups["in_train"]) | set(groups["in_val"]))
    sizes = {"train": len(train), "val": len(val), "test": len(test), "total": len(known)}
    return groups, sizes


def paired(rows, domains, binomtest):
    present = [d for d in domains if d in rows]
    if len(present) < MIN_DOMAINS:
        return {"domains": len(present), "note": "too few domains to summarize"}
    thermo = [rows[d]["thermompnn_d"] for d in present]
    frozen = [rows[d]["frozen"] for d in present]
    better = sum(1 for d in present if rows[d]["thermompnn_d"] < rows[d]["frozen"])
    nonties = sum(1 for d in present if rows[d]["thermompnn_d"] != rows[d]["frozen"])
    return {
        "domains": len(present),
        "thermompnn_d_median": statistics.median(thermo),
        "frozen_median": statistics.median(frozen),
        "median_gap": statistics.median(thermo) - statistics.median(frozen),
        "thermompnn_stronger": better, "nonties": nonties,
        "p_value": binomtest(better, nonties, 0.5).pvalue if nonties else None,
    }


def run(args):
    from scipy.stats import binomtest

    splits, digest = load_splits(args.splits, not args.allow_checksum_mismatch)
    domains = sorted({r["domain"] for r in csv.DictReader(
        Path(args.contrasts).open(newline="", encoding="utf-8-sig"))})
    groups, sizes = classify(splits, domains)

    comparison = json.loads(Path(args.comparison).read_text())
    rows = {e["domain"]: e for e in comparison["per_domain"]}

    blocks = {name: paired(rows, groups[name], binomtest)
              for name in ("clean", "contaminated", "in_test", "unseen")}
    blocks["all"] = paired(rows, sorted(rows), binomtest)

    summary = {
        "schema_version": 1,
        "purpose": "thermompnn_d_comparison_split_by_training_contamination",
        "splits_source": ("https://github.com/Kuhlman-Lab/ThermoMPNN/raw/main/"
                          "dataset_splits/mega_splits.pkl"),
        "splits_sha256": digest,
        "split_sizes": sizes,
        "our_domains": len(domains),
        "membership_counts": {k: len(groups[k]) for k in
                              ("in_train", "in_val", "in_test", "unseen",
                               "clean", "contaminated")},
        "confound": ("Clean and contaminated groups are not a controlled contrast: the splits "
                     "were built by homology clustering, and frozen ProteinMPNN also scores "
                     "worse on the clean group despite no exposure to these splits. The "
                     "paired comparison within the clean group is the valid part."),
        "comparisons": blocks,
        "membership": {k: groups[k] for k in
                       ("in_train", "in_val", "in_test", "unseen", "clean", "contaminated")},
    }
    args.output.resolve().mkdir(parents=True, exist_ok=True)
    (args.output / "thermompnn_split_membership.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    print("ThermoMPNN splits: train %d, val %d, test %d  (%d domains)"
          % (sizes["train"], sizes["val"], sizes["test"], sizes["total"]))
    print("our domains      : %d" % len(domains))
    for name in ("in_train", "in_val", "in_test", "unseen"):
        print("  %-10s %3d (%.1f%%)" % (name, len(groups[name]),
                                        100 * len(groups[name]) / len(domains)))
    print()
    for name in ("all", "clean", "in_test", "unseen", "contaminated"):
        b = blocks[name]
        if "thermompnn_d_median" not in b:
            print("  %-14s %s" % (name, b.get("note"))); continue
        print("  %-14s n=%-4d ThermoMPNN-D %.4f  frozen %.4f  gap %+.4f  "
              "TM better %d/%d  p=%.3g"
              % (name, b["domains"], b["thermompnn_d_median"], b["frozen_median"],
                 b["median_gap"], b["thermompnn_stronger"], b["nonties"], b["p_value"]))
    print("\nComplete: " + str(args.output / "thermompnn_split_membership.json"))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--splits", required=True, help="mega_splits.pkl")
    parser.add_argument("--contrasts", required=True, help="swap_contrasts.csv")
    parser.add_argument("--comparison", required=True, help="thermompnn_comparison.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-checksum-mismatch", action="store_true",
                        help="proceed if upstream changed the splits file")
    run(parser.parse_args())


if __name__ == "__main__":
    main()

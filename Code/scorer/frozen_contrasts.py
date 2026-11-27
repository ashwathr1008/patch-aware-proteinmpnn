"""Reduce the frozen model's per-variant scores to one signed contrast per arrangement pair.

The frozen baseline already exists as one row per variant per decoding order in
`swap_variant_scores.csv`. Rebuilding the comparison from that file rather than re-running
the model keeps the baseline *literally* the one behind the committed -0.501, and costs
seconds of CPU instead of half an hour of GPU.

**Signs, since this is where they are easiest to get wrong.** `full_nll_mean` is a mean
negative log likelihood: lower means the model prefers that arrangement. A preference score
that rises with preference is therefore `-nll`, and the frozen contrast is

    q_frozen = (-nll(A)) - (-nll(B)) = nll(B) - nll(A)

so `q_frozen > 0` means the model prefers A, matching `d > 0` meaning A is the more stable
arrangement. Agreement is then a *positive* correlation between `q_frozen` and `d`, which is
the same statement as the committed summary's "agreement is a negative correlation" between
the nll difference and measured stability. Orders are averaged before differencing, matching
the committed run.

Arrangement A carries `aa_i` at `site_i`; the scores file keys each variant by the ordered
identity pair it placed, so A is the row `(aa_i, aa_j)` and B is the row `(aa_j, aa_i)`.
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def load_variant_scores(path, column="full_nll_mean"):
    """(domain, site_i, site_j, aa_at_i, aa_at_j) -> mean of `column` over decoding orders."""
    totals, counts = defaultdict(float), defaultdict(int)
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            key = (row["domain"], int(row["site_i"]), int(row["site_j"]),
                   row["aa_i"], row["aa_j"])
            totals[key] += float(row[column])
            counts[key] += 1
    return {key: totals[key] / counts[key] for key in totals}, dict(counts)


def frozen_contrast(scores, domain, site_i, site_j, aa_i, aa_j):
    """nll(B) - nll(A): positive when the frozen model prefers arrangement A."""
    a = scores.get((domain, site_i, site_j, aa_i, aa_j))
    b = scores.get((domain, site_i, site_j, aa_j, aa_i))
    if a is None or b is None:
        return None
    return b - a


def join(contrasts_path, scores_path, column="full_nll_mean"):
    scores, counts = load_variant_scores(scores_path, column)
    orders = sorted(set(counts.values()))
    rows, missing = [], 0
    with Path(contrasts_path).open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            value = frozen_contrast(scores, row["domain"], int(row["site_i"]),
                                    int(row["site_j"]), row["aa_i"], row["aa_j"])
            if value is None:
                missing += 1
                continue
            rows.append({"contrast_id": row["contrast_id"], "q_frozen": value})
    return rows, {"variants_scored": len(scores), "orders_per_variant": orders,
                  "contrasts_joined": len(rows), "contrasts_without_frozen_scores": missing}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--contrasts", required=True, type=Path,
                        help="contrasts.csv from extract_patch_features.py")
    parser.add_argument("--scores", required=True, type=Path,
                        help="swap_variant_scores.csv from score_swaps_full.py")
    parser.add_argument("--column", default="full_nll_mean")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    rows, record = join(args.contrasts, args.scores, args.column)
    if not rows:
        raise ValueError("No contrasts could be joined to frozen scores")
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "frozen_contrasts.csv").open("w", newline="", encoding="utf-8") as h:
        writer = csv.DictWriter(h, fieldnames=["contrast_id", "q_frozen"])
        writer.writeheader()
        writer.writerows(rows)
    record["sign_convention"] = ("q_frozen = nll(B) - nll(A); positive means the frozen model "
                                "prefers arrangement A, matching d > 0")
    (args.output / "frozen_join_record.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main()

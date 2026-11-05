"""Figure for the first-order/joint decomposition of the arrangement gap.

Left panel: per source protein and checkpoint, the order-averaged full-chain gap split into
its first-order burial term and its joint residual, as stacked totals in nats per chain.
Right panel: the joint residual of every pair against its full gap, so the reader can see
that the residual stays small and mostly positive across the whole range of gap sizes.

Colour encodes the decomposition term in both panels, never the checkpoint: the checkpoint is
carried by the tick label on the left and by marker shape on the right. Slots one and two of
the documented categorical palette are used, which are validated for all-pairs separation.
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path
import statistics

FIRST_ORDER_COLOR = "#2a78d6"
JOINT_COLOR = "#eb6834"
TEXT_PRIMARY = "#2b2b2b"
TEXT_SECONDARY = "#595959"
AXIS_COLOR = "#b4b4b4"
MARKERS = {"vanilla": "o", "soluble": "^"}
REQUIRED = frozenset(
    "pair_id structure_id chain model_type delta_full_total delta_fo_total delta_joint_total".split())


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED - set(reader.fieldnames or ())
        if missing:
            raise ValueError("first_order_scores.csv is missing columns: " + ", ".join(sorted(missing)))
        rows = list(reader)
    if not rows:
        raise ValueError("first_order_scores.csv contains no rows")
    for row in rows:
        for field in ("delta_full_total", "delta_fo_total", "delta_joint_total"):
            row[field] = float(row[field])
    unexpected = {r["model_type"] for r in rows} - set(MARKERS)
    if unexpected:
        raise ValueError("Unknown checkpoint label: " + ", ".join(sorted(unexpected)))
    return rows


def protein_means(rows):
    """Mean decomposition per source protein and checkpoint, ordered by descending full gap."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["structure_id"], row["chain"], row["model_type"])].append(row)
    records = [{
        "label": "{}:{} {}".format(key[0], key[1], key[2]),
        "structure": "{}:{}".format(key[0], key[1]), "model_type": key[2],
        "pair_count": len(group),
        "first_order": statistics.fmean(r["delta_fo_total"] for r in group),
        "joint": statistics.fmean(r["delta_joint_total"] for r in group),
        "full": statistics.fmean(r["delta_full_total"] for r in group),
    } for key, group in grouped.items()]
    order = {record["structure"]: 0.0 for record in records}
    for record in records:
        order[record["structure"]] += record["full"]
    # Group the two checkpoints of one protein together, proteins by descending mean gap.
    records.sort(key=lambda r: (-order[r["structure"]], r["model_type"]))
    return records


def draw(rows, destination, dpi):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    records = protein_means(rows)
    figure, (left, right) = plt.subplots(
        1, 2, figsize=(11.2, 4.8), gridspec_kw={"width_ratios": [1.15, 1.0]})

    positions = list(range(len(records)))[::-1]
    for position, record in zip(positions, records):
        left.barh(position, record["first_order"], height=0.68,
                  color=FIRST_ORDER_COLOR, edgecolor="white", linewidth=1.0, zorder=3)
        left.barh(position, record["joint"], height=0.68, left=record["first_order"],
                  color=JOINT_COLOR, edgecolor="white", linewidth=1.0, zorder=3)
        # One direct label per bar carries the share the burial term accounts for.
        left.text(record["full"] + 0.45, position,
                  "{:.0f}%".format(100 * record["first_order"] / record["full"]),
                  va="center", ha="left", fontsize=7.5, color=TEXT_SECONDARY, zorder=4)
    left.set_yticks(positions)
    left.set_yticklabels([r["label"] for r in records], fontsize=8, color=TEXT_PRIMARY)
    left.set_xlabel("Mean gap, permuted minus original (nats per chain)",
                    fontsize=8.5, color=TEXT_PRIMARY)
    left.set_title("Burial accounts for most of every protein's gap",
                   fontsize=9.5, color=TEXT_PRIMARY, loc="left", pad=10)
    left.set_xlim(0, max(r["full"] for r in records) * 1.14)
    left.xaxis.grid(True, color=AXIS_COLOR, linewidth=0.6, alpha=0.5, zorder=0)
    left.set_axisbelow(True)
    left.legend(handles=[
        plt.Rectangle((0, 0), 1, 1, color=FIRST_ORDER_COLOR, label="first-order burial"),
        plt.Rectangle((0, 0), 1, 1, color=JOINT_COLOR, label="joint residual")],
        frameon=False, fontsize=8, loc="lower right", labelcolor=TEXT_SECONDARY)

    for model_type in sorted({r["model_type"] for r in rows}):
        group = [r for r in rows if r["model_type"] == model_type]
        right.scatter([r["delta_full_total"] for r in group],
                      [r["delta_joint_total"] for r in group],
                      s=30, marker=MARKERS[model_type], facecolor=JOINT_COLOR,
                      edgecolor="white", linewidth=0.7, alpha=0.85, zorder=3)
    right.axhline(0, color=TEXT_SECONDARY, linewidth=1.0, zorder=2)
    right.set_xlabel("Full-chain gap (nats per chain)", fontsize=8.5, color=TEXT_PRIMARY)
    right.set_ylabel("Joint residual (nats per chain)", fontsize=8.5, color=TEXT_PRIMARY)
    right.set_title("The residual stays small and mostly positive",
                    fontsize=9.5, color=TEXT_PRIMARY, loc="left", pad=10)
    right.grid(True, color=AXIS_COLOR, linewidth=0.6, alpha=0.5, zorder=0)
    right.set_axisbelow(True)
    right.legend(handles=[
        Line2D([], [], linestyle="none", marker=MARKERS[m], markersize=6,
               markerfacecolor=JOINT_COLOR, markeredgecolor="white", label=m)
        for m in sorted(MARKERS)], frameon=False, fontsize=8, loc="upper left",
        labelcolor=TEXT_SECONDARY, title="checkpoint", title_fontsize=8)

    for axis in (left, right):
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["left", "bottom"]].set_color(AXIS_COLOR)
        axis.tick_params(colors=TEXT_SECONDARY, labelsize=8, length=3)
    figure.tight_layout()
    written = []
    for suffix in (".png", ".pdf"):
        target = destination.with_suffix(suffix)
        figure.savefig(target, dpi=dpi, bbox_inches="tight")
        written.append(target)
    plt.close(figure)
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scores", type=Path, required=True,
                        help="first_order_scores.csv from score_first_order.py")
    parser.add_argument("--output", type=Path, required=True, help="path without a suffix")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    rows = read_rows(args.scores)
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    for path in draw(rows, args.output.resolve(), args.dpi):
        print("Wrote " + str(path), flush=True)


if __name__ == "__main__":
    main()

"""Score the SAP surrogate on the matched pairs and on the generated candidate pools.

Three questions are answered from one burial computation per structure.

1. Descriptor sensitivity. The hydrophobic-contact descriptor used so far is a count, and it
   ties in roughly half the pair comparisons, so it cannot discriminate there at all. A
   continuous burial-weighted form should tie far less often.

2. Is the model's first-order term the SAP statistic? The decomposition found that most of the
   arrangement gap is first-order, and a first-order term driven by burial would *be*
   burial-weighted composition, which is SAP in closed form. That inference needs testing
   rather than assuming: if delta_fo and the SAP change track each other, the model is
   reproducing a 2009 formula; if they do not, the model's marginals encode something else.
   Composition is held exactly equal within every pair, so the SAP change here comes only
   from rearrangement.

3. Does the model's ranking carry developability signal among its own samples? The 84/85 result
   compares a native sequence against a deliberately scrambled one. The deployment task is
   ranking among ProteinMPNN's own candidates, which it finds comparably plausible. Within each
   structure and checkpoint pool, the generated sequences' NLL rank is compared against their
   SAP rank. Composition varies freely here, unlike in the pairs.

Burial comes from the backbone plus a virtual CB and never from residue identity, so the same
scaffold scores every assignment on a structure. See Code/descriptors/sap.py.
"""
import argparse
import csv
from collections import defaultdict
import itertools
import json
import math
from pathlib import Path
import statistics
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Code" / "baseline"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Code" / "descriptors"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Code" / "scoring"))
import run_baseline as baseline
import sap as surrogate
import score_pairs as pair_scoring

TIE_TOLERANCE = 1e-9
# Enumerating a pool's orderings is exact but factorial; above this, report no p-value.
MAXIMUM_EXACT_POOL = 8
PAIR_FIELDS = (
    "pair_id structure_id chain split group patch_id patch_size changed_identity_count "
    "original_sap_simple permuted_sap_simple delta_sap_simple "
    "original_sap_window permuted_sap_window delta_sap_window "
    "burial_patch_mean hydrophobic_fraction_delta contact_descriptor_tie sap_simple_tie "
    "delta_fo_total delta_full_total model_type"
).split()
CANDIDATE_FIELDS = (
    "sequence_id structure_id chain model_type record_type sample_index length "
    "generation_score native_recovery sap_simple sap_window"
).split()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, fields, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def format_optional(value, pattern="{:+.3f}"):
    """Render a statistic that may be undefined, so a summary never fails on None."""
    return "n/a" if value is None else pattern.format(value)


def pearson(xs, ys):
    if len(xs) < 3:
        return None
    mean_x, mean_y = statistics.fmean(xs), statistics.fmean(ys)
    dx = [x - mean_x for x in xs]
    dy = [y - mean_y for y in ys]
    denominator = math.sqrt(sum(v * v for v in dx) * sum(v * v for v in dy))
    if denominator <= 0:
        return None
    return sum(a * b for a, b in zip(dx, dy)) / denominator


def ranks(values):
    """Average ranks, so ties do not bias the rank correlation."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    result = [0.0] * len(values)
    position = 0
    while position < len(order):
        stop = position
        while stop + 1 < len(order) and values[order[stop + 1]] == values[order[position]]:
            stop += 1
        shared = (position + stop) / 2.0
        for index in range(position, stop + 1):
            result[order[index]] = shared
        position = stop + 1
    return result


def spearman(xs, ys):
    return pearson(ranks(xs), ranks(ys))


def exact_spearman_p(xs, ys):
    """Two-sided exact p by enumerating orderings; None when the pool is too large.

    The pools here hold a handful of samples each, where an asymptotic p-value is not
    trustworthy and a single hardcoded critical value would silently assume one pool size.
    """
    observed = spearman(xs, ys)
    if observed is None or len(xs) > MAXIMUM_EXACT_POOL:
        return observed, None
    total = extreme = 0
    for permuted in itertools.permutations(ys):
        value = spearman(xs, list(permuted))
        total += 1
        if value is not None and abs(value) >= abs(observed) - 1e-12:
            extreme += 1
    return observed, extreme / total


def sign_test(values):
    positive = sum(1 for v in values if v > TIE_TOLERANCE)
    negative = sum(1 for v in values if v < -TIE_TOLERANCE)
    trials = positive + negative
    result = {"positive": positive, "negative": negative,
              "ties": len(values) - trials, "nonties": trials}
    if not trials:
        return {**result, "p_value": None}
    tail = sum(math.comb(trials, k) for k in range(min(positive, negative) + 1)) / (2 ** trials)
    return {**result, "p_value": min(1.0, 2 * tail)}


def parse_staged_backbone(path, length):
    """Read the staged N/CA/C/O file into an (L, 4, 3) array in residue order."""
    import numpy as np
    coordinates, current = [], []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line[:4] != "ATOM":
            continue
        current.append([float(line[30 + 8 * i:38 + 8 * i]) for i in range(3)])
        if len(current) == 4:
            coordinates.append(current)
            current = []
    if current:
        raise ValueError("Staged backbone ended mid-residue")
    array = np.asarray(coordinates, dtype="float64")
    if array.shape != (length, 4, 3):
        raise ValueError("Staged backbone shape does not match the reported length")
    return array


def build_scaffolds(dataset, dataset_root, window_radius, sphere_points):
    """One identity-independent burial profile and neighbourhood per structure."""
    scaffolds = {}
    with tempfile.TemporaryDirectory() as workspace:
        for item in dataset["structures"]:
            key = (item["structure_id"], item["chain"])
            source = (dataset_root / item["pdb_path"]).resolve()
            expected = item.get("sha256", "")
            if baseline.sha256(source) != expected:
                raise ValueError("Structure input checksum mismatch: " + key[0])
            staged = Path(workspace) / (key[0] + "_" + key[1] + ".pdb")
            info = baseline.stage_backbone(source, key[1], staged)
            backbone = parse_staged_backbone(staged, info["length"])
            scaffolds[key] = {
                "burial": surrogate.burial_profile(backbone, sphere_points),
                "window": surrogate.window_matrix(backbone, window_radius),
                "native_sequence": info["native_sequence"], "length": info["length"],
                "split": item.get("split"), "group": item.get("group"),
            }
    return scaffolds


def score_pairs(dataset, scaffolds, decomposition):
    """Score every pair, after checking each one still satisfies the premise.

    The claim this script supports is that composition is held exactly equal, so the SAP
    change reflects only rearrangement. That premise is checked here rather than assumed:
    a regenerated manifest with different residue selection or shifted patch indices would
    otherwise be scored against the wrong burial positions and reported as a rearrangement
    effect, with only the length check inside sap_profile standing in the way.
    """
    rows = []
    for pair in dataset["pairs"]:
        key = (pair["structure_id"], pair["chain"])
        if key not in scaffolds:
            raise ValueError("Pair references an absent structure: " + pair["pair_id"])
        scaffold = scaffolds[key]
        score_pairs_validate(pair, scaffold)
        positions = [i - 1 for i in pair["patch_indices"]]
        original = surrogate.descriptors(
            scaffold["burial"], pair["original_sequence"], scaffold["window"], positions)
        permuted = surrogate.descriptors(
            scaffold["burial"], pair["permuted_sequence"], scaffold["window"], positions)
        controls = pair.get("controls", {})
        contact_delta = float(controls.get("hydrophobic_fraction_delta", 0.0))
        record = {
            "pair_id": pair["pair_id"], "structure_id": key[0], "chain": key[1],
            "split": scaffold["split"], "group": scaffold["group"],
            "patch_id": pair["patch_id"], "patch_size": len(positions),
            "changed_identity_count": sum(
                a != b for a, b in zip(pair["original_sequence"], pair["permuted_sequence"])),
            "burial_patch_mean": original["burial_patch_mean"],
            "hydrophobic_fraction_delta": contact_delta,
            "contact_descriptor_tie": abs(contact_delta) < TIE_TOLERANCE,
        }
        for name in ("sap_simple", "sap_window"):
            record["original_" + name] = original[name]
            record["permuted_" + name] = permuted[name]
            record["delta_" + name] = permuted[name] - original[name]
        record["sap_simple_tie"] = abs(record["delta_sap_simple"]) < TIE_TOLERANCE
        reference = decomposition.get(pair["pair_id"], {})
        record["delta_fo_total"] = reference.get("delta_fo_total")
        record["delta_full_total"] = reference.get("delta_full_total")
        record["model_type"] = reference.get("model_type")
        rows.append(record)
    return rows


def score_pairs_validate(pair, scaffold):
    """Reuse the scoring module's invariants, then tie the pair to this backbone."""
    pair_scoring.validate_pair(pair, scaffold["length"], scaffold["native_sequence"])


def score_candidates(candidates, scaffolds):
    """Score candidates against the backbone of their own structure AND chain."""
    rows, skipped = [], []
    for row in candidates:
        key = (row["structure_id"], row.get("chain", ""))
        scaffold = scaffolds.get(key)
        if scaffold is None:
            skipped.append({"sequence_id": row.get("sequence_id", ""),
                            "structure_id": row["structure_id"],
                            "chain": row.get("chain", ""),
                            "reason": "no scaffold for this structure and chain"})
            continue
        sequence = row["sequence"]
        if len(sequence) != scaffold["length"]:
            raise ValueError("Candidate length differs from its backbone: " + row["sequence_id"])
        rows.append({
            "sequence_id": row["sequence_id"], "structure_id": row["structure_id"],
            "chain": row["chain"], "model_type": row["model_type"],
            "record_type": row["record_type"], "sample_index": row["sample_index"],
            "length": len(sequence), "generation_score": float(row["score"]),
            "native_recovery": float(row["native_recovery"]) if row["native_recovery"] else None,
            "sap_simple": surrogate.score(scaffold["burial"], sequence),
            "sap_window": surrogate.score(scaffold["burial"], sequence, scaffold["window"]),
        })
    return rows, skipped


def summarize_pairs(rows):
    scored = [r for r in rows if r["delta_fo_total"] is not None]
    summary = {
        "pair_count": len(rows),
        "contact_descriptor_ties": sum(1 for r in rows if r["contact_descriptor_tie"]),
        "sap_simple_ties": sum(1 for r in rows if r["sap_simple_tie"]),
        "sap_window_ties": sum(1 for r in rows if abs(r["delta_sap_window"]) < TIE_TOLERANCE),
        "delta_sap_simple_sign_test": sign_test([r["delta_sap_simple"] for r in rows]),
        "delta_sap_window_sign_test": sign_test([r["delta_sap_window"] for r in rows]),
    }
    if scored:
        for name in ("sap_simple", "sap_window"):
            values = [r["delta_" + name] for r in scored]
            summary["pooled_" + name + "_vs_first_order"] = {
                "pearson_r": pearson(values, [r["delta_fo_total"] for r in scored]),
                "spearman_rho": spearman(values, [r["delta_fo_total"] for r in scored]),
                "pearson_r_vs_full_gap": pearson(values, [r["delta_full_total"] for r in scored]),
                "pair_count": len(scored),
            }
        within = defaultdict(list)
        for row in scored:
            within[row["structure_id"]].append(row)
        summary["within_structure_simple_vs_first_order"] = {
            key: {"pair_count": len(group),
                  "pearson_r": pearson([r["delta_sap_simple"] for r in group],
                                       [r["delta_fo_total"] for r in group])}
            for key, group in sorted(within.items())}
    return summary


def summarize_candidates(rows):
    generated = [r for r in rows if r["record_type"] != "native"]
    pools, natives = [], {}
    for row in rows:
        if row["record_type"] == "native":
            natives.setdefault(row["structure_id"], []).append(row["sap_simple"])
    grouped = defaultdict(list)
    for row in generated:
        grouped[(row["structure_id"], row["model_type"])].append(row)
    for key in sorted(grouped):
        pool = grouped[key]
        scores = [r["generation_score"] for r in pool]
        simple_rho, simple_p = exact_spearman_p(scores, [r["sap_simple"] for r in pool])
        pools.append({
            "structure_id": key[0], "model_type": key[1], "sample_count": len(pool),
            "spearman_nll_vs_sap_simple": simple_rho,
            "spearman_exact_p": simple_p,
            "spearman_nll_vs_sap_window": spearman(scores, [r["sap_window"] for r in pool]),
            "generation_score_min": min(scores), "generation_score_max": max(scores),
            "sap_simple_mean": statistics.fmean(r["sap_simple"] for r in pool),
            "native_sap_simple": statistics.fmean(natives[key[0]]) if key[0] in natives else None,
        })
    simple = [p["spearman_nll_vs_sap_simple"] for p in pools
              if p["spearman_nll_vs_sap_simple"] is not None]
    sizes = sorted({p["sample_count"] for p in pools})
    return {
        "generated_count": len(generated), "pool_count": len(pools), "pools": pools,
        "pool_sizes": sizes,
        "mean_spearman_simple": statistics.fmean(simple) if simple else None,
        "pool_sign_test": sign_test(simple),
        # Significance is decided per pool by its own exact p-value, so pools of differing
        # size are not judged against one borrowed critical value.
        "pools_individually_significant": sum(
            1 for p in pools if p["spearman_exact_p"] is not None and p["spearman_exact_p"] < 0.05),
        "pools_without_exact_p": sum(1 for p in pools if p["spearman_exact_p"] is None),
        "significance_note":
            "Each pool's p-value is exact, enumerated over that pool's own orderings, so pools "
            "of different sizes are handled correctly. Individual pools of this size rarely "
            "reach significance alone; the agreement of sign across pools carries the result",
    }


def two_regime(pair_rows, decomposition_by_model, candidate_rows):
    """Compare each checkpoint's agreement with SAP in two regimes.

    Rearrangement holds composition exactly equal, so the SAP change reflects only where
    residues sit. Generation lets composition vary freely, so SAP moves mostly with which
    residues were chosen. Spearman is used in both regimes so the two are one statistic.
    """
    generated = [r for r in candidate_rows if r["record_type"] != "native"]
    pools = defaultdict(list)
    for row in generated:
        pools[(row["structure_id"], row["model_type"])].append(row)
    result = {}
    for model_type, reference in sorted(decomposition_by_model.items()):
        paired = [(r, reference[r["pair_id"]]) for r in pair_rows if r["pair_id"] in reference]
        by_structure = defaultdict(list)
        for row, entry in paired:
            by_structure[row["structure_id"]].append((row, entry))
        pool_values = []
        for key in sorted(pools):
            if key[1] != model_type:
                continue
            pool = pools[key]
            value = spearman([r["generation_score"] for r in pool],
                             [r["sap_simple"] for r in pool])
            if value is not None:
                pool_values.append({"structure_id": key[0], "spearman": value,
                                    "sample_count": len(pool)})
        result[model_type] = {
            "rearrangement": {
                "pair_count": len(paired),
                "pooled_spearman": spearman([r["delta_sap_simple"] for r, _ in paired],
                                            [e["delta_fo_total"] for _, e in paired]),
                "pooled_pearson": pearson([r["delta_sap_simple"] for r, _ in paired],
                                          [e["delta_fo_total"] for _, e in paired]),
                "per_structure": [
                    {"structure_id": key, "pair_count": len(group),
                     "spearman": spearman([r["delta_sap_simple"] for r, _ in group],
                                          [e["delta_fo_total"] for _, e in group])}
                    for key, group in sorted(by_structure.items())],
            },
            "generation": {
                "pool_count": len(pool_values), "per_pool": pool_values,
                "mean_spearman": statistics.fmean(p["spearman"] for p in pool_values)
                if pool_values else None,
            },
        }
    return result


def figure_two_regime(regimes, path):
    """Grouped bars of each checkpoint's SAP agreement, with the underlying units overlaid.

    Colour encodes the checkpoint here because the checkpoint comparison is the subject of
    this figure, unlike the decomposition figure where colour encodes the term.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"vanilla": "#2a78d6", "soluble": "#eb6834"}
    models = [m for m in ("vanilla", "soluble") if m in regimes]
    labels = ["Arrangement\n(composition fixed)", "Composition\n(generation)"]
    figure_, axis = plt.subplots(figsize=(7.0, 4.4))
    width = 0.36
    for offset, model_type in enumerate(models):
        block = regimes[model_type]
        centres = [index + (offset - 0.5) * width for index in range(2)]
        raw = [block["rearrangement"]["pooled_spearman"],
               block["generation"]["mean_spearman"]]
        # Either regime can be undefined: too few matched pairs for a correlation, or no
        # candidate pool for this checkpoint. Draw nothing there rather than failing after
        # the tables have already been written.
        heights = [0.0 if h is None else h for h in raw]
        axis.bar(centres, heights, width=width * 0.92, color=colors[model_type],
                 edgecolor="white", linewidth=1.0, zorder=3, label=model_type)
        spread = [[item["spearman"] for item in block["rearrangement"]["per_structure"]
                   if item["spearman"] is not None],
                  [item["spearman"] for item in block["generation"]["per_pool"]]]
        for centre, points in zip(centres, spread):
            axis.scatter([centre] * len(points), points, s=16, facecolor="white",
                         edgecolor="#2b2b2b", linewidth=0.8, zorder=4)
        for centre, height, value in zip(centres, heights, raw):
            axis.text(centre, height + (0.04 if height >= 0 else -0.07),
                      "n/a" if value is None else "{:+.2f}".format(value),
                      ha="center", fontsize=8, color="#595959", zorder=5)
    axis.axhline(0, color="#595959", linewidth=1.0, zorder=2)
    axis.set_xticks(range(2))
    axis.set_xticklabels(labels, fontsize=9, color="#2b2b2b")
    axis.set_ylabel("Spearman correlation with SAP", fontsize=9, color="#2b2b2b")
    axis.set_title("Both checkpoints track SAP when choosing residues, neither when placing them",
                   fontsize=9.5, color="#2b2b2b", loc="left", pad=12)
    axis.set_ylim(-0.75, 1.05)
    axis.yaxis.grid(True, color="#b4b4b4", linewidth=0.6, alpha=0.5, zorder=0)
    axis.set_axisbelow(True)
    axis.legend(frameon=False, fontsize=8.5, loc="upper left", labelcolor="#595959")
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#b4b4b4")
    axis.tick_params(colors="#595959", labelsize=8, length=3)
    # Open dots are the per-structure and per-pool values behind each bar.
    axis.text(0.5, -0.70, "open dots: per-structure (left) and per-pool (right) values",
              fontsize=7.5, color="#595959", ha="center")
    figure_.tight_layout()
    for suffix in (".png", ".pdf"):
        figure_.savefig(path.with_suffix(suffix), dpi=300, bbox_inches="tight")
    plt.close(figure_)


def figure(pair_rows, candidate_rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    scored = [r for r in pair_rows if r["delta_fo_total"] is not None]
    generated = [r for r in candidate_rows if r["record_type"] != "native"]
    figure_, (left, right) = plt.subplots(1, 2, figsize=(11.0, 4.6))
    sap_color, model_color = "#2a78d6", "#eb6834"

    left.scatter([r["delta_fo_total"] for r in scored], [r["delta_sap_simple"] for r in scored],
                 s=28, facecolor=sap_color, edgecolor="white", linewidth=0.7, alpha=0.85, zorder=3)
    left.axhline(0, color="#595959", linewidth=1.0, zorder=2)
    left.set_xlabel("Model first-order term (nats per chain)", fontsize=8.5)
    left.set_ylabel("SAP surrogate change", fontsize=8.5)
    left.set_title("Composition fixed: the two disagree", fontsize=9.5, loc="left", pad=10)

    # The reported correlation is computed within each structure/checkpoint pool, so the
    # panel centres both axes within pool. Plotting raw values instead would show a trend
    # driven largely by the different NLL ranges of different structures.
    markers = {"vanilla": "o", "soluble": "^"}
    pools = defaultdict(list)
    for row in generated:
        pools[(row["structure_id"], row["model_type"])].append(row)
    centred = defaultdict(list)
    for key, pool in pools.items():
        mean_nll = statistics.fmean(r["generation_score"] for r in pool)
        mean_sap = statistics.fmean(r["sap_simple"] for r in pool)
        for row in pool:
            centred[key[1]].append((row["generation_score"] - mean_nll,
                                    row["sap_simple"] - mean_sap))
    for model_type in sorted(centred):
        points = centred[model_type]
        right.scatter([p[0] for p in points], [p[1] for p in points],
                      s=30, marker=markers.get(model_type, "o"), facecolor=model_color,
                      edgecolor="white", linewidth=0.7, alpha=0.85, zorder=3)
    right.axhline(0, color="#595959", linewidth=1.0, zorder=2)
    right.axvline(0, color="#595959", linewidth=1.0, zorder=2)
    right.set_xlabel("Generation NLL, centred within pool", fontsize=8.5)
    right.set_ylabel("SAP surrogate, centred within pool", fontsize=8.5)
    right.set_title("Composition free: NLL partly tracks SAP", fontsize=9.5, loc="left", pad=10)
    right.legend(handles=[Line2D([], [], linestyle="none", marker=markers[m], markersize=6,
                                 markerfacecolor=model_color, markeredgecolor="white", label=m)
                          for m in sorted(markers)],
                 frameon=False, fontsize=8, title="checkpoint", title_fontsize=8)
    for axis in (left, right):
        axis.grid(True, color="#b4b4b4", linewidth=0.6, alpha=0.5, zorder=0)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["left", "bottom"]].set_color("#b4b4b4")
        axis.tick_params(colors="#595959", labelsize=8, length=3)
    figure_.tight_layout()
    for suffix in (".png", ".pdf"):
        figure_.savefig(path.with_suffix(suffix), dpi=300, bbox_inches="tight")
    plt.close(figure_)


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    pairs_path = args.pairs.resolve()
    dataset = json.loads(pairs_path.read_text(encoding="utf-8-sig"))
    if dataset.get("schema_version") != 1:
        raise ValueError("Expected a schema-1 pair dataset")
    scaffolds = build_scaffolds(dataset, pairs_path.parent, args.window_radius,
                                args.sphere_points)
    decomposition, by_model = {}, defaultdict(dict)
    if args.decomposition:
        for row in read_csv(args.decomposition):
            entry = {"delta_fo_total": float(row["delta_fo_total"]),
                     "delta_full_total": float(row["delta_full_total"]),
                     "model_type": row["model_type"]}
            by_model[row["model_type"]][row["pair_id"]] = entry
            if row["model_type"] == args.decomposition_model:
                decomposition[row["pair_id"]] = entry
        if not decomposition:
            raise ValueError("No decomposition rows for checkpoint " + args.decomposition_model)
    pair_rows = score_pairs(dataset, scaffolds, decomposition)
    candidate_rows, skipped_candidates = (
        score_candidates(read_csv(args.candidates), scaffolds) if args.candidates else ([], []))
    summary = {
        "schema_version": 1, "purpose": "sap_surrogate_baseline",
        "burial_definition":
            "Relative accessibility of {N, CA, C, O, virtual CB} by Shrake-Rupley, normalized "
            "by the same residue in isolation; independent of residue identity",
        "hydrophobicity_scale": "Black and Mould (1991) normalized, shifted so glycine is zero",
        "window_radius_angstrom": args.window_radius, "sphere_points": args.sphere_points,
        "pairs": summarize_pairs(pair_rows),
        "candidates": summarize_candidates(candidate_rows) if candidate_rows else None,
        "skipped_candidates": skipped_candidates,
        "two_regime": two_regime(pair_rows, by_model, candidate_rows)
        if by_model and candidate_rows else None,
        "limitations": [
            "Backbone-derived burial is a surrogate for side-chain SAP; the staged inputs carry "
            "no side chains, so a weak correlation may reflect surrogate error as well as a "
            "genuine difference",
            "Generation scores were produced under per-sample decoding orders, so the candidate "
            "ranking carries order noise that the paired experiment controls for",
            "Eight samples per pool and seven structures do not support per-pool inference",
        ],
    }
    write_csv(output / "sap_pairs.csv", PAIR_FIELDS, pair_rows)
    if candidate_rows:
        write_csv(output / "sap_candidates.csv", CANDIDATE_FIELDS, candidate_rows)
    write_json(output / "sap_summary.json", summary)
    if not args.no_figure and decomposition and candidate_rows:
        figure(pair_rows, candidate_rows, output / "sap_surrogate")
        if summary["two_regime"]:
            figure_two_regime(summary["two_regime"], output / "sap_two_regime")
    block = summary["pairs"]
    print("Pairs: contact-count ties {}/{}, SAP ties {}/{}".format(
        block["contact_descriptor_ties"], block["pair_count"],
        block["sap_simple_ties"], block["pair_count"]), flush=True)
    if "pooled_sap_simple_vs_first_order" in block:
        fit = block["pooled_sap_simple_vs_first_order"]
        print("  SAP change vs model first-order term: r = {}, rho = {}".format(
            format_optional(fit["pearson_r"]), format_optional(fit["spearman_rho"])), flush=True)
    if summary["candidates"]:
        candidates = summary["candidates"]
        print("Candidates: mean rho(NLL, SAP) = {} over {} pools; {} individually "
              "significant".format(format_optional(candidates["mean_spearman_simple"]),
                                   candidates["pool_count"],
                                   candidates["pools_individually_significant"]), flush=True)
    print("Complete: " + str(output / "sap_summary.json"), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, help="baseline candidates.csv")
    parser.add_argument("--decomposition", type=Path,
                        help="first_order_scores.csv, to correlate against the model's terms")
    parser.add_argument("--decomposition-model", default="vanilla",
                        choices=("vanilla", "soluble"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window-radius", type=float, default=surrogate.DEFAULT_WINDOW_RADIUS)
    parser.add_argument("--sphere-points", type=int, default=surrogate.DEFAULT_SPHERE_POINTS)
    parser.add_argument("--no-figure", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()

"""Fit a patch-local ranking head on cached features and compare it with the frozen model.

The head scores one arrangement at a time; the prediction is the contrast
`q = score(A) - score(B)`, so `q > 0` predicts A is the more stable arrangement, matching
`d > 0`. The objective is a margin ranking loss, not regression onto the measured difference:
the question is which arrangement wins, and the margin only scales by how large the measured
effect was.

Because the features are cached, a fit costs seconds. That makes cluster cross-validation
affordable, which matters more than it sounds: a single held-out split leaves 31 test domains,
while k-fold over whole HSSP clusters gives an out-of-fold prediction for every one of the 208
and makes the paired per-domain test directly comparable with the committed frozen numbers.

Two controls are built in rather than optional. `--shuffle-labels` permutes the sign of `d`
within each domain, keeping inputs and splits fixed, and must collapse accuracy to chance.
And the A/B antisymmetry of `q` is asserted before any fitting, since a head that failed it
would produce confident, meaningless rankings.
"""

import argparse
import csv
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scoring"))
from score_swaps import spearman  # noqa: E402

TIE = 1e-8


def load_cohort(features, contrasts, frozen=None, resolved_only=True):
    import numpy
    table = numpy.load(features)
    rows = []
    with Path(contrasts).open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if resolved_only and row["resolved"] != "True":
                continue
            rows.append({
                "contrast_id": row["contrast_id"], "domain": row["domain"],
                "family": row["family"], "split": row["split"],
                "row_a": int(row["row_a"]), "row_b": int(row["row_b"]),
                "d": float(row["d"]),
                "restores_wild_type": row["restores_wild_type"] == "True",
            })
    if frozen is not None:
        lookup = {}
        with Path(frozen).open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                lookup[row["contrast_id"]] = float(row["q_frozen"])
        kept = [r for r in rows if r["contrast_id"] in lookup]
        for row in kept:
            row["q_frozen"] = lookup[row["contrast_id"]]
        rows = kept
    rows = [r for r in rows if r["d"] != 0.0]
    if not rows:
        raise ValueError("No eligible contrasts")
    return table, rows


def logprob_columns(width, patch_size=2, alphabet=21):
    """Which feature columns are log-probabilities rather than identity one-hots.

    Layout per patch member is `alphabet` log-probabilities then `alphabet` one-hot bits;
    neighbour blocks that follow are one-hot only. Only the continuous block is standardised:
    rescaling a one-hot by its own standard deviation blows up rare amino acids, whose
    variance is near zero, and gains nothing.
    """
    columns = []
    for member in range(patch_size):
        start = member * 2 * alphabet
        columns.extend(range(start, min(start + alphabet, width)))
    return [c for c in columns if c < width]


def fit_standardizer(table, rows, width):
    """Mean and scale from the training arrangements only, so no fold leaks into another."""
    import numpy
    used = sorted({r["row_a"] for r in rows} | {r["row_b"] for r in rows})
    block = table[numpy.asarray(used)]
    columns = logprob_columns(width)
    mean = numpy.zeros(width, dtype="float32")
    scale = numpy.ones(width, dtype="float32")
    if columns:
        index = numpy.asarray(columns)
        mean[index] = block[:, index].mean(axis=0)
        # A floor keeps a degenerate column from being amplified into noise.
        scale[index] = numpy.maximum(block[:, index].std(axis=0), 1e-3)
    return mean, scale


def apply_standardizer(table, mean, scale):
    return ((table - mean) / scale).astype("float32")


def make_head(width, hidden, torch):
    from torch import nn
    return nn.Sequential(nn.Linear(width, hidden), nn.ReLU(),
                         nn.Linear(hidden, hidden), nn.ReLU(),
                         nn.Linear(hidden, 1, bias=False))


def contrast_scores(head, table, rows, torch, device, batch=4096):
    """q = score(A) - score(B) for each row, in order."""
    import numpy
    out = []
    a = numpy.asarray([r["row_a"] for r in rows])
    b = numpy.asarray([r["row_b"] for r in rows])
    head.eval()
    with torch.no_grad():
        for start in range(0, len(rows), batch):
            xa = torch.from_numpy(table[a[start:start + batch]]).to(device)
            xb = torch.from_numpy(table[b[start:start + batch]]).to(device)
            out.extend((head(xa) - head(xb)).squeeze(-1).cpu().tolist())
    return out


def margin(d):
    return min(max(abs(d), 0.1), 2.0)


def family_macro_accuracy(rows, q):
    """Score each contrast, average within domain, then family, then across families."""
    by_family = defaultdict(lambda: defaultdict(list))
    for row, value in zip(rows, q):
        point = 0.5 if abs(value) <= TIE else (1.0 if value * row["d"] > 0 else 0.0)
        by_family[row["family"]][row["domain"]].append(point)
    families = []
    for family, domains in by_family.items():
        families.append(statistics.fmean(statistics.fmean(v) for v in domains.values()))
    return statistics.fmean(families), len(families)


def flat_accuracy(rows, q):
    points = [0.5 if abs(v) <= TIE else (1.0 if v * r["d"] > 0 else 0.0)
              for r, v in zip(rows, q)]
    return statistics.fmean(points)


def per_domain_spearman(rows, q, minimum=20):
    """Tie-aware Spearman of q against d within each domain with enough contrasts."""
    grouped = defaultdict(lambda: ([], []))
    for row, value in zip(rows, q):
        grouped[row["domain"]][0].append(row["d"])
        grouped[row["domain"]][1].append(value)
    out, excluded = [], []
    for domain, (d, value) in sorted(grouped.items()):
        if len(d) < minimum or len(set(d)) < 2 or len(set(value)) < 2:
            excluded.append({"domain": domain, "contrasts": len(d),
                             "reason": "below minimum" if len(d) < minimum else "constant"})
            continue
        rho = spearman(d, value)
        if rho is None:
            excluded.append({"domain": domain, "contrasts": len(d), "reason": "undefined"})
            continue
        out.append({"domain": domain, "contrasts": len(d), "spearman": rho})
    return out, excluded


def paired_sign_test(a, b):
    """Two-sided exact binomial sign test over paired per-domain values."""
    wins = sum(1 for x, y in zip(a, b) if x > y)
    losses = sum(1 for x, y in zip(a, b) if x < y)
    n = wins + losses
    if n == 0:
        return {"n": 0, "wins": 0, "losses": 0, "ties": len(a), "p_value": 1.0}
    tail = sum(math.comb(n, k) for k in range(0, min(wins, losses) + 1))
    p = min(1.0, 2.0 * tail / (2 ** n))
    return {"n": n, "wins": wins, "losses": losses, "ties": len(a) - n, "p_value": p}


def bootstrap_family_difference(rows, q_learned, q_frozen, draws, seed):
    """Resample whole families; report the percentile interval on the paired difference."""
    by_family = defaultdict(lambda: ([], []))
    for row, learned, frozen in zip(rows, q_learned, q_frozen):
        point = lambda v: 0.5 if abs(v) <= TIE else (1.0 if v * row["d"] > 0 else 0.0)
        by_family[row["family"]][0].append(point(learned))
        by_family[row["family"]][1].append(point(frozen))
    families = sorted(by_family)
    diffs = {f: statistics.fmean(by_family[f][0]) - statistics.fmean(by_family[f][1])
             for f in families}
    rng = random.Random(seed)
    samples = []
    for _ in range(draws):
        picked = [diffs[rng.choice(families)] for _ in families]
        samples.append(statistics.fmean(picked))
    samples.sort()
    lo = samples[int(0.025 * (len(samples) - 1))]
    hi = samples[int(0.975 * (len(samples) - 1))]
    observed = statistics.fmean(diffs.values())
    return {"families": len(families), "observed_mean_difference": observed,
            "ci95": [lo, hi], "draws": draws,
            "family_wins": sum(1 for v in diffs.values() if v > 0),
            "family_ties": sum(1 for v in diffs.values() if v == 0),
            "family_losses": sum(1 for v in diffs.values() if v < 0)}


def assert_antisymmetry(head, table, rows, torch, device):
    """Reversing A and B must negate q, and identical arrangements must tie exactly."""
    sample = rows[:256]
    forward = contrast_scores(head, table, sample, torch, device)
    reversed_rows = [{**r, "row_a": r["row_b"], "row_b": r["row_a"]} for r in sample]
    backward = contrast_scores(head, table, reversed_rows, torch, device)
    worst = max(abs(f + b) for f, b in zip(forward, backward))
    if worst > 1e-4:
        raise RuntimeError("A/B reversal does not negate q (worst %.3g)" % worst)
    same = [{**r, "row_b": r["row_a"]} for r in sample]
    ties = contrast_scores(head, table, same, torch, device)
    if max(abs(v) for v in ties) > 1e-6:
        raise RuntimeError("Identical arrangements do not tie")
    return {"max_antisymmetry_error": worst}


def fit(table, train_rows, val_rows, args, seed, torch, device):
    import numpy
    head = make_head(table.shape[1], args.hidden, torch).to(device)
    torch.manual_seed(seed)
    for layer in head:
        if hasattr(layer, "weight"):
            torch.nn.init.xavier_uniform_(layer.weight)
            if getattr(layer, "bias", None) is not None:
                torch.nn.init.zeros_(layer.bias)
    assert_antisymmetry(head, table, train_rows, torch, device)

    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    by_domain = defaultdict(list)
    for index, row in enumerate(train_rows):
        by_domain[row["domain"]].append(index)
    domains = sorted(by_domain)
    rng = random.Random(seed)

    a_all = numpy.asarray([r["row_a"] for r in train_rows])
    b_all = numpy.asarray([r["row_b"] for r in train_rows])
    d_all = numpy.asarray([r["d"] for r in train_rows], dtype="float32")
    m_all = numpy.asarray([margin(r["d"]) for r in train_rows], dtype="float32")

    best = {"accuracy": -1.0, "loss": float("inf"), "state": None, "epoch": 0}
    history, stale = [], 0
    steps = max(1, len(train_rows) // args.batch_size)
    for epoch in range(1, args.epochs + 1):
        head.train()
        losses = []
        for _ in range(steps):
            picked = [rng.choice(by_domain[rng.choice(domains)])
                      for _ in range(args.batch_size)]
            index = numpy.asarray(picked)
            xa = torch.from_numpy(table[a_all[index]]).to(device)
            xb = torch.from_numpy(table[b_all[index]]).to(device)
            d = torch.from_numpy(d_all[index]).to(device)
            m = torch.from_numpy(m_all[index]).to(device)
            q = (head(xa) - head(xb)).squeeze(-1)
            loss = torch.clamp(m - torch.sign(d) * q, min=0).mean()
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        q_val = contrast_scores(head, table, val_rows, torch, device)
        accuracy, families = family_macro_accuracy(val_rows, q_val)
        val_loss = statistics.fmean(
            max(0.0, margin(r["d"]) - (1 if r["d"] > 0 else -1) * v)
            for r, v in zip(val_rows, q_val))
        history.append({"epoch": epoch, "train_loss": statistics.fmean(losses),
                        "val_family_macro_accuracy": accuracy, "val_ranking_loss": val_loss})
        better = (accuracy > best["accuracy"] + 1e-12
                  or (abs(accuracy - best["accuracy"]) <= 1e-12 and val_loss < best["loss"]))
        if better:
            best = {"accuracy": accuracy, "loss": val_loss, "epoch": epoch,
                    "state": {k: v.detach().clone() for k, v in head.state_dict().items()}}
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    head.load_state_dict(best["state"])
    return head, history, best


def partition(rows, folds, fold_index, seed):
    """Whole-cluster folds, so no family is split across train, validation and test."""
    families = sorted(set(r["family"] for r in rows))
    rng = random.Random(seed)
    shuffled = list(families)
    rng.shuffle(shuffled)
    buckets = [shuffled[i::folds] for i in range(folds)]
    test = set(buckets[fold_index])
    val = set(buckets[(fold_index + 1) % folds])
    return ([r for r in rows if r["family"] not in test and r["family"] not in val],
            [r for r in rows if r["family"] in val],
            [r for r in rows if r["family"] in test])


def run(args):
    import numpy
    import torch

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    device = "cuda" if (args.device != "cpu" and torch.cuda.is_available()) else "cpu"
    table, rows = load_cohort(args.features, args.contrasts, args.frozen,
                              resolved_only=not args.all_contrasts)

    if args.shuffle_labels:
        rng = random.Random(args.shuffle_seed)
        by_domain = defaultdict(list)
        for row in rows:
            by_domain[row["domain"]].append(row)
        for group in by_domain.values():
            signs = [1 if r["d"] > 0 else -1 for r in group]
            rng.shuffle(signs)
            for row, sign in zip(group, signs):
                row["d"] = abs(row["d"]) * sign

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    predictions, fold_records = [], []
    folds = args.folds if args.folds > 1 else 1

    for fold_index in range(folds):
        if args.folds > 1:
            train_rows, val_rows, test_rows = partition(rows, folds, fold_index, args.fold_seed)
        else:
            train_rows = [r for r in rows if r["split"] == "train"]
            val_rows = [r for r in rows if r["split"] == "val"]
            test_rows = [r for r in rows if r["split"] == "test"]
        if not (train_rows and val_rows and test_rows):
            raise ValueError("Empty partition in fold %d" % fold_index)

        # Standardisation is refitted inside the fold, on training arrangements only. Fitting
        # it once on the whole cohort would leak held-out feature statistics into training.
        if args.no_standardize:
            scaled = table
        else:
            mean, scale = fit_standardizer(table, train_rows, table.shape[1])
            scaled = apply_standardizer(table, mean, scale)

        for seed in seeds:
            head, history, best = fit(scaled, train_rows, val_rows, args, seed, torch, device)
            q_test = contrast_scores(head, scaled, test_rows, torch, device)
            for row, value in zip(test_rows, q_test):
                predictions.append({"contrast_id": row["contrast_id"], "domain": row["domain"],
                                    "family": row["family"], "fold": fold_index, "seed": seed,
                                    "d": row["d"], "q_learned": value,
                                    "q_frozen": row.get("q_frozen"),
                                    "restores_wild_type": row["restores_wild_type"]})
            accuracy, families = family_macro_accuracy(test_rows, q_test)
            fold_records.append({
                "fold": fold_index, "seed": seed,
                "train_contrasts": len(train_rows), "val_contrasts": len(val_rows),
                "test_contrasts": len(test_rows), "test_families": families,
                "selected_epoch": best["epoch"],
                "val_family_macro_accuracy": best["accuracy"],
                "test_family_macro_accuracy": accuracy,
                "epochs_run": len(history),
            })
            print(json.dumps(fold_records[-1]), flush=True)

    with (output / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)

    summary = summarize(predictions, args)
    summary["folds"] = fold_records
    summary["config"] = {k: (str(v) if isinstance(v, Path) else v)
                         for k, v in vars(args).items()}
    summary["device"] = device
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n",
                                         encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ("folds", "config", "per_domain")}, indent=2), flush=True)


def summarize(predictions, args):
    """Average seeds per contrast, then report every endpoint on the pooled out-of-fold set."""
    merged = defaultdict(list)
    for row in predictions:
        merged[row["contrast_id"]].append(row)
    rows, learned, frozen = [], [], []
    for group in merged.values():
        first = group[0]
        rows.append(first)
        learned.append(statistics.fmean(r["q_learned"] for r in group))
        frozen.append(first["q_frozen"])

    have_frozen = all(v is not None for v in frozen)
    out = {
        "contrasts": len(rows),
        "domains": len(set(r["domain"] for r in rows)),
        "families": len(set(r["family"] for r in rows)),
        "shuffled_label_control": bool(args.shuffle_labels),
        "learned": endpoint_block(rows, learned),
    }
    if have_frozen:
        out["frozen"] = endpoint_block(rows, frozen)
        ld, _ = per_domain_spearman(rows, learned)
        fd, _ = per_domain_spearman(rows, frozen)
        shared = sorted(set(r["domain"] for r in ld) & set(r["domain"] for r in fd))
        lmap = {r["domain"]: r["spearman"] for r in ld}
        fmap = {r["domain"]: r["spearman"] for r in fd}
        out["paired_per_domain_spearman"] = {
            "domains": len(shared),
            "learned_median": statistics.median([lmap[d] for d in shared]) if shared else None,
            "frozen_median": statistics.median([fmap[d] for d in shared]) if shared else None,
            "sign_test": paired_sign_test([lmap[d] for d in shared], [fmap[d] for d in shared]),
        }
        out["family_bootstrap"] = bootstrap_family_difference(
            rows, learned, frozen, args.bootstrap, args.bootstrap_seed)
        for label, keep in (("genuine_double_mutants", False), ("restores_wild_type", True)):
            subset = [(r, l, f) for r, l, f in zip(rows, learned, frozen)
                      if r["restores_wild_type"] == keep]
            if not subset:
                continue
            sub_rows = [s[0] for s in subset]
            out[label] = {
                "contrasts": len(sub_rows),
                "learned_flat_accuracy": flat_accuracy(sub_rows, [s[1] for s in subset]),
                "frozen_flat_accuracy": flat_accuracy(sub_rows, [s[2] for s in subset]),
            }
    return out


def endpoint_block(rows, q):
    accuracy, families = family_macro_accuracy(rows, q)
    domains, excluded = per_domain_spearman(rows, q)
    values = [r["spearman"] for r in domains]
    return {
        "family_macro_accuracy": accuracy, "families": families,
        "flat_accuracy": flat_accuracy(rows, q),
        "per_domain_spearman": {
            "domains": len(domains),
            "median": statistics.median(values) if values else None,
            "p25": statistics.quantiles(values, n=4)[0] if len(values) > 3 else None,
            "p75": statistics.quantiles(values, n=4)[2] if len(values) > 3 else None,
            "excluded": len(excluded),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--contrasts", required=True, type=Path)
    parser.add_argument("--frozen", type=Path, help="frozen_contrasts.csv for the comparison")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--folds", type=int, default=0,
                        help="k-fold over whole families; 0 uses the committed split")
    parser.add_argument("--fold-seed", type=int, default=7)
    parser.add_argument("--seeds", default="7,17,27")
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=7)
    parser.add_argument("--shuffle-labels", action="store_true",
                        help="Chance control: permute the sign of d within each domain")
    parser.add_argument("--shuffle-seed", type=int, default=99)
    parser.add_argument("--no-standardize", action="store_true",
                        help="Ablation: feed raw log-probabilities without rescaling")
    parser.add_argument("--all-contrasts", action="store_true",
                        help="Include contrasts below twice their measurement error")
    parser.add_argument("--device", default="cuda")
    run(parser.parse_args())


if __name__ == "__main__":
    main()

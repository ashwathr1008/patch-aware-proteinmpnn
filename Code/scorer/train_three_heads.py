"""Fit the spec's three heads on one shared frozen decoder-state cache and compare them.

All three read the same `[arrangements, L, 149]` tensor and the same folds, seeds and sampled
pairs, so the differences between them are the pooling region and the attention block, not the
data. The frozen full-chain likelihood is carried alongside as the external reference.

The prespecified contrasts, in the spec's order:

    learned vs frozen        does fitting a stability readout help at all
    local_mean vs whole_chain_mean   does restricting the readout to the patch help
    local_attention vs local_mean    does letting local residues interact again help

The second is the one this experiment exists for, and it is the best powered: both arms share
identical features, identical folds and the same parameter count, differing only in a mask.
"""

import argparse
import csv
import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import three_heads as th  # noqa: E402
from train_patch_head import (  # noqa: E402
    TIE, bootstrap_family_difference, endpoint_block, family_macro_accuracy, flat_accuracy,
    margin, paired_sign_test, partition, per_domain_spearman)

STATE_CHANNELS = 128   # decoder hidden width
IDENTITY_CHANNELS = 21  # one-hot of the placed residue, immediately after the state
NESTING_CHANNEL = STATE_CHANNELS + IDENTITY_CHANNELS + 21  # log p(placed), if present


def continuous_channels(width):
    """Channels to standardise: the decoder state and every log-probability channel.

    The 21-dim identity one-hot in between is left alone -- dividing a one-hot by its own
    standard deviation amplifies rare amino acids whose variance is near zero and buys
    nothing. Everything after it (the log-probability vector, log p(placed), entropy and the
    margin) is continuous and is scaled.
    """
    return (list(range(STATE_CHANNELS))
            + list(range(STATE_CHANNELS + IDENTITY_CHANNELS, width)))


def load_cohort(states_path, lengths_path, contrasts_path, frozen_path, resolved_only=True):
    import numpy
    states = numpy.load(states_path, mmap_mode="r")
    lengths = numpy.load(lengths_path)
    rows = []
    with Path(contrasts_path).open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if resolved_only and row["resolved"] != "True":
                continue
            if float(row["d"]) == 0.0:
                continue
            rows.append({
                "contrast_id": row["contrast_id"], "domain": row["domain"],
                "family": row["family"], "split": row["split"],
                "row_a": int(row["row_a"]), "row_b": int(row["row_b"]),
                "d": float(row["d"]), "length": int(row["length"]),
                "region": [int(p) for p in row["region"].split()],
                "restores_wild_type": row["restores_wild_type"] == "True",
            })
    if frozen_path is not None:
        lookup = {}
        with Path(frozen_path).open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                lookup[row["contrast_id"]] = float(row["q_frozen"])
        rows = [r for r in rows if r["contrast_id"] in lookup]
        for row in rows:
            row["q_frozen"] = lookup[row["contrast_id"]]
    if not rows:
        raise ValueError("No eligible contrasts")
    return states, lengths, rows


def fit_channel_standardizer(states, lengths, rows, max_length):
    """Per-channel mean and scale over valid positions of training arrangements only.

    Only the decoder-state channels are rescaled. The trailing one-hot block is left alone:
    dividing a one-hot by its own standard deviation amplifies rare amino acids whose variance
    is near zero and buys nothing.
    """
    import numpy
    width = states.shape[2]
    channels = numpy.asarray(continuous_channels(width))
    used = sorted({r["row_a"] for r in rows} | {r["row_b"] for r in rows})
    sample = used[:: max(1, len(used) // 20000)]  # 20k arrangements is ample for the moments
    total = numpy.zeros(len(channels), dtype=numpy.float64)
    square = numpy.zeros(len(channels), dtype=numpy.float64)
    count = 0
    for row in sample:
        length = int(lengths[row])
        block = numpy.asarray(states[row, :length, :], dtype=numpy.float64)[:, channels]
        total += block.sum(axis=0)
        square += (block ** 2).sum(axis=0)
        count += length
    mean = total / count
    variance = numpy.maximum(square / count - mean ** 2, 0.0)
    scale = numpy.maximum(numpy.sqrt(variance), 1e-3)
    full_mean = numpy.zeros(width, dtype=numpy.float32)
    full_scale = numpy.ones(width, dtype=numpy.float32)
    full_mean[channels] = mean
    full_scale[channels] = scale
    return full_mean, full_scale


def gather(states, lengths, rows, indices, side, mean, scale, max_length, head_name, torch,
           device):
    """Build (states, mask) for one side of a batch of contrasts."""
    import numpy
    picked = [rows[i] for i in indices]
    raw = numpy.stack([numpy.asarray(states[r[side]], dtype=numpy.float32) for r in picked])
    raw = (raw - mean) / scale
    mask = th.region_mask([r["region"] for r in picked], [r["length"] for r in picked],
                          max_length, head_name)
    return (torch.from_numpy(raw).to(device), mask.to(device))


def contrast_scores(head, states, lengths, rows, mean, scale, max_length, head_name,
                    torch, device, batch=256):
    out = []
    head.eval()
    with torch.no_grad():
        for start in range(0, len(rows), batch):
            index = list(range(start, min(start + batch, len(rows))))
            xa, ma = gather(states, lengths, rows, index, "row_a", mean, scale,
                            max_length, head_name, torch, device)
            xb, mb = gather(states, lengths, rows, index, "row_b", mean, scale,
                            max_length, head_name, torch, device)
            out.extend((head(xa, ma) - head(xb, mb)).cpu().tolist())
    return out


def assert_properties(head, states, lengths, rows, mean, scale, max_length, head_name,
                      torch, device):
    """A/B reversal must negate q and an arrangement against itself must tie."""
    sample = rows[:128]
    index = list(range(len(sample)))
    xa, ma = gather(states, lengths, sample, index, "row_a", mean, scale, max_length,
                    head_name, torch, device)
    xb, mb = gather(states, lengths, sample, index, "row_b", mean, scale, max_length,
                    head_name, torch, device)
    head.eval()
    with torch.no_grad():
        forward = (head(xa, ma) - head(xb, mb))
        backward = (head(xb, mb) - head(xa, ma))
        same = (head(xa, ma) - head(xa, ma))
    worst = float((forward + backward).abs().max())
    if worst > 1e-3:
        raise RuntimeError("%s: A/B reversal does not negate q (%.3g)" % (head_name, worst))
    if float(same.abs().max()) > 1e-5:
        raise RuntimeError("%s: identical arrangements do not tie" % head_name)
    return worst


def fit(states, lengths, train_rows, val_rows, head_name, args, seed, mean, scale, torch,
        device):
    head = th.build(head_name, states.shape[2], args.hidden).to(device)
    torch.manual_seed(seed)
    assert_properties(head, states, lengths, train_rows, mean, scale, args.max_length,
                      head_name, torch, device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    by_domain = defaultdict(list)
    for index, row in enumerate(train_rows):
        by_domain[row["domain"]].append(index)
    domains = sorted(by_domain)
    rng = random.Random(seed)

    best = {"accuracy": -1.0, "loss": float("inf"), "state": None, "epoch": 0}
    history, stale = [], 0
    steps = max(1, min(args.max_steps, len(train_rows) // args.batch_size))
    for epoch in range(1, args.epochs + 1):
        head.train()
        losses = []
        for _ in range(steps):
            index = [rng.choice(by_domain[rng.choice(domains)])
                     for _ in range(args.batch_size)]
            xa, ma = gather(states, lengths, train_rows, index, "row_a", mean, scale,
                            args.max_length, head_name, torch, device)
            xb, mb = gather(states, lengths, train_rows, index, "row_b", mean, scale,
                            args.max_length, head_name, torch, device)
            d = torch.tensor([train_rows[i]["d"] for i in index], device=device)
            m = torch.tensor([margin(train_rows[i]["d"]) for i in index], device=device)
            q = head(xa, ma) - head(xb, mb)
            loss = torch.clamp(m - torch.sign(d) * q, min=0).mean()
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        q_val = contrast_scores(head, states, lengths, val_rows, mean, scale,
                                args.max_length, head_name, torch, device)
        accuracy, _ = family_macro_accuracy(val_rows, q_val)
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


def run(args):
    import torch

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    device = "cuda" if (args.device != "cpu" and torch.cuda.is_available()) else "cpu"
    states, lengths, rows = load_cohort(args.states, args.lengths, args.contrasts,
                                        args.frozen)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    heads = [h for h in args.heads.split(",") if h.strip()]
    for name in heads:
        if name not in th.HEADS:
            raise ValueError("Unknown head: " + name)

    folds = args.folds if args.folds > 1 else 1
    predictions, records = [], []
    for fold_index in range(folds):
        if args.folds > 1:
            train_rows, val_rows, test_rows = partition(rows, folds, fold_index,
                                                        args.fold_seed)
        else:
            train_rows = [r for r in rows if r["split"] == "train"]
            val_rows = [r for r in rows if r["split"] == "val"]
            test_rows = [r for r in rows if r["split"] == "test"]
        if not (train_rows and val_rows and test_rows):
            raise ValueError("Empty partition in fold %d" % fold_index)
        mean, scale = fit_channel_standardizer(states, lengths, train_rows, args.max_length)

        for name in heads:
            for seed in seeds:
                head, history, best = fit(states, lengths, train_rows, val_rows, name,
                                          args, seed, mean, scale, torch, device)
                q_test = contrast_scores(head, states, lengths, test_rows, mean, scale,
                                         args.max_length, name, torch, device)
                for row, value in zip(test_rows, q_test):
                    predictions.append({
                        "contrast_id": row["contrast_id"], "domain": row["domain"],
                        "family": row["family"], "head": name, "fold": fold_index,
                        "seed": seed, "d": row["d"], "q_learned": value,
                        "q_frozen": row.get("q_frozen"),
                        "restores_wild_type": row["restores_wild_type"]})
                accuracy, families = family_macro_accuracy(test_rows, q_test)
                records.append({"head": name, "fold": fold_index, "seed": seed,
                                "parameters": sum(p.numel() for p in head.parameters()),
                                "train": len(train_rows), "test": len(test_rows),
                                "test_families": families,
                                "selected_epoch": best["epoch"],
                                "val_family_macro_accuracy": best["accuracy"],
                                "test_family_macro_accuracy": accuracy,
                                "epochs_run": len(history)})
                print(json.dumps(records[-1]), flush=True)

    with (output / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    summary = summarize(predictions, heads, args)
    summary["runs"] = records
    summary["device"] = device
    summary["config"] = {k: (str(v) if isinstance(v, Path) else v)
                         for k, v in vars(args).items()}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n",
                                         encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ("runs", "config")}, indent=2), flush=True)


def pooled(predictions, head_name):
    """Average seeds per contrast for one head; returns aligned rows, learned q, frozen q."""
    merged = defaultdict(list)
    for row in predictions:
        if row["head"] == head_name:
            merged[row["contrast_id"]].append(row)
    rows, learned, frozen = [], [], []
    for group in merged.values():
        rows.append(group[0])
        learned.append(statistics.fmean(r["q_learned"] for r in group))
        frozen.append(group[0]["q_frozen"])
    return rows, learned, frozen


def compare(label, rows_a, qa, rows_b, qb, draws, seed):
    """Paired per-domain Spearman sign test plus the family bootstrap, A minus B."""
    da, _ = per_domain_spearman(rows_a, qa)
    db, _ = per_domain_spearman(rows_b, qb)
    amap = {r["domain"]: r["spearman"] for r in da}
    bmap = {r["domain"]: r["spearman"] for r in db}
    shared = sorted(set(amap) & set(bmap))
    out = {"contrast": label, "domains": len(shared),
           "a_median": statistics.median([amap[d] for d in shared]) if shared else None,
           "b_median": statistics.median([bmap[d] for d in shared]) if shared else None,
           "sign_test": paired_sign_test([amap[d] for d in shared], [bmap[d] for d in shared])}
    if [r["contrast_id"] for r in rows_a] == [r["contrast_id"] for r in rows_b]:
        out["family_bootstrap"] = bootstrap_family_difference(rows_a, qa, qb, draws, seed)
    return out


def summarize(predictions, heads, args):
    out = {"heads": {}, "contrasts": None, "comparisons": []}
    cache = {}
    for name in heads:
        rows, learned, frozen = pooled(predictions, name)
        cache[name] = (rows, learned, frozen)
        out["heads"][name] = endpoint_block(rows, learned)
        out["contrasts"] = len(rows)
        out["domains"] = len(set(r["domain"] for r in rows))
        out["families"] = len(set(r["family"] for r in rows))
    first = heads[0]
    rows, _, frozen = cache[first]
    if all(v is not None for v in frozen):
        out["frozen"] = endpoint_block(rows, frozen)
        for name in heads:
            r, learned, f = cache[name]
            out["comparisons"].append(
                compare("%s vs frozen" % name, r, learned, r, f,
                        args.bootstrap, args.bootstrap_seed))
    pairs = [("local_mean", "whole_chain_mean"), ("local_attention", "local_mean")]
    for a, b in pairs:
        if a in cache and b in cache:
            ra, qa, _ = cache[a]
            rb, qb, _ = cache[b]
            order = {r["contrast_id"]: i for i, r in enumerate(rb)}
            if set(order) == set(r["contrast_id"] for r in ra):
                aligned_b = [qb[order[r["contrast_id"]]] for r in ra]
                out["comparisons"].append(
                    compare("%s vs %s" % (a, b), ra, qa, ra, aligned_b,
                            args.bootstrap, args.bootstrap_seed))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--states", required=True, type=Path)
    parser.add_argument("--lengths", required=True, type=Path)
    parser.add_argument("--contrasts", required=True, type=Path)
    parser.add_argument("--frozen", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--heads", default=",".join(th.HEADS))
    parser.add_argument("--folds", type=int, default=0)
    parser.add_argument("--fold-seed", type=int, default=7)
    parser.add_argument("--seeds", default="7,17,27")
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-length", type=int, default=72)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=7)
    parser.add_argument("--device", default="cuda")
    run(parser.parse_args())


if __name__ == "__main__":
    main()

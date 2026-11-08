"""Score the measured swap contrasts with ThermoMPNN-D, on our site pairs only.

ThermoMPNN-D is the published baseline closest to this project: frozen ProteinMPNN features
with a light trained head, a double-mutant mode, and the same mega-scale stability data behind
it. Comparing against it matters more than comparing against a model we wrote ourselves.

Its own entry point cannot do this job affordably. `v2_ssm.py` takes no mutation list: it
enumerates every residue pair within a Ca-Ca radius and scores all 361 non-wild-type identity
combinations of each. Covering our 578 measured pairs needs `--distance 12`, which across 208
domains enumerates on the order of 90 million predictions to obtain the ~209 thousand we
actually want. That is a GPU job with a queue wait; this is minutes on a laptop.

So this substitutes the enumeration and nothing else. `get_ssm_mutations_double` is replaced by
our own pair list; the featurization, the single ProteinMPNN forward pass per structure, and
`run_double` are called unmodified from the upstream checkout. The arithmetic is theirs.

**Only genuine double mutants are scored, and that is forced rather than chosen.** Their
pipeline filters out any row where a position carries its wild-type residue, because the
Siamese head is trained on pairs of substitutions. An arrangement that returns one site to
wild type is a single mutant, which belongs to a different checkpoint; mixing two models inside
one comparison would be worse than reporting a clean subset. 81,769 of 100,572 contrasts (81.3%)
have both arrangements genuinely double, spanning 577 of 578 site pairs. Contrasts that fail
this test are written to the run record with a reason, not silently dropped.

Sign convention. ThermoMPNN reports ddG where more negative means more stabilizing, while the
measurement reports stability where higher means more stable. Agreement is therefore a negative
correlation, the same convention as the frozen-ProteinMPNN comparison, so the per-domain median
is directly comparable to that baseline's -0.497.
"""

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import time
from collections import defaultdict
from pathlib import Path

ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
RESIDUES = set("ACDEFGHIKLMNPQRSTVWY")

FIELDS = ("domain site_i site_j aa_i aa_j measured_difference measured_uncertainty resolved "
          "ddg_forward ddg_reverse model_difference").split()


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def residue_index(pdb):
    """PDB residue number -> sequence index, mirroring upstream `idx_to_pdb_num` exactly.

    Upstream walks `resn_list_<chain>` in order and assigns consecutive indices across chains.
    Inverting that is the only safe way to turn our PDB-numbered sites into the indices the
    model uses; assuming position == number - 1 would be wrong for any non-contiguous chain.
    """
    keys = [k for k in pdb if k.startswith("resn_list_")]
    if len(keys) != 1:
        raise ValueError("Expected a single chain, found %d: %s" % (len(keys), sorted(keys)))
    mapping, offset = {}, 0
    for key in keys:
        for index, resid in enumerate(pdb[key]):
            number = str(resid).strip()
            if number in mapping:
                raise ValueError("Duplicate residue number %s" % number)
            mapping[number] = index + offset
        offset += len(pdb[key])
    return mapping


def read_contrasts(path):
    grouped = defaultdict(list)
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            grouped[row["domain"]].append(row)
    return grouped


def plan_domain(rows, index, sequence):
    """Decide which contrasts can be scored, and the exact prediction rows they need.

    Returns (requests, usable, rejected). A request is (pos_i, pos_j, mut_i, mut_j) with
    pos_i < pos_j, as upstream requires. Positions are ordered once here so that a contrast's
    forward and reverse arrangements map onto the same position pair.
    """
    requests, usable, rejected = {}, [], []
    for row in rows:
        site_i, site_j = str(row["site_i"]), str(row["site_j"])
        if site_i not in index or site_j not in index:
            rejected.append((row, "site absent from structure"))
            continue
        pos_i, pos_j = index[site_i], index[site_j]
        aa_i, aa_j = row["aa_i"], row["aa_j"]
        if not ({aa_i, aa_j} <= RESIDUES):
            rejected.append((row, "non-standard residue"))
            continue
        # Upstream keeps only pos_i < pos_j. Swapping positions means swapping the identities
        # that sit on them, so the arrangement being described is preserved.
        flip = pos_i > pos_j
        if flip:
            pos_i, pos_j = pos_j, pos_i
        wt_i, wt_j = sequence[pos_i], sequence[pos_j]

        # forward = aa_i on site_i, aa_j on site_j; reverse is the other assignment.
        first, second = (aa_j, aa_i) if flip else (aa_i, aa_j)
        forward = (pos_i, pos_j, first, second)
        reverse = (pos_i, pos_j, second, first)
        double = all(m_i != wt_i and m_j != wt_j
                     for _, _, m_i, m_j in (forward, reverse))
        if not double:
            rejected.append((row, "an arrangement is not a double mutant"))
            continue
        for request in (forward, reverse):
            requests[request] = (ALPHABET.index(wt_i), ALPHABET.index(wt_j))
        usable.append((row, forward, reverse))
    return requests, usable, rejected


def score_domain(ssm, torch, DataLoader, tied_featurize_mut, Mutation,
                 pdb, cfg, model, requests, batch_size):
    """One ProteinMPNN forward pass for the structure, then upstream `run_double` on our rows."""
    ordered = list(requests)
    pdb["mutation"] = Mutation([0], ["A"], ["A"], [0.0], "")
    batch = tied_featurize_mut([pdb])
    X, S, mask, lengths, chain_M, chain_encoding_all, residue_idx = batch[:7]

    device = ssm._DEVICE
    X = torch.nan_to_num(X.to(device), nan=0.0)
    S = S.to(device)
    mask = mask.to(device)
    chain_M = chain_M.to(device)
    chain_encoding_all = chain_encoding_all.to(device)
    residue_idx = residue_idx.to(device)

    with torch.no_grad():
        all_hid, embed, _, edges = model.prot_mpnn(
            X, S, mask, chain_M, residue_idx, chain_encoding_all)

        pos = torch.tensor([[r[0], r[1]] for r in ordered], dtype=torch.long)
        wt = torch.tensor([list(requests[r]) for r in ordered], dtype=torch.long)
        mut = torch.tensor([[ALPHABET.index(r[2]), ALPHABET.index(r[3])] for r in ordered],
                           dtype=torch.long)
        loader = DataLoader(ssm.SSMDataset(pos, wt, mut), shuffle=False,
                            batch_size=batch_size, num_workers=0)
        preds = ssm.run_double(all_hid, embed, cfg, loader, batch_size, model, X, mask, edges)

    values = [float(v) for v in preds.reshape(-1)]
    if len(values) != len(ordered):
        raise ValueError("run_double returned %d predictions for %d rows"
                         % (len(values), len(ordered)))
    return dict(zip(ordered, values))


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    # The device patch reads this at import, so it must be set before v2_ssm is imported.
    os.environ["THERMOMPNN_DEVICE"] = args.device
    os.environ.setdefault("WANDB_MODE", "disabled")
    repo = args.repo.resolve()
    if not (repo / "v2_ssm.py").exists():
        raise SystemExit("No v2_ssm.py under " + str(repo))
    sys.path.insert(0, str(repo))

    import torch
    from torch.utils.data import DataLoader
    import v2_ssm as ssm
    from thermompnn.datasets.dataset_utils import Mutation
    from thermompnn.datasets.v2_datasets import tied_featurize_mut

    checkpoint = repo / "model_weights" / "ThermoMPNN-D-ens1.ckpt"
    cfg = ssm.get_config("epistatic")
    model = ssm.get_model("epistatic", cfg)
    model.eval()
    model.to(ssm._DEVICE)
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    grouped = read_contrasts(args.contrasts)
    rows_out, rejected_all, per_domain = [], [], []
    started = time.time()

    for position, domain in enumerate(sorted(grouped), 1):
        structure = Path(args.structures) / (domain + ".pdb")
        if not structure.exists():
            rejected_all.append({"domain": domain, "reason": "no structure",
                                 "contrasts": len(grouped[domain])})
            continue
        pdb = ssm.load_pdb(str(structure), None)
        index = residue_index(pdb)
        requests, usable, rejected = plan_domain(grouped[domain], index, pdb["seq"])
        for row, reason in rejected:
            rejected_all.append({"domain": domain, "site_i": row["site_i"],
                                 "site_j": row["site_j"], "aa_i": row["aa_i"],
                                 "aa_j": row["aa_j"], "reason": reason})
        if not usable:
            continue

        predictions = score_domain(ssm, torch, DataLoader, tied_featurize_mut, Mutation,
                                   pdb, cfg, model, requests, args.batch_size)
        for row, forward, reverse in usable:
            ddg_f, ddg_r = predictions[forward], predictions[reverse]
            rows_out.append({
                "domain": domain, "site_i": row["site_i"], "site_j": row["site_j"],
                "aa_i": row["aa_i"], "aa_j": row["aa_j"],
                "measured_difference": row["difference"],
                "measured_uncertainty": row["uncertainty"],
                "resolved": row["resolved"],
                "ddg_forward": repr(ddg_f), "ddg_reverse": repr(ddg_r),
                "model_difference": repr(ddg_f - ddg_r),
            })
        per_domain.append({"domain": domain, "contrasts": len(usable),
                           "predictions": len(requests)})
        print("  %-6s %4d contrasts, %5d predictions   [%d/%d, %ds]"
              % (domain, len(usable), len(requests), position, len(grouped),
                 time.time() - started), flush=True)

    predictions_path = output / "thermompnn_pair_predictions.csv"
    with predictions_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows_out)

    reasons = defaultdict(int)
    for entry in rejected_all:
        reasons[entry["reason"]] += 1
    write_json(output / "thermompnn_run_record.json", {
        "schema_version": 1,
        "purpose": "thermompnn_d_on_measured_composition_matched_swaps",
        "model": "ThermoMPNN-D-ens1",
        "checkpoint_sha256": sha256(checkpoint),
        "upstream_entry_sha256": sha256(repo / "v2_ssm.py"),
        "substitution": ("Upstream get_ssm_mutations_double replaced by our 578 measured site "
                         "pairs; featurization and run_double called unmodified."),
        "sign_convention": ("ThermoMPNN ddG is more negative when more stabilizing; measured "
                            "stability is higher when more stable. Agreement is a negative "
                            "correlation, as for the frozen ProteinMPNN baseline."),
        "device": args.device,
        "batch_size": args.batch_size,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "domains_requested": len(grouped),
        "domains_scored": len(per_domain),
        "contrasts_scored": len(rows_out),
        "contrasts_rejected": len(rejected_all),
        "rejection_reasons": dict(reasons),
        "elapsed_seconds": round(time.time() - started, 1),
        "per_domain": per_domain,
        "rejected": rejected_all[:2000],
    })
    print("\ncontrasts scored : %d" % len(rows_out))
    print("contrasts skipped: %d  %s" % (len(rejected_all), dict(reasons)))
    print("elapsed          : %.1f min" % ((time.time() - started) / 60))
    print("Complete: " + str(predictions_path), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", type=Path, required=True,
                        help="a ThermoMPNN-D checkout (patched for device selection)")
    parser.add_argument("--contrasts", required=True, help="swap_contrasts.csv")
    parser.add_argument("--structures", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--batch-size", type=int, default=256)
    run(parser.parse_args())


if __name__ == "__main__":
    main()

"""Run two frozen-model checks from a prepared directory, locally or in Slurm."""
import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    for name in ("results", "logs", "metadata"):
        (root / name).mkdir(exist_ok=True)
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    for name, record in config["source_files"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError("Staged input changed: " + name)
    if any((root / "results").iterdir()):
        raise RuntimeError("Results directory is nonempty; stage a new run instead")
    import numpy as np
    import torch
    cuda = torch.cuda.is_available()
    if args.require_cuda and not cuda:
        raise RuntimeError("GPU smoke test requires a usable CUDA device")
    if cuda:
        try:
            torch.arange(4, device="cuda", dtype=torch.float32).square().sum().item()
            torch.cuda.synchronize()
        except RuntimeError as exc:
            raise RuntimeError("CUDA was detected but a kernel failed; use a compatible GPU/PyTorch build.") from exc
    summary = {"purpose": config["purpose"], "job_id": os.environ.get("SLURM_JOB_ID"),
               "host": platform.node(), "started_utc": datetime.now(timezone.utc).isoformat(),
               "python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
               "cuda_available": cuda, "cuda_version": torch.version.cuda,
               "gpu": torch.cuda.get_device_name(0) if cuda else None, "runs": []}
    metadata = root / "metadata"
    with (metadata / "pip-freeze.txt").open("w", encoding="utf-8") as packages:
        subprocess.run([sys.executable, "-m", "pip", "freeze"], check=True, stdout=packages)
    vendor = root / "vendor" / "ProteinMPNN"
    success = False
    try:
        for model_type in ("vanilla", "soluble"):
            out = root / "results" / model_type
            command = [sys.executable, str(vendor / "protein_mpnn_run.py"),
                       "--pdb_path", str(root / "inputs" / "1UBQ.pdb"),
                       "--pdb_path_chains", config["chain"], "--out_folder", str(out),
                       "--path_to_model_weights", str(vendor / (model_type + "_model_weights")),
                       "--model_name", config["model"], "--seed", str(config["seed"]),
                       "--num_seq_per_target", str(config["samples_per_model"]),
                       "--batch_size", "1", "--sampling_temp", config["temperature"],
                       "--save_score", "1"]
            if model_type == "soluble":
                command.append("--use_soluble_model")
            started = time.monotonic()
            with (root / "logs" / (model_type + ".log")).open("w", encoding="utf-8") as log:
                result = subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT)
            record = {"model_type": model_type, "command": command,
                      "elapsed_seconds": time.monotonic() - started, "exit_code": result.returncode}
            summary["runs"].append(record)
            if result.returncode:
                raise RuntimeError(model_type + " failed; inspect its retained log")
            lines = (out / "seqs" / "1UBQ.fa").read_text(encoding="utf-8").splitlines()
            sequences = [line for line in lines if line and not line.startswith(">")]
            if len(sequences) != config["samples_per_model"] + 1 or any(len(s) != 76 for s in sequences):
                raise ValueError("Unexpected FASTA count or chain length")
            score_files = list((out / "scores").glob("*.npz"))
            if not score_files:
                raise ValueError("Missing saved numerical scores")
            for score_file in score_files:
                with np.load(score_file, allow_pickle=False) as values:
                    for key in values.files:
                        if np.issubdtype(values[key].dtype, np.number) and not np.isfinite(values[key]).all():
                            raise ValueError("Nonfinite saved scores: " + key)
            record["generated_sequences"] = len(sequences) - 1
            record["sequence_length"] = 76
            print(model_type + ": output and score checks passed", flush=True)
        success = True
    finally:
        summary["status"] = "complete" if success else "failed"
        summary["finished_utc"] = datetime.now(timezone.utc).isoformat()
        (metadata / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

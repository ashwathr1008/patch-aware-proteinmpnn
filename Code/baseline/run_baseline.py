"""Run a small, offline ProteinMPNN development batch with retained provenance."""

import argparse
import csv
import hashlib
import importlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path


MODEL = "v_48_020"
UPSTREAM_COMMIT = "8907e6671bfbfc92303b5f79c4b5e6ce47cdef57"
UPSTREAM_HASHES = {
    "protein_mpnn_run.py": "61f2c519a7f73fa12da9eb90da97b97ec2f8d5f31d42605639c7600cbd321cbe",
    "protein_mpnn_utils.py": "74c8f9b7553422a7a0bbd705874844ee103c8926c2c96f154a87e0b824071e1b",
    "vanilla_model_weights/v_48_020.pt": "c9cb4a671d79604111231f8dbfc7c590e06f1197453b7a6854ac6661a642f5bd",
    "soluble_model_weights/v_48_020.pt": "7af52d090172c230c7f0e9d21e02203f6b3a38b16db58d3c7a3960e0a9a6e31a",
}
AA3 = dict(zip(
    "ALA CYS ASP GLU PHE GLY HIS ILE LYS LEU MET ASN PRO GLN ARG SER THR VAL TRP TYR".split(),
    "ACDEFGHIKLMNPQRSTVWY",
))
CANONICAL = frozenset(AA3.values())
BACKBONE = ("N", "CA", "C", "O")
CSV_FIELDS = [
    "structure_id", "chain", "model_type", "model_name", "record_type",
    "sequence_id", "sequence_sha256", "sample_index", "seed", "temperature",
    "length", "native_identity_count", "native_recovery", "score", "global_score",
    "score_source", "sequence", "runtime_seconds", "fasta_path", "scores_path",
]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_manifest(path):
    manifest = json.loads(path.read_text(encoding="utf-8-sig"))
    if manifest.get("schema_version") != 1 or manifest.get("purpose") != "development_only":
        raise ValueError("Manifest must have schema_version=1 and purpose='development_only'")
    structures = manifest.get("structures")
    if not isinstance(structures, list) or not structures:
        raise ValueError("Manifest needs a nonempty structures list")
    seen = set()
    for record in structures:
        structure_id, chain = record["structure_id"], record["chain"]
        if not isinstance(structure_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", structure_id):
            raise ValueError("structure_id must contain only letters, digits, '_' or '-'")
        if not isinstance(chain, str) or not re.fullmatch(r"[A-Za-z0-9]", chain):
            raise ValueError("This starter supports a single alphanumeric chain ID")
        key = (structure_id.casefold(), chain.casefold())
        if key in seen:
            raise ValueError("Duplicate structure/chain in manifest")
        seen.add(key)
        expected_hash = record.get("sha256", "")
        if not re.fullmatch(r"[0-9a-fA-F]{64}", expected_hash):
            raise ValueError("Each manifest input requires a SHA256")
        pdb_path = (path.parent / record["pdb_path"]).resolve()
        if sha256(pdb_path) != expected_hash.lower():
            raise ValueError("Input checksum mismatch: " + str(pdb_path))
        record["resolved_path"] = str(pdb_path)
        if "expected_length" in record:
            if type(record["expected_length"]) is not int or record["expected_length"] < 1:
                raise ValueError("expected_length must be a positive integer")
    return manifest


def stage_backbone(source, chain, destination):
    """Choose the first model and a complete canonical backbone, preserving IDs.

    Alternate conformers are selected per residue, using shared blank-altloc atoms
    and one complete conformer. Highest summed backbone occupancy wins; ties use
    the lexical altloc label. Other chains, side chains and HETATM are not staged.
    """
    residues = {}
    model_label = None
    saw_model = False
    for line in Path(source).read_text(encoding="utf-8").splitlines():
        kind = line[:6].strip()
        if kind == "MODEL":
            if saw_model:
                break
            saw_model = True
            model_label = line[10:14].strip() or "first"
            continue
        if kind == "ENDMDL":
            break
        if kind != "ATOM" or line[21:22] != chain:
            continue
        resname = line[17:20]
        if resname not in AA3:
            raise ValueError("Noncanonical ATOM residue: " + resname)
        key = (int(line[22:26]), line[26:27].strip())
        item = residues.setdefault(key, {"name": resname, "atoms": {}})
        if item["name"] != resname:
            raise ValueError("Multiple residue identities at " + str(key))
        atom = line[12:16].strip()
        if atom not in BACKBONE:
            continue
        coordinates = tuple(float(line[start:start + 8]) for start in (30, 38, 46))
        occupancy = float(line[54:60].strip() or "0")
        if not all(math.isfinite(x) for x in coordinates + (occupancy,)):
            raise ValueError("Nonfinite PDB coordinates/occupancy at " + str(key))
        altloc = line[16:17].strip()
        slot = item["atoms"].setdefault(atom, {})
        if altloc in slot:
            raise ValueError("Duplicate backbone atom/altloc at " + str(key))
        slot[altloc] = (line, occupancy)
    if not residues:
        raise ValueError("No ATOM residues found in requested first-model chain")
    staged_lines, mapping, native = [], [], []
    for position, (key, item) in enumerate(sorted(residues.items()), 1):
        atoms = item["atoms"]
        labels = {""} | {label for options in atoms.values() for label in options}
        choices = []
        for label in sorted(labels):
            selected = []
            for atom in BACKBONE:
                options = atoms.get(atom, {})
                value = options.get(label, options.get(""))
                if value is None:
                    break
                selected.append(value)
            if len(selected) == len(BACKBONE):
                choices.append((-sum(value[1] for value in selected), label, selected))
        if not choices:
            raise ValueError("Incomplete N/CA/C/O backbone at " + str(key))
        _, label, selected = min(choices, key=lambda choice: (choice[0], choice[1]))
        for line, _ in selected:
            staged_lines.append(line[:16] + " " + line[17:])
        native.append(AA3[item["name"]])
        mapping.append({"position": position, "chain": chain, "residue_number": key[0],
                        "insertion_code": key[1], "resname": item["name"],
                        "aa": AA3[item["name"]], "selected_altloc": label})
    destination.write_text("\n".join(staged_lines + ["TER", "END", ""]), encoding="utf-8")
    return {"native_sequence": "".join(native), "length": len(native),
            "selected_model": model_label, "residue_mapping": mapping,
            "preprocessing": "first model; selected chain; complete canonical N/CA/C/O only; original residue IDs retained"}


def read_fasta(path):
    records = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if not line[1:].strip():
                raise ValueError("Empty FASTA header")
            records.append([line[1:], ""])
        elif not records:
            raise ValueError("Sequence precedes FASTA header")
        else:
            records[-1][1] += line
    if not records or any(not sequence for _, sequence in records):
        raise ValueError("Empty FASTA or sequence")
    return records


def numeric_header(header, key):
    match = re.search(r"(?:^|,\s*)" + re.escape(key) + r"=([^,]+)", header)
    if not match:
        raise ValueError("Missing FASTA header field: " + key)
    value = float(match.group(1))
    if not math.isfinite(value):
        raise ValueError("Nonfinite FASTA header field: " + key)
    return value


def validate_outputs(output, structure, model_type, samples, seed, temperature, runtime, root):
    import numpy as np

    fasta_files = list((output / "seqs").glob("*.fa"))
    score_files = list((output / "scores").glob("*.npz"))
    if len(fasta_files) != 1 or len(score_files) != 1:
        raise ValueError("Expected one FASTA and one score archive per run")
    fasta, score_path = fasta_files[0], score_files[0]
    records = read_fasta(fasta)
    if len(records) != samples + 1:
        raise ValueError("Unexpected FASTA record count")
    native = structure["native_sequence"]
    if records[0][1] != native:
        raise ValueError("Native FASTA does not match the parsed input backbone")
    if numeric_header(records[0][0], "seed") != seed:
        raise ValueError("Native FASTA reports the wrong seed")
    with np.load(score_path, allow_pickle=False) as archive:
        arrays = {}
        for key in ("score", "global_score"):
            array = archive[key]
            if array.shape != (samples,) or not np.issubdtype(array.dtype, np.number):
                raise ValueError("Unexpected saved score shape/type: " + key)
            if not np.isfinite(array).all():
                raise ValueError("Nonfinite saved score: " + key)
            arrays[key] = array.copy()
    rows = []
    for index, (header, sequence) in enumerate(records):
        if len(sequence) != len(native) or not set(sequence).issubset(CANONICAL):
            raise ValueError("Sequence length or canonical-AA validation failed")
        identities = sum(left == right for left, right in zip(sequence, native))
        recovery = identities / len(native)
        header_scores = {key: numeric_header(header, key) for key in ("score", "global_score")}
        if index:
            if numeric_header(header, "sample") != index:
                raise ValueError("Unexpected sample numbering")
            if not math.isclose(numeric_header(header, "T"), temperature, abs_tol=1e-12):
                raise ValueError("Unexpected sampling temperature")
            if abs(numeric_header(header, "seq_recovery") - recovery) > 0.000051:
                raise ValueError("Native recovery does not match FASTA sequences")
            scores = {key: float(arrays[key][index - 1]) for key in arrays}
            if any(abs(scores[key] - header_scores[key]) > 0.000051 for key in scores):
                raise ValueError("FASTA and saved numerical scores disagree")
        else:
            scores = header_scores
        record_type = "sample" if index else "native"
        sequence_id = "{structure_id}_{chain}_{model}_{kind}_{index:03d}".format(
            structure_id=structure["structure_id"], chain=structure["chain"],
            model=model_type, kind=record_type, index=index)
        rows.append({
            "structure_id": structure["structure_id"], "chain": structure["chain"],
            "model_type": model_type, "model_name": MODEL, "record_type": record_type,
            "sequence_id": sequence_id, "sequence_sha256": hashlib.sha256(sequence.encode("ascii")).hexdigest(),
            "sample_index": index, "seed": seed, "temperature": temperature if index else "",
            "length": len(sequence), "native_identity_count": identities, "native_recovery": recovery,
            "score": scores["score"], "global_score": scores["global_score"],
            "score_source": "npz_float32" if index else "fasta_rounded_4dp",
            "sequence": sequence, "runtime_seconds": runtime,
            "fasta_path": fasta.relative_to(root).as_posix(),
            "scores_path": score_path.relative_to(root).as_posix() if index else "",
        })
    validation = {"status": "passed", "generated_count": samples, "native_count": 1,
                  "sequence_length": len(native), "finite_scores": True,
                  "canonical_amino_acids": True, "native_sequence_matches_input": True,
                  "recovery_recomputed": True, "fasta_npz_scores_agree": True,
                  "fasta_sha256": sha256(fasta), "scores_sha256": sha256(score_path)}
    return rows, validation


def write_candidates(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def device_preflight(device):
    import numpy as np
    import torch

    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but torch.cuda.is_available() is false")
    started = time.monotonic()
    tensor = torch.arange(256, dtype=torch.float32, device=device).reshape(16, 16) / 256
    result = torch.sin(tensor) + tensor @ tensor.T
    if device == "cuda":
        torch.cuda.synchronize()
    if not bool(torch.isfinite(result).all().item()):
        raise RuntimeError("Device tensor-operation preflight produced nonfinite values")
    environment = {
        "python": sys.version, "python_executable": sys.executable,
        "platform": platform.platform(), "host": platform.node(),
        "torch": str(torch.__version__), "numpy": np.__version__, "device_requested": device,
        "cuda_available": torch.cuda.is_available(), "torch_cuda_version": torch.version.cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "omp_num_threads": os.environ["OMP_NUM_THREADS"], "mkl_num_threads": os.environ["MKL_NUM_THREADS"],
        "preflight": {"status": "passed", "operations": "arange, divide, sin, matmul, add, isfinite; CUDA synchronize if requested",
                      "elapsed_seconds": time.monotonic() - started},
    }
    if device == "cuda":
        environment.update({"gpu": torch.cuda.get_device_name(0),
                            "gpu_compute_capability": list(torch.cuda.get_device_capability(0)),
                            "torch_compiled_cuda_arches": torch.cuda.get_arch_list()})
    return environment


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    for name in ("metadata", "logs", "inputs", "results"):
        (output / name).mkdir()
    summary_path = output / "metadata" / "summary.json"
    summary = {"schema_version": 1, "purpose": "development_only", "status": "running",
               "started_utc": utc_now(), "runs": [], "validated_generated_count": 0,
               "interpretation": "Raw generation-run scores; not controlled-order score comparisons, folding validation, or aggregation/developability measurements."}
    rows = []
    write_json(summary_path, summary)
    try:
        if args.samples < 1 or not 1 <= args.seed < 2 ** 32:
            raise ValueError("samples must be positive; seed must be in [1, 2**32)")
        if not math.isfinite(args.temperature) or args.temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        manifest_path, mpnn_dir = args.manifest.resolve(), args.mpnn_dir.resolve()
        manifest = load_manifest(manifest_path)
        write_json(output / "metadata" / "manifest.json", manifest)
        config = {"samples_per_model_per_structure": args.samples, "seed": args.seed,
                  "temperature": args.temperature, "batch_size": 1, "backbone_noise": 0.0,
                  "model": MODEL, "model_types": ["vanilla", "soluble"],
                  "device": args.device, "upstream_commit": UPSTREAM_COMMIT,
                  "manifest_path": str(manifest_path), "manifest_sha256": sha256(manifest_path),
                  "mpnn_dir": str(mpnn_dir), "runner_sha256": sha256(Path(__file__)),
                  "upstream_files": {}, "arguments": vars(args).copy()}
        config["arguments"] = {key: str(value) if isinstance(value, Path) else value
                               for key, value in config["arguments"].items()}
        for relative, expected in UPSTREAM_HASHES.items():
            actual = sha256(mpnn_dir / relative)
            if actual != expected:
                raise ValueError("Pinned upstream code/checkpoint checksum mismatch: " + relative)
            config["upstream_files"][relative] = {"sha256": actual, "bytes": (mpnn_dir / relative).stat().st_size}
        write_json(output / "metadata" / "config.json", config)
        os.environ["OMP_NUM_THREADS"] = "2"
        os.environ["MKL_NUM_THREADS"] = "2"
        if args.device == "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        summary["environment"] = device_preflight(args.device)
        write_json(output / "metadata" / "environment.json", summary["environment"])
        with (output / "metadata" / "pip-freeze.txt").open("w", encoding="utf-8") as handle:
            freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], stdout=handle, stderr=subprocess.STDOUT)
        summary["pip_freeze_exit_code"] = freeze.returncode
        sys.path.insert(0, str(mpnn_dir))
        utils = importlib.import_module("protein_mpnn_utils")
        import numpy as np

        prepared = []
        for item in manifest["structures"]:
            stem = item["structure_id"] + "_" + item["chain"]
            staged_path = output / "inputs" / (stem + ".pdb")
            info = stage_backbone(item["resolved_path"], item["chain"], staged_path)
            if "expected_length" in item and item["expected_length"] != info["length"]:
                raise ValueError("Parsed native length differs from manifest: " + stem)
            parsed = utils.parse_PDB(staged_path.as_posix(), input_chain_list=[item["chain"]])
            if len(parsed) != 1 or parsed[0]["seq"] != info["native_sequence"]:
                raise ValueError("Upstream parse/native mapping mismatch (numbering gaps are unsupported): " + stem)
            coords = parsed[0]["coords_chain_" + item["chain"]]
            for atom in BACKBONE:
                array = np.asarray(coords[atom + "_chain_" + item["chain"]])
                if array.shape != (info["length"], 3) or not np.isfinite(array).all():
                    raise ValueError("Upstream parsed incomplete/nonfinite backbone: " + stem)
            info.update({"structure_id": item["structure_id"], "chain": item["chain"],
                         "raw_sha256": item["sha256"].lower(), "staged_sha256": sha256(staged_path),
                         "staged_path": staged_path.relative_to(output).as_posix()})
            prepared.append(info)
        write_json(output / "metadata" / "structures.json", prepared)
        summary["expected_generated_count"] = len(prepared) * 2 * args.samples
        for structure in prepared:
            stem = structure["structure_id"] + "_" + structure["chain"]
            for model_type in ("vanilla", "soluble"):
                run_output = output / "results" / stem / model_type
                command = [sys.executable, (mpnn_dir / "protein_mpnn_run.py").as_posix(),
                           "--pdb_path", (output / structure["staged_path"]).as_posix(),
                           "--pdb_path_chains", structure["chain"], "--out_folder", run_output.as_posix(),
                           "--path_to_model_weights", (mpnn_dir / (model_type + "_model_weights")).as_posix(),
                           "--model_name", MODEL, "--seed", str(args.seed),
                           "--num_seq_per_target", str(args.samples), "--batch_size", "1",
                           "--sampling_temp", str(args.temperature), "--save_score", "1",
                           "--backbone_noise", "0.0", "--omit_AAs", "X"]
                if model_type == "soluble":
                    command.append("--use_soluble_model")
                log_path = output / "logs" / (stem + "_" + model_type + ".log")
                record = {"structure_id": structure["structure_id"], "chain": structure["chain"],
                          "model_type": model_type, "command": command, "started_utc": utc_now(),
                          "log_path": log_path.relative_to(output).as_posix(), "status": "running"}
                summary["runs"].append(record)
                write_json(summary_path, summary)
                write_json(output / "metadata" / (stem + "_" + model_type + "_command.json"), command)
                print("Starting " + stem + " / " + model_type, flush=True)
                started = time.monotonic()
                with log_path.open("w", encoding="utf-8") as log:
                    process = subprocess.run(command, cwd=output, env=os.environ.copy(), stdout=log, stderr=subprocess.STDOUT)
                record.update({"elapsed_seconds": time.monotonic() - started,
                               "exit_code": process.returncode, "finished_utc": utc_now()})
                if process.returncode:
                    record["status"] = "failed"
                    raise RuntimeError("ProteinMPNN failed; see " + str(log_path))
                run_rows, validation = validate_outputs(run_output, structure, model_type, args.samples,
                                                        args.seed, args.temperature, record["elapsed_seconds"], output)
                rows.extend(run_rows)
                record.update({"status": "complete", "validation": validation})
                summary["validated_generated_count"] += args.samples
                write_candidates(output / "candidates.csv", rows)
                write_json(summary_path, summary)
                print(stem + " / " + model_type + ": " + str(args.samples) + " samples validated", flush=True)
        summary["status"] = "complete"
        summary["candidates_csv_sha256"] = sha256(output / "candidates.csv")
        summary["candidates_row_count"] = len(rows)
    except BaseException as error:
        summary["status"] = "failed"
        summary["error"] = {"type": type(error).__name__, "message": str(error)}
        if summary["runs"] and summary["runs"][-1]["status"] == "running":
            summary["runs"][-1]["status"] = "failed"
        (output / "metadata" / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        summary["finished_utc"] = utc_now()
        write_json(summary_path, summary)
    print("Complete: " + str(output / "candidates.csv"), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--mpnn-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="A new directory; existing paths are refused")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    run(parser.parse_args())


if __name__ == "__main__":
    main()

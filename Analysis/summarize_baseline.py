"""Export a completed development baseline as portable tables and recovery plots."""

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path


CANDIDATE_FIELDS = [
    "structure_id", "chain", "model_type", "model_name", "record_type",
    "sequence_id", "sequence_sha256", "sample_index", "seed", "temperature",
    "length", "native_identity_count", "native_recovery", "score", "global_score",
    "score_source", "sequence", "runtime_seconds",
]
SUMMARY_FIELDS = [
    "structure_id", "chain", "length", "model_type", "model_name", "sample_count",
    "unique_sequences", "native_recovery_mean", "native_recovery_min", "native_recovery_max",
    "raw_generation_score_mean", "raw_generation_global_score_mean", "seed", "temperature",
    "subprocess_elapsed_seconds",
]
ENVIRONMENT_FIELDS = (
    "python", "platform", "torch", "numpy", "device_requested", "cuda_available",
    "torch_cuda_version", "gpu", "gpu_compute_capability", "torch_compiled_cuda_arches",
    "omp_num_threads", "mkl_num_threads", "preflight",
)
CONFIG_FIELDS = (
    "samples_per_model_per_structure", "seed", "temperature", "batch_size", "backbone_noise",
    "model", "model_types", "device", "upstream_commit", "manifest_sha256", "runner_sha256",
)
STRUCTURE_FIELDS = (
    "structure_id", "chain", "length", "native_sequence", "selected_model",
    "raw_sha256", "staged_sha256", "preprocessing",
)
VALIDATION_FIELDS = (
    "status", "generated_count", "native_count", "sequence_length", "finite_scores",
    "canonical_amino_acids", "native_sequence_matches_input", "recovery_recomputed",
    "fasta_npz_scores_agree", "fasta_sha256", "scores_sha256",
)
UPSTREAM_FILES = (
    "protein_mpnn_run.py", "protein_mpnn_utils.py",
    "vanilla_model_weights/v_48_020.pt", "soluble_model_weights/v_48_020.pt",
)
CANONICAL = frozenset("ACDEFGHIKLMNPQRSTVWY")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def finite(value, label):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Nonfinite " + label)
    return result


def subset(value, keys):
    return {key: value[key] for key in keys if key in value}


def write_csv(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def collect_run(run_dir):
    """Validate the completed source, then return shareable generated-only summaries."""
    metadata = run_dir / "metadata"
    summary = load_json(metadata / "summary.json")
    config = load_json(metadata / "config.json")
    structures = load_json(metadata / "structures.json")
    if summary.get("status") != "complete" or summary.get("purpose") != "development_only":
        raise ValueError("Analysis requires a completed development-only run")
    candidates_path = run_dir / "candidates.csv"
    if sha256(candidates_path) != summary.get("candidates_csv_sha256"):
        raise ValueError("Candidate CSV checksum differs from completed run summary")
    expected_samples = config["samples_per_model_per_structure"]
    if type(expected_samples) is not int or expected_samples < 1:
        raise ValueError("Invalid configured sample count")
    if config.get("model_types") != ["vanilla", "soluble"]:
        raise ValueError("Expected the vanilla and soluble model configurations")
    structure_map = {(item["structure_id"], item["chain"]): item for item in structures}
    if len(structure_map) != len(structures) or not structures:
        raise ValueError("Missing or duplicate structure metadata")
    expected_groups = {(sid, chain, model) for sid, chain in structure_map
                       for model in config["model_types"]}
    runs = {}
    for run in summary.get("runs", []):
        key = (run["structure_id"], run["chain"], run["model_type"])
        if key in runs or key not in expected_groups:
            raise ValueError("Duplicate or unexpected model subprocess")
        validation = run.get("validation", {})
        if run.get("status") != "complete" or run.get("exit_code") != 0 or validation.get("status") != "passed":
            raise ValueError("Every subprocess must complete and pass validation")
        for check in ("finite_scores", "canonical_amino_acids", "native_sequence_matches_input",
                      "recovery_recomputed", "fasta_npz_scores_agree"):
            if validation.get(check) is not True:
                raise ValueError("Missing successful run validation: " + check)
        if validation.get("generated_count") != expected_samples or validation.get("native_count") != 1:
            raise ValueError("Run validation reports unexpected record counts")
        if validation.get("sequence_length") != structure_map[key[:2]]["length"]:
            raise ValueError("Validation sequence length differs from structure metadata")
        if finite(run["elapsed_seconds"], "subprocess duration") < 0:
            raise ValueError("Negative subprocess duration")
        runs[key] = run
    if set(runs) != expected_groups:
        raise ValueError("Missing completed structure/model runs")
    expected_generated = len(expected_groups) * expected_samples
    if (summary.get("expected_generated_count") != expected_generated
            or summary.get("validated_generated_count") != expected_generated):
        raise ValueError("Incomplete validated generation count")
    with candidates_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not set(CANDIDATE_FIELDS).issubset(reader.fieldnames):
            raise ValueError("Candidate CSV is missing required columns")
        rows = list(reader)
    if len(rows) != summary.get("candidates_row_count") or len(rows) != expected_generated + len(expected_groups):
        raise ValueError("Candidate count differs from completed-run totals")
    groups = defaultdict(list)
    native_groups = defaultdict(list)
    sequence_ids = set()
    for row in rows:
        key = (row["structure_id"], row["chain"], row["model_type"])
        if key not in expected_groups or row["model_name"] != config["model"]:
            raise ValueError("Unexpected candidate structure/checkpoint")
        if row["sequence_id"] in sequence_ids:
            raise ValueError("Duplicate candidate identifier")
        sequence_ids.add(row["sequence_id"])
        sequence = row["sequence"]
        structure = structure_map[key[:2]]
        native = structure["native_sequence"]
        if (len(sequence) != structure["length"] or int(row["length"]) != len(sequence)
                or not set(sequence).issubset(CANONICAL)):
            raise ValueError("Candidate length or amino-acid validation failed")
        if hashlib.sha256(sequence.encode("ascii")).hexdigest() != row["sequence_sha256"]:
            raise ValueError("Candidate sequence checksum mismatch")
        identities = sum(a == b for a, b in zip(sequence, native))
        recovery = identities / len(native)
        if int(row["native_identity_count"]) != identities or not math.isclose(
                finite(row["native_recovery"], "recovery"), recovery, rel_tol=0, abs_tol=1e-12):
            raise ValueError("Candidate native identity/recovery mismatch")
        for field in ("score", "global_score", "runtime_seconds"):
            finite(row[field], field)
        if not math.isclose(float(row["runtime_seconds"]), runs[key]["elapsed_seconds"], rel_tol=1e-12):
            raise ValueError("Candidate runtime differs from subprocess record")
        if int(row["seed"]) != config["seed"]:
            raise ValueError("Candidate seed differs from run configuration")
        if row["record_type"] == "sample":
            if row["score_source"] != "npz_float32":
                raise ValueError("Generated score must come from the numerical archive")
            if not math.isclose(finite(row["temperature"], "temperature"), config["temperature"], rel_tol=1e-12):
                raise ValueError("Sampling temperature differs from run configuration")
            groups[key].append(row)
        elif row["record_type"] == "native":
            if sequence != native or int(row["sample_index"]) != 0:
                raise ValueError("Invalid native record")
            if row["score_source"] != "fasta_rounded_4dp":
                raise ValueError("Native score provenance is incorrect")
            native_groups[key].append(row)
        else:
            raise ValueError("Unknown candidate record type")
    table = []
    for key in sorted(expected_groups):
        generated = sorted(groups[key], key=lambda row: int(row["sample_index"]))
        if len(native_groups[key]) != 1 or [int(row["sample_index"]) for row in generated] != list(range(1, expected_samples + 1)):
            raise ValueError("Each model/structure needs one native and every expected sample")
        recovery = [float(row["native_recovery"]) for row in generated]
        table.append({
            "structure_id": key[0], "chain": key[1], "length": structure_map[key[:2]]["length"],
            "model_type": key[2], "model_name": config["model"], "sample_count": len(generated),
            "unique_sequences": len({row["sequence"] for row in generated}),
            "native_recovery_mean": statistics.mean(recovery), "native_recovery_min": min(recovery),
            "native_recovery_max": max(recovery),
            "raw_generation_score_mean": statistics.mean(float(row["score"]) for row in generated),
            "raw_generation_global_score_mean": statistics.mean(float(row["global_score"]) for row in generated),
            "seed": config["seed"], "temperature": config["temperature"],
            "subprocess_elapsed_seconds": runs[key]["elapsed_seconds"],
        })
    wall = (datetime.fromisoformat(summary["finished_utc"]) - datetime.fromisoformat(summary["started_utc"])).total_seconds()
    subprocess_total = sum(run["elapsed_seconds"] for run in runs.values())
    if wall < 0 or wall + 0.1 < subprocess_total:
        raise ValueError("Pipeline wall time is inconsistent with sequential subprocess durations")
    provenance = {
        "schema_version": 1, "purpose": "development_only", "source_run_status": "complete",
        "started_utc": summary["started_utc"], "finished_utc": summary["finished_utc"],
        "sample_count": expected_generated, "native_record_count": len(expected_groups),
        "config": subset(config, CONFIG_FIELDS),
        "upstream_files": {name: subset(config["upstream_files"][name], ("sha256", "bytes"))
                           for name in UPSTREAM_FILES},
        "structures": [subset(item, STRUCTURE_FIELDS) for item in structures],
        "environment": subset(summary["environment"], ENVIRONMENT_FIELDS),
        "timing": {"subprocess_count": len(runs), "subprocess_elapsed_total_seconds": subprocess_total,
                   "pipeline_wall_seconds": wall,
                   "definitions": "Subprocess time includes interpreter/model loading plus generation. Pipeline wall time additionally includes staging, preflight, package listing and validation."},
        "runs": [{**subset(runs[key], ("structure_id", "chain", "model_type", "elapsed_seconds", "exit_code", "status")),
                  "validation": subset(runs[key]["validation"], VALIDATION_FIELDS)} for key in sorted(runs)],
        "source_file_sha256": {name: sha256(run_dir / name) for name in (
            "candidates.csv", "metadata/summary.json", "metadata/config.json", "metadata/structures.json", "metadata/manifest.json")},
        "analysis_script_sha256": sha256(Path(__file__)),
        "interpretation": [
            "Summary statistics include generated sequences only; native rows are excluded.",
            "Native recovery is the fraction of identities against the input sequence; it is not a protein-quality measurement.",
            "Every generated sample is plotted with a mean mark; no inferential intervals are estimated.",
            "Scores are raw generation-run mean negative log probabilities. They are not comparable improvements across checkpoints or controlled-order rescoring.",
            "These small development inputs are not a held-out benchmark. No folding, aggregation or stability validation is represented.",
        ],
    }
    portable_rows = [subset(row, CANDIDATE_FIELDS) for row in rows]
    return portable_rows, table, provenance


def plot_recovery(rows, table, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import PercentFormatter

    groups = defaultdict(list)
    for row in rows:
        if row["record_type"] == "sample":
            groups[(row["structure_id"], row["chain"], row["model_type"])].append(row)
    structures = sorted({(row["structure_id"], row["chain"], row["length"]) for row in table})
    colors = {"vanilla": "#0072B2", "soluble": "#D55E00"}
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42}):
        figure, axis = plt.subplots(figsize=(8.5, 4.9))
        for index, (sid, chain, _) in enumerate(structures):
            for model, offset in (("vanilla", -0.15), ("soluble", 0.15)):
                samples = sorted(groups[(sid, chain, model)], key=lambda row: int(row["sample_index"]))
                values = [float(row["native_recovery"]) for row in samples]
                jitter = [0.0] if len(samples) == 1 else [0.14 * (i / (len(samples) - 1) - 0.5) for i in range(len(samples))]
                center = index + offset
                axis.scatter([center + shift for shift in jitter], values, color=colors[model],
                             marker="o" if model == "vanilla" else "s", s=43,
                             alpha=0.8, edgecolors="white", linewidths=0.5, zorder=3)
                mean = statistics.mean(values)
                axis.plot([center - 0.09, center + 0.09], [mean, mean], color="#222222", linewidth=2.5, zorder=4)
        axis.set_xticks(range(len(structures)), [f"{sid} chain {chain}\n{length} residues" for sid, chain, length in structures])
        axis.set_xlim(-0.55, len(structures) - 0.45)
        axis.set_ylim(-0.025, 1.025)
        axis.set_yticks([i / 5 for i in range(6)])
        axis.yaxis.set_major_formatter(PercentFormatter(1.0))
        axis.set_ylabel("Native recovery (sequence identity)")
        axis.grid(axis="y", color="#dddddd", linewidth=0.6)
        axis.set_axisbelow(True)
        figure.suptitle("ProteinMPNN development baseline", fontsize=14, y=0.98)
        axis.set_title("Each point is one generated sequence", fontsize=10, pad=12)
        handles = [Line2D([], [], color=colors[model], marker=marker, linestyle="none", label=label)
                   for model, marker, label in (("vanilla", "o", "Vanilla"), ("soluble", "s", "Soluble"))]
        handles.append(Line2D([], [], color="#222222", linewidth=2.5, label="Mean"))
        axis.legend(handles=handles, loc="upper right", frameon=True, fontsize=9)
        sizes = sorted({row["sample_count"] for row in table})
        size_label = str(sizes[0]) if len(sizes) == 1 else "/".join(map(str, sizes))
        figure.text(0.5, 0.025, f"n = {size_label} per model and backbone. Recovery is not a protein-quality measure.", ha="center", fontsize=9)
        figure.tight_layout(rect=(0, 0.065, 1, 0.96))
        figure.savefig(output / "native_recovery.png", dpi=220,
                       metadata={"Software": "Matplotlib " + matplotlib.__version__, "Title": "ProteinMPNN native recovery"})
        figure.savefig(output / "native_recovery.pdf", metadata={"Title": "ProteinMPNN native recovery",
                       "Creator": "summarize_baseline.py", "CreationDate": None, "ModDate": None})
        plt.close(figure)
    return {"matplotlib": matplotlib.__version__, "font": "DejaVu Sans", "png_dpi": 220,
            "figure_inches": [8.5, 4.9], "point_jitter": "deterministic evenly spaced positions by sample index",
            "intervals": "none", "summary_marker": "arithmetic mean of generated native recovery"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New shareable artifact directory")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("Analysis output must be a new directory")
    rows, table, record = collect_run(args.run_dir.resolve())
    output.mkdir(parents=True, exist_ok=False)
    record["analysis_status"] = "running"
    write_json(output / "run_record.json", record)
    try:
        write_csv(output / "candidates.csv", CANDIDATE_FIELDS, rows)
        write_csv(output / "summary.csv", SUMMARY_FIELDS, table)
        record["plot_config"] = plot_recovery(rows, table, output)
        record["artifacts_sha256"] = {name: sha256(output / name) for name in (
            "candidates.csv", "summary.csv", "native_recovery.png", "native_recovery.pdf")}
        record["analysis_status"] = "complete"
    except BaseException as error:
        record["analysis_status"] = "failed"
        record["analysis_error_type"] = type(error).__name__
        raise
    finally:
        write_json(output / "run_record.json", record)
    print("Exported " + str(record["sample_count"]) + " generated sequences across " + str(len(table)) + " groups to " + str(output), flush=True)


if __name__ == "__main__":
    main()

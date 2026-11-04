"""Checks that report exports reject incomplete data and omit personal paths."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import summarize_baseline as analysis


class ExportValidationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.metadata = self.root / "metadata"
        self.metadata.mkdir()
        self.rows = []
        runs = []
        secret = "PERSONAL_PATH_TOKEN"
        for model in ("vanilla", "soluble"):
            for index, sequence in enumerate(("ACD", "ACE", "CDE")):
                identity = sum(a == b for a, b in zip(sequence, "ACD"))
                self.rows.append({
                    "structure_id": "fixture", "chain": "A", "model_type": model,
                    "model_name": "v_48_020", "record_type": "sample" if index else "native",
                    "sequence_id": model + str(index), "sequence": sequence,
                    "sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
                    "sample_index": index, "seed": 37, "temperature": 0.1 if index else "",
                    "length": 3, "native_identity_count": identity, "native_recovery": identity / 3,
                    "score": 0.8, "global_score": 0.8,
                    "score_source": "npz_float32" if index else "fasta_rounded_4dp",
                    "runtime_seconds": 2.0,
                })
            runs.append({"structure_id": "fixture", "chain": "A", "model_type": model,
                         "elapsed_seconds": 2.0, "exit_code": 0, "status": "complete",
                         "command": [secret], "log_path": secret,
                         "validation": {"status": "passed", "generated_count": 2, "native_count": 1,
                                        "sequence_length": 3, "finite_scores": True,
                                        "canonical_amino_acids": True, "native_sequence_matches_input": True,
                                        "recovery_recomputed": True, "fasta_npz_scores_agree": True,
                                        "fasta_sha256": "0" * 64, "scores_sha256": "0" * 64}})
        analysis.write_csv(self.root / "candidates.csv", analysis.CANDIDATE_FIELDS, self.rows)
        self.summary = {
            "status": "complete", "purpose": "development_only", "candidates_csv_sha256": analysis.sha256(self.root / "candidates.csv"),
            "candidates_row_count": 6, "expected_generated_count": 4, "validated_generated_count": 4,
            "started_utc": "2026-11-04T12:00:00+00:00", "finished_utc": "2026-11-04T12:00:05+00:00",
            "runs": runs, "environment": {"python": "3.12 fixture", "gpu": "fixture GPU",
                                          "host": secret, "python_executable": secret},
        }
        analysis.write_json(self.metadata / "summary.json", self.summary)
        config = {"samples_per_model_per_structure": 2, "seed": 37, "temperature": 0.1,
                  "model": "v_48_020", "model_types": ["vanilla", "soluble"],
                  "manifest_path": secret, "mpnn_dir": secret, "arguments": {"output": secret},
                  "upstream_files": {name: {"sha256": "0" * 64, "bytes": 1} for name in analysis.UPSTREAM_FILES}}
        analysis.write_json(self.metadata / "config.json", config)
        analysis.write_json(self.metadata / "structures.json", [{"structure_id": "fixture", "chain": "A", "length": 3,
                            "native_sequence": "ACD", "staged_path": secret, "raw_sha256": "0" * 64,
                            "staged_sha256": "0" * 64}])
        analysis.write_json(self.metadata / "manifest.json", {"resolved_path": secret})

    def test_generated_only_statistics_portable_provenance_and_distinct_runtimes(self):
        rows, table, record = analysis.collect_run(self.root)
        self.assertEqual(len(rows), 6)
        self.assertEqual(len(table), 2)
        self.assertAlmostEqual(table[0]["native_recovery_mean"], 1 / 3)
        self.assertEqual(table[0]["sample_count"], 2)
        self.assertEqual(table[0]["unique_sequences"], 2)
        self.assertEqual(record["timing"]["subprocess_elapsed_total_seconds"], 4.0)
        self.assertEqual(record["timing"]["pipeline_wall_seconds"], 5.0)
        self.assertNotIn("PERSONAL_PATH_TOKEN", json.dumps(record))

    def test_incomplete_and_failed_subprocess_cannot_be_exported(self):
        self.summary["runs"][0]["status"] = "failed"
        analysis.write_json(self.metadata / "summary.json", self.summary)
        with self.assertRaisesRegex(ValueError, "Every subprocess"):
            analysis.collect_run(self.root)
        self.summary["status"] = "running"
        analysis.write_json(self.metadata / "summary.json", self.summary)
        with self.assertRaisesRegex(ValueError, "completed"):
            analysis.collect_run(self.root)

    def test_changed_candidate_file_is_rejected(self):
        with (self.root / "candidates.csv").open("a", encoding="utf-8") as handle:
            handle.write("\n")
        with self.assertRaisesRegex(ValueError, "checksum"):
            analysis.collect_run(self.root)

    def test_nonfinite_scores_and_incorrect_recovery_are_rejected(self):
        for field, value, message in (("score", float("nan"), "Nonfinite"),
                                      ("native_recovery", 0.5, "recovery mismatch")):
            with self.subTest(field=field):
                changed = [dict(row) for row in self.rows]
                changed[1][field] = value
                analysis.write_csv(self.root / "candidates.csv", analysis.CANDIDATE_FIELDS, changed)
                self.summary["candidates_csv_sha256"] = analysis.sha256(self.root / "candidates.csv")
                analysis.write_json(self.metadata / "summary.json", self.summary)
                with self.assertRaisesRegex(ValueError, message):
                    analysis.collect_run(self.root)


if __name__ == "__main__":
    unittest.main()

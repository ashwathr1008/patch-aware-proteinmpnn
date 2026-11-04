"""Check paired-score coverage, arithmetic, grouping and audit enforcement."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import summarize_pairs as analysis


def order_hash(index):
    values = [1, 2, 3, 4] if index == 1 else [4, 3, 2, 1]
    return hashlib.sha256(json.dumps(values, separators=(",", ":")).encode("ascii")).hexdigest()


def fixture():
    structures = [{"structure_id": sid, "chain": "A", "split": "development", "group": group}
                  for sid, group in (("p1", "g1"), ("p2", "g1"), ("p3", "g2"))]
    pairs, rows = [], []
    # Unequal pair counts make equal-protein and pair-pooled summaries differ.
    for sid, number, gaps in (("p1", 1, [-0.1, 0.1]), ("p1", 2, [0.2, 0.4]),
                              ("p2", 1, [-0.2, -0.4]), ("p3", 1, [0.6, 0.8])):
        structure = next(s for s in structures if s["structure_id"] == sid)
        pair_id = sid + "_" + str(number)
        pairs.append({"pair_id": pair_id, "structure_id": sid, "chain": "A", "patch_id": sid + "_patch",
                      "original_sequence": "AVLK", "permuted_sequence": "VALK", "patch_indices": [1, 2, 3],
                      "patch_size": 3, "changed_identity_count": 2,
                      "controls": {"composition_tie": True, "geometry_tie": True,
                                   "contact_count": 2, "hydrophobic_contacts_original": 1,
                                   "hydrophobic_contacts_permuted": 2, "hydrophobic_fraction_original": 0.5,
                                   "hydrophobic_fraction_permuted": 1.0, "hydrophobic_fraction_delta": 0.5},
                      "checks": {"composition": True}})
        for order_index, gap in enumerate(gaps, 1):
            rows.append({"pair_id": pair_id, "structure_id": sid, "chain": "A", "split": "development",
                         "group": structure["group"], "patch_id": sid + "_patch", "model_type": "vanilla",
                         "model_name": "v_48_020", "order_index": order_index,
                         "order_sha256": order_hash(order_index), "length": 4, "patch_size": 3,
                         "changed_identity_count": 2, "presentation_first": "original",
                         "original_full_nll": 1.0, "permuted_full_nll": 1.0 + gap, "full_nll_gap": gap,
                         "original_patch_nll": 1.2, "permuted_patch_nll": 1.2 + gap, "patch_nll_gap": gap})
    return {"schema_version": 1, "purpose": "patch_compatibility_pilot", "structures": structures, "pairs": pairs}, rows


class AggregateTests(unittest.TestCase):
    def test_orders_pairs_proteins_and_groups_are_distinct_units(self):
        data, rows = fixture()
        checked, pairs, proteins, groups, splits, controls = analysis.summarize_rows(rows, data, ["vanilla"], 2)
        self.assertEqual(len(checked), 8)
        self.assertEqual(len(pairs), 4)
        self.assertEqual(len(proteins), 3)
        self.assertEqual(len(groups), 2)
        tied = next(r for r in pairs if r["pair_id"] == "p1_1")
        self.assertEqual(tied["native_preference"], 0.5)
        self.assertTrue(tied["order_sign_disagreement"])
        self.assertAlmostEqual(tied["native_preference_across_orders"], 0.5)
        p1 = next(r for r in proteins if r["structure_id"] == "p1")
        self.assertAlmostEqual(p1["full_nll_gap_mean"], 0.15)
        self.assertAlmostEqual(p1["native_preference_fraction"], 0.75)
        self.assertEqual(p1["patch_count"], 1)
        self.assertAlmostEqual(splits[0]["equal_protein_mean_full_nll_gap"], (0.15 - 0.3 + 0.7) / 3)
        self.assertAlmostEqual(splits[0]["equal_group_mean_full_nll_gap"], ((0.15 - 0.3) / 2 + 0.7) / 2)
        self.assertEqual(splits[0]["source_group_count"], 2)
        self.assertEqual(sum(c["composition_tie"] for c in controls), 4)

    def test_missing_duplicate_or_unshared_order_is_rejected(self):
        data, rows = fixture()
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            analysis.summarize_rows(rows[:-1], data, ["vanilla"], 2)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            analysis.summarize_rows(rows + rows[:1], data, ["vanilla"], 2)
        rows[2]["order_sha256"] = "9" * 64
        with self.assertRaisesRegex(ValueError, "Decoding order differs"):
            analysis.summarize_rows(rows, data, ["vanilla"], 2)

    def test_nonfinite_and_wrong_gap_rejected(self):
        for field, value, error in (("original_full_nll", float("nan"), "Nonfinite"),
                                    ("full_nll_gap", 42, "Incorrect signed")):
            data, rows = fixture()
            rows[0][field] = value
            with self.assertRaisesRegex(ValueError, error):
                analysis.summarize_rows(rows, data, ["vanilla"], 2)

    def test_control_failures_and_cross_split_group_rejected(self):
        data, rows = fixture()
        data["pairs"][0]["controls"]["hydrophobic_fraction_delta"] = 0.4
        with self.assertRaisesRegex(ValueError, "descriptor delta"):
            analysis.summarize_rows(rows, data, ["vanilla"], 2)
        data, rows = fixture()
        data["structures"][1]["split"] = "diagnostic"
        for row in rows:
            if row["structure_id"] == "p2":
                row["split"] = "diagnostic"
        with self.assertRaisesRegex(ValueError, "crosses"):
            analysis.summarize_rows(rows, data, ["vanilla"], 2)

    def test_actual_metadata_checksums_and_audits_required(self):
        data, rows = fixture()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata = root / "metadata"
            metadata.mkdir()
            pair_path = root / "input.json"
            analysis.write_json(pair_path, data)
            analysis.write_csv(root / "pair_scores.csv", analysis.SCORE_FIELDS, rows)
            analysis.write_csv(root / "pair_summary.csv", ["dummy"], [{"dummy": 1}])
            audits = [{"structure_id": sid, "chain": "A", "model_type": "vanilla", "order_index": index,
                       "identical_sequence_max_abs_difference": 0, "reversed_presentation_max_abs_difference": 0,
                       "explicit_vs_rank_derived_max_abs_difference": 0 if index == 1 else None,
                       "status": "passed"} for sid in ("p1", "p2", "p3") for index in (1, 2)]
            summary = {"purpose": "patch_compatibility_pilot", "status": "complete", "pair_count": 4,
                       "structure_count": 3, "order_count": 2, "model_types": ["vanilla"],
                       "expected_score_rows": 8, "completed_score_rows": 8, "pair_scores_sha256": analysis.sha256(root / "pair_scores.csv"),
                       "pair_summary_sha256": analysis.sha256(root / "pair_summary.csv"), "model_forward_calls": 28,
                       "order_audit": {"method": "fixture", "verified_forward_calls": 28}, "audits": audits,
                       "elapsed_seconds": 1, "started_utc": "2026-11-04T00:00:00+00:00", "finished_utc": "2026-11-04T00:00:01+00:00"}
            config = {"pairs_sha256": analysis.sha256(pair_path), "models": ["vanilla"], "orders": 2,
                      "model_name": "v_48_020", "backbone_noise": 0, "tie_tolerance": 1e-6,
                      "private_path": "DO_NOT_EXPORT"}
            analysis.write_json(metadata / "config.json", config)
            analysis.write_json(metadata / "summary.json", summary)
            analysis.write_json(metadata / "environment.json", {"python": "fixture", "host": "DO_NOT_EXPORT"})
            analysis.write_json(metadata / "structures.json", data["structures"])
            orders = [{"structure_id": sid, "chain": "A", "orders": [
                {"order_index": i, "order_sha256": order_hash(i), "decoding_order_one_based": [1, 2, 3, 4] if i == 1 else [4, 3, 2, 1]}
                for i in (1, 2)]} for sid in ("p1", "p2", "p3")]
            analysis.write_json(metadata / "orders.json", orders)
            tables, record, _ = analysis.collect_run(root, pair_path)
            self.assertEqual(record["counts"]["source_groups"], 2)
            self.assertNotIn("DO_NOT_EXPORT", json.dumps(record))
            summary["audits"] = audits[:-1]
            analysis.write_json(metadata / "summary.json", summary)
            with self.assertRaisesRegex(ValueError, "Missing null"):
                analysis.collect_run(root, pair_path)
            summary["status"] = "running"
            analysis.write_json(metadata / "summary.json", summary)
            with self.assertRaisesRegex(ValueError, "completed"):
                analysis.collect_run(root, pair_path)


if __name__ == "__main__":
    unittest.main()

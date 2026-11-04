"""Boundary checks for structure staging and exported ProteinMPNN outputs."""

import argparse
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import run_baseline as baseline


def atom_line(serial, atom, resname, number, x, altloc="", occupancy=1.0, chain="A", insertion=""):
    return (f"ATOM  {serial:5d} {atom:^4s}{altloc or ' '}{resname:3s} {chain}"
            f"{number:4d}{insertion or ' '}   {x:8.3f}{2.0:8.3f}{3.0:8.3f}"
            f"{occupancy:6.2f}{20.0:6.2f}          {atom[0]:>2s}")


def complete_residue(resname="ALA", number=1, x=1.0, altloc="", occupancy=1.0):
    return [atom_line(index, atom, resname, number, x, altloc, occupancy)
            for index, atom in enumerate(baseline.BACKBONE, 1)]


class StructureTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def stage(self, lines):
        source, target = self.root / "source.pdb", self.root / "staged.pdb"
        source.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return baseline.stage_backbone(source, "A", target), target

    def test_selects_first_model_and_preserves_residue_ids(self):
        lines = ["MODEL        1"] + complete_residue(number=42)
        lines += [atom_line(i + 10, atom, "CYS", 42, 5.0, insertion="A")
                  for i, atom in enumerate(baseline.BACKBONE)]
        lines += ["ENDMDL", "MODEL        2"] + complete_residue("GLY", 42, 99.0) + ["ENDMDL"]
        info, target = self.stage(lines)
        self.assertEqual(info["native_sequence"], "AC")
        self.assertEqual(info["selected_model"], "1")
        self.assertEqual(info["residue_mapping"][1]["insertion_code"], "A")
        self.assertEqual(info["residue_mapping"][0]["residue_number"], 42)
        self.assertNotIn("99.000", target.read_text())

    def test_chooses_complete_conformer_without_mixing_alternates(self):
        lines = complete_residue(altloc="A", occupancy=0.2)
        lines += complete_residue(x=7.0, altloc="B", occupancy=0.8)
        info, target = self.stage(lines)
        self.assertEqual(info["residue_mapping"][0]["selected_altloc"], "B")
        self.assertTrue(all(line[16] == " " for line in target.read_text().splitlines()[:4]))
        self.assertTrue(all(float(line[30:38]) == 7.0 for line in target.read_text().splitlines()[:4]))

    def test_rejects_missing_backbone_and_noncanonical_atom_residue(self):
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            self.stage(complete_residue()[:-1])
        with self.assertRaisesRegex(ValueError, "Noncanonical"):
            self.stage(complete_residue("MSE"))

    def test_rejects_duplicate_atoms_and_nonfinite_coordinates(self):
        lines = complete_residue()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.stage(lines + [lines[0]])
        lines[0] = atom_line(1, "N", "ALA", 1, float("nan"))
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            self.stage(lines)

    def test_manifest_rejects_checksum_change_and_duplicate_identity(self):
        source = self.root / "source.pdb"
        source.write_text("\n".join(complete_residue()), encoding="utf-8")
        record = {"structure_id": "fixture", "chain": "A", "pdb_path": "source.pdb",
                  "sha256": baseline.sha256(source), "expected_length": 1}
        manifest = self.root / "manifest.json"
        value = {"schema_version": 1, "purpose": "development_only", "structures": [record]}
        baseline.write_json(manifest, value)
        self.assertEqual(len(baseline.load_manifest(manifest)["structures"]), 1)
        source.write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "checksum"):
            baseline.load_manifest(manifest)
        record["sha256"] = baseline.sha256(source)
        value["structures"].append(dict(record))
        baseline.write_json(manifest, value)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            baseline.load_manifest(manifest)

    def test_existing_output_is_never_overwritten(self):
        marker = self.root / "keep.txt"
        marker.write_text("keep", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            baseline.run(argparse.Namespace(output=self.root))
        self.assertEqual(marker.read_text(), "keep")
        self.assertEqual(len(list(self.root.iterdir())), 1)

    def test_early_failure_is_retained(self):
        output = self.root / "failed"
        with self.assertRaisesRegex(ValueError, "samples"):
            baseline.run(argparse.Namespace(output=output, samples=0, seed=37))
        summary = json.loads((output / "metadata" / "summary.json").read_text())
        self.assertEqual(summary["status"], "failed")
        self.assertEqual(summary["validated_generated_count"], 0)
        self.assertTrue((output / "metadata" / "error.txt").is_file())


class OutputTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.output = self.root / "results"
        (self.output / "seqs").mkdir(parents=True)
        (self.output / "scores").mkdir()
        self.fasta = self.output / "seqs" / "fixture_A.fa"
        self.scores = self.output / "scores" / "fixture_A.npz"
        self.text = (
            ">fixture_A, score=1.2345, global_score=1.2345, model_name=v_48_020, seed=37\nACD\n"
            ">T=0.1, sample=1, score=0.7123, global_score=0.7123, seq_recovery=0.6667\nACE\n"
            ">T=0.1, sample=2, score=0.9000, global_score=0.9000, seq_recovery=1.0000\nACD\n"
        )
        self.fasta.write_text(self.text, encoding="utf-8")
        self.values = np.array([0.712345, 0.9], dtype=np.float32)
        np.savez(self.scores, score=self.values, global_score=self.values)
        self.structure = {"structure_id": "fixture", "chain": "A", "native_sequence": "ACD"}

    def validate(self):
        return baseline.validate_outputs(self.output, self.structure, "vanilla", 2, 37, 0.1, 1.5, self.root)

    def test_uses_full_precision_saved_scores_and_recomputes_recovery(self):
        rows, validation = self.validate()
        self.assertEqual(validation["status"], "passed")
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[1]["native_identity_count"], 2)
        self.assertEqual(rows[1]["native_recovery"], 2 / 3)
        self.assertEqual(rows[1]["score"], float(self.values[0]))
        self.assertEqual(rows[0]["score_source"], "fasta_rounded_4dp")
        self.assertEqual(rows[1]["score_source"], "npz_float32")
        self.assertEqual(len({row["sequence_id"] for row in rows}), 3)

    def test_rejects_incorrect_count_native_and_noncanonical_sample(self):
        for text, expected in [
            (self.text.rsplit(">", 1)[0], "count"),
            (self.text.replace("\nACD\n", "\nAAA\n", 1), "Native FASTA"),
            (self.text.replace("\nACE\n", "\nACX\n"), "canonical"),
        ]:
            with self.subTest(expected=expected):
                self.fasta.write_text(text, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, expected):
                    self.validate()

    def test_rejects_wrong_recovery_and_wrong_sample_number(self):
        for text, expected in [
            (self.text.replace("seq_recovery=0.6667", "seq_recovery=0.3333"), "recovery"),
            (self.text.replace("sample=2", "sample=3"), "sample numbering"),
        ]:
            with self.subTest(expected=expected):
                self.fasta.write_text(text, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, expected):
                    self.validate()

    def test_rejects_nonfinite_misshaped_and_disagreeing_score_archives(self):
        for values, expected in [
            (np.array([float("nan"), 0.9]), "Nonfinite"),
            (np.array([[0.712345, 0.9]]), "shape/type"),
            (np.array([0.8, 0.9]), "disagree"),
        ]:
            with self.subTest(expected=expected):
                np.savez(self.scores, score=values, global_score=values)
                with self.assertRaisesRegex(ValueError, expected):
                    self.validate()

    def test_multiline_fasta_is_supported_but_orphan_sequence_is_rejected(self):
        self.fasta.write_text(self.text.replace("\nACE\n", "\nAC\nE\n"), encoding="utf-8")
        rows, _ = self.validate()
        self.assertEqual(rows[1]["sequence"], "ACE")
        self.fasta.write_text("ACD\n" + self.text, encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "precedes"):
            baseline.read_fasta(self.fasta)


if __name__ == "__main__":
    unittest.main()

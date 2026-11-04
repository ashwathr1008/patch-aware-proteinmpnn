"""Boundary checks for multipatch controls; fixtures are not biological data."""
import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from patches.build_dataset import (build_dataset, controls_for_pair, distinct_permutations,
                                   patch_contacts, select_patches)
from patches.starter import BACKBONE, parse_pdb


def fixture():
    lines = []
    names = "ALA GLY SER VAL LYS THR ILE LEU PHE MET ASP TYR".split()
    for index, name in enumerate(names, 1):
        x = ((index - 1) // 4) * 20 + (index - 1) % 4
        for offset, atom in enumerate(BACKBONE):
            lines.append(f"ATOM  {(index - 1) * 4 + offset + 1:5d} {atom:^4s} {name:3s} A{index:4d}    "
                         f"{x:8.3f}{0.0:8.3f}{0.0:8.3f}{1.0:6.2f}{20.0:6.2f}          {atom[0]:>2s}")
    return "\n".join(lines) + "\n"


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.pdb = self.root / "fixture.pdb"
        self.pdb.write_text(fixture(), encoding="ascii")
        self.manifest = self.root / "manifest.json"
        self.manifest.write_text(json.dumps({"schema_version": 1, "purpose": "patch_compatibility_pilot",
            "structures": [{"structure_id": "FIXTURE", "chain": "A", "pdb_path": self.pdb.name,
                            "sha256": hashlib.sha256(self.pdb.read_bytes()).hexdigest(),
                            "expected_length": 12, "split": "development", "group": "synthetic"}]}))
        self.args = argparse.Namespace(manifest=self.manifest, output=self.root / "output", neighbor_radius=10.0,
            patch_radius=8.0, contact_radius=8.0, min_patch_size=4, patches_per_structure=4,
            permutations_per_patch=5, max_jaccard=0.5, seed=7)

    def test_selection_obeys_membership_overlap_and_reports_shortfall(self):
        residues, _ = parse_pdb(fixture(), "A")
        selected, counts, report = select_patches(residues, min_patch_size=4, max_patches=4)
        self.assertEqual([s["indices"] for s in selected], [list(range(1, 5)), list(range(5, 9)), list(range(9, 13))])
        self.assertEqual(report["patch_shortfall"], 1)
        self.assertEqual(set(counts.values()), {3})
        self.assertGreater(report["duplicate_memberships"], 0)
        selected_again, _, _ = select_patches(residues, min_patch_size=4, max_patches=4)
        self.assertEqual(selected, selected_again)

    def test_full_dataset_has_unique_variants_and_traceable_identity_sources(self):
        source_bytes = self.pdb.read_bytes()
        summary = build_dataset(self.args)
        self.assertEqual(summary["status"], "complete")
        self.assertEqual((summary["patch_count"], summary["pair_count"], summary["pair_shortfall"]), (3, 15, 5))
        payload = json.loads((self.args.output / "pairs.json").read_text())
        per_patch = {}
        for pair in payload["pairs"]:
            self.assertTrue(all(pair["checks"].values()))
            self.assertTrue(pair["controls"]["composition_tie"])
            self.assertTrue(pair["controls"]["geometry_tie"])
            self.assertEqual(Counter(pair["original_sequence"]), Counter(pair["permuted_sequence"]))
            for target, source in zip(pair["patch_indices"], pair["identity_source_indices"]):
                self.assertEqual(pair["permuted_sequence"][target - 1], pair["original_sequence"][source - 1])
            variants = per_patch.setdefault(pair["patch_id"], set())
            self.assertNotIn(pair["permuted_sequence"], variants)
            variants.add(pair["permuted_sequence"])
        self.assertEqual(self.pdb.read_bytes(), source_bytes)
        structure = payload["structures"][0]
        self.assertFalse(Path(structure["pdb_path"]).is_absolute())
        self.assertEqual(hashlib.sha256((self.args.output / structure["pdb_path"]).read_bytes()).hexdigest(), structure["sha256"])
        with self.assertRaises(FileExistsError):
            build_dataset(self.args)

    def test_different_memberships_above_jaccard_cap_are_rejected(self):
        residues, _ = parse_pdb(fixture(), "A")
        line = [replace(residue, coordinates=((float(index), 0.0, 0.0),) * 4)
                for index, residue in enumerate(residues)]
        selected, _, report = select_patches(line, neighbor_radius=3.0, patch_radius=3.0,
                                            min_patch_size=4, max_patches=3, max_jaccard=0.5)
        self.assertEqual(len(selected), 2)
        self.assertEqual(report["overlap_rejections"], 2)
        self.assertEqual(report["patch_shortfall"], 1)

    def test_descriptor_excludes_adjacent_contacts_and_preserves_zero_edge_case(self):
        residues, _ = parse_pdb(fixture(), "A")
        distances, contacts = patch_contacts(residues[:4])
        self.assertEqual(contacts, [(1, 3), (1, 4), (2, 4)])
        self.assertEqual(len(distances), 6)
        control = controls_for_pair("AVKT", "AKVT", [1, 2, 3, 4], contacts)
        self.assertTrue(control["composition_tie"])
        self.assertEqual(control["hydrophobic_fraction_original"], 0)
        self.assertEqual(control["hydrophobic_fraction_permuted"], 1 / 3)
        empty = controls_for_pair("AVKT", "AKVT", [1, 2, 3, 4], [])
        self.assertIsNone(empty["hydrophobic_fraction_delta"])
        self.assertIsNone(empty["hydrophobic_fraction_original"])

    def test_permutations_are_bounded_unique_and_exclude_identity(self):
        output, attempts = distinct_permutations("AAC", 5, 7, max_attempts=100)
        self.assertEqual({assignment for assignment, _ in output}, {tuple("ACA"), tuple("CAA")})
        self.assertEqual(attempts, 100)
        self.assertEqual(output, distinct_permutations("AAC", 5, 7, max_attempts=100)[0])

    def test_no_eligible_patch_is_explicit_failure_without_parameter_relaxation(self):
        self.args.min_patch_size = 8
        with self.assertRaisesRegex(ValueError, "No valid pairs"):
            build_dataset(self.args)
        summary = json.loads((self.args.output / "metadata/summary.json").read_text())
        self.assertEqual(summary["status"], "failed")
        self.assertEqual(summary["pair_count"], 0)
        self.assertIn("No eligible patches", summary["exclusions"][0]["reason"])
        self.assertEqual(summary["structure_reports"][0]["selected_patch_count"], 0)

    def test_checksum_change_and_invalid_parameters_fail_with_metadata(self):
        self.pdb.write_text(fixture() + "REMARK changed\n")
        with self.assertRaisesRegex(ValueError, "checksum"):
            build_dataset(self.args)
        summary = json.loads((self.args.output / "metadata/summary.json").read_text())
        self.assertEqual(summary["status"], "failed")
        self.args.output = self.root / "invalid_parameter"
        self.args.max_jaccard = float("nan")
        with self.assertRaisesRegex(ValueError, "Jaccard"):
            build_dataset(self.args)


if __name__ == "__main__":
    unittest.main()

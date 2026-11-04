"""Scientific bookkeeping and parser edge cases, using synthetic PDB fixtures."""

from collections import Counter
import csv
import json
from pathlib import Path
import tempfile
import unittest

from patches.starter import BACKBONE, build_example, parse_pdb, write_example


def atom_line(serial, atom, residue, number, x, chain="A", icode="", altloc="", occupancy=1.0):
    return ("ATOM  {serial:5d} {atom:^4s}{altloc:1s}{residue:3s} {chain:1s}"
            "{number:4d}{icode:1s}   {x:8.3f}{y:8.3f}{z:8.3f}{occupancy:6.2f}{b:6.2f}          {element:>2s}\n").format(
                serial=serial, atom=atom, altloc=altloc, residue=residue, chain=chain,
                number=number, icode=icode, x=x, y=0.0, z=0.0, occupancy=occupancy,
                b=20.0, element=atom[0])


def pdb_fixture(names=("ALA", "GLY", "SER", "LYS", "VAL", "THR"), positions=(0, 2, 4, 6, 8, 30)):
    lines = []
    for index, (name, position) in enumerate(zip(names, positions), 1):
        for offset, atom in enumerate(BACKBONE):
            lines.append(atom_line((index - 1) * 4 + offset + 1, atom, name, index, position))
    return "".join(lines)


class ParserTests(unittest.TestCase):
    def test_first_model_and_selected_chain_only(self):
        text = "MODEL        1\n" + pdb_fixture() + "ENDMDL\n"
        text += "MODEL        2\n" + pdb_fixture(names=("CYS",) * 6) + "ENDMDL\n"
        residues, diagnostics = parse_pdb(text, "A")
        self.assertEqual("".join(r.aa for r in residues), "AGSKVT")
        self.assertEqual(diagnostics["explicit_model_count"], 2)
        with self.assertRaisesRegex(ValueError, "No standard"):
            parse_pdb(text, "B")
        with self.assertRaisesRegex(ValueError, "exactly one"):
            parse_pdb(text, "AB")

    def test_coherent_altloc_prefers_completeness_then_occupancy(self):
        lines = [atom_line(1, "N", "ALA", 10, 0), atom_line(2, "O", "ALA", 10, 0)]
        lines += [atom_line(3, "CA", "ALA", 10, 1, altloc="A", occupancy=0.6),
                  atom_line(4, "C", "ALA", 10, 1, altloc="A", occupancy=0.4),
                  atom_line(5, "CA", "ALA", 10, 2, altloc="B", occupancy=0.4),
                  atom_line(6, "C", "ALA", 10, 2, altloc="B", occupancy=0.7)]
        residues, _ = parse_pdb("".join(lines), "A")
        self.assertEqual(residues[0].altloc, "B")
        self.assertEqual(residues[0].ca, (2.0, 0.0, 0.0))
        self.assertEqual(residues[0].coordinates[2], (2.0, 0.0, 0.0))
        # High-occupancy CA alone cannot replace a complete conformer.
        lines.append(atom_line(7, "CA", "ALA", 10, 3, altloc="C", occupancy=5.0))
        residues_again, _ = parse_pdb("".join(reversed(lines)), "A")
        self.assertEqual(residues, residues_again)

    def test_numbering_insertions_and_incomplete_residue_are_preserved(self):
        lines = []
        for index, (number, insertion, name) in enumerate(((10, "", "ALA"), (10, "A", "GLY"), (13, "", "SER"))):
            for atom_index, atom in enumerate(BACKBONE):
                if index == 2 and atom == "O":
                    continue
                lines.append(atom_line(index * 4 + atom_index + 1, atom, name, number, index * 2, icode=insertion))
        residues, diagnostics = parse_pdb("".join(lines), "A")
        self.assertEqual([(r.index, r.resseq, r.icode) for r in residues], [(1, 10, ""), (2, 10, "A"), (3, 13, "")])
        self.assertEqual(diagnostics["incomplete_backbone_indices"], [3])
        self.assertEqual(diagnostics["insertion_code_indices"], [2])
        self.assertEqual(diagnostics["numbering_gaps"][0]["missing_number_start"], 11)
        self.assertEqual(diagnostics["numbering_gaps"][0]["missing_number_end"], 12)
        self.assertTrue(diagnostics["requires_mapping_review"])

    def test_nonfinite_coordinates_and_ambiguous_identity_rejected(self):
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            parse_pdb(atom_line(1, "CA", "ALA", 1, float("nan")), "A")
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            parse_pdb(atom_line(1, "CA", "ALA", 1, 0) + atom_line(2, "N", "GLY", 1, 0), "A")


class ExampleTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.pdb = self.root / "fixture.pdb"
        self.pdb.write_text(pdb_fixture(), encoding="ascii")

    def test_composition_geometry_membership_and_outside_invariants(self):
        before_bytes = self.pdb.read_bytes()
        manifest = build_example(self.pdb, "A", seed=7)
        self.assertEqual(manifest, build_example(self.pdb, "A", seed=7))
        self.assertEqual(self.pdb.read_bytes(), before_bytes)
        self.assertTrue(all(manifest["checks"].values()))
        self.assertEqual(manifest["selection"]["patch_observed_sequence_indices"], [1, 2, 3, 4, 5])
        self.assertEqual(manifest["geometry"]["original_sha256"], manifest["geometry"]["permuted_sha256"])
        patch = [row for row in manifest["residues"] if row["in_patch"]]
        self.assertEqual(Counter(row["original_aa"] for row in patch), Counter(row["permuted_aa"] for row in patch))
        self.assertNotEqual([row["original_aa"] for row in patch], [row["permuted_aa"] for row in patch])
        for row, residue in zip(manifest["residues"], parse_pdb(before_bytes.decode(), "A")[0]):
            self.assertEqual(row["backbone_coordinates"], {atom: list(xyz) for atom, xyz in zip(BACKBONE, residue.coordinates)})
            if not row["in_patch"]:
                self.assertEqual(row["original_aa"], row["permuted_aa"])

    def test_incomplete_backbone_retained_in_fasta_but_excluded_from_patch(self):
        text = pdb_fixture() + "".join(atom_line(25 + i, atom, "CYS", 7, 3) for i, atom in enumerate(BACKBONE[:-1]))
        self.pdb.write_text(text, encoding="ascii")
        manifest = build_example(self.pdb, "A")
        self.assertEqual(manifest["sequences"]["original_observed_chain"], "AGSKVTC")
        self.assertEqual(manifest["sequences"]["permuted_observed_chain"][-1], "C")
        self.assertFalse(manifest["residues"][-1]["in_patch"])
        self.assertEqual(manifest["mapping"]["incomplete_backbone_indices"], [7])

    def test_homogeneous_and_too_small_inputs_fail(self):
        self.pdb.write_text(pdb_fixture(names=("ALA",) * 6), encoding="ascii")
        with self.assertRaisesRegex(ValueError, "No heterogeneous patch"):
            build_example(self.pdb, "A")
        self.pdb.write_text(pdb_fixture(names=("ALA",), positions=(0,)), encoding="ascii")
        with self.assertRaisesRegex(ValueError, "Too few complete"):
            build_example(self.pdb, "A")

    def test_parameter_validation(self):
        for radius in (-1.0, 0.0, float("nan"), float("inf")):
            with self.subTest(radius=radius), self.assertRaisesRegex(ValueError, "finite and positive"):
                build_example(self.pdb, "A", patch_radius=radius)
        with self.assertRaisesRegex(ValueError, "at least two"):
            build_example(self.pdb, "A", min_patch_size=1)

    def test_files_are_consistent_and_existing_outputs_are_protected(self):
        manifest = build_example(self.pdb, "A")
        output = self.root / "output"
        write_example(manifest, output)
        self.assertEqual(json.loads((output / "manifest.json").read_text()), manifest)
        with (output / "residues.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), len(manifest["residues"]))
        for label in ("original", "permuted"):
            fasta = (output / (label + ".fasta")).read_text().splitlines()
            self.assertEqual(fasta[1], manifest["sequences"][label + "_observed_chain"])
        before = {path.name: path.read_bytes() for path in output.iterdir()}
        with self.assertRaisesRegex(ValueError, "already exists"):
            write_example(manifest, output)
        self.assertEqual(before, {path.name: path.read_bytes() for path in output.iterdir()})


if __name__ == "__main__":
    unittest.main()

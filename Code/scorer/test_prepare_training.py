"""Small fixture tests for filtering, measured swaps and split-before-pair preparation."""
import csv
import io
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scorer.prepare_training import stage1, map_structures, group_parents, stage3, MEMBER, valid_mutation_name
from scorer.data import load_dataset


def pdb_text(sequence):
    names = dict(zip("ACDEGW", ("ALA", "CYS", "ASP", "GLU", "GLY", "TRP")))
    lines = []
    for i, aa in enumerate(sequence, 1):
        for a, atom in enumerate(("N", "CA", "C", "O")):
            lines.append(f"ATOM  {len(lines)+1:5d} {atom:^4s} {names[aa]} A{i:4d}    {float(i*4+a):8.3f}{0.:8.3f}{0.:8.3f}{1.:6.2f}{0.:6.2f}           C")
    return "\n".join(lines + ["END", ""])


class PreparationTests(unittest.TestCase):
    def test_annotations_reject_undeclared_mutations_and_indels(self):
        self.assertTrue(valid_mutation_name("p_A1C_C2A", "p", "ACDE", "CADE"))
        self.assertTrue(valid_mutation_name("p_dmutv5_1A:2C_A1C:C2A", "p", "ACDE", "CADE"))
        self.assertTrue(valid_mutation_name("p_hnet0_A1C:C2A", "p", "ACDE", "CADE"))
        self.assertFalse(valid_mutation_name("p_hnet0_A1C:G2A", "p", "ACDE", "CADE"))
        self.assertFalse(valid_mutation_name("p_dmutv5_1G:2C_A1C:C2A", "p", "ACDE", "CADE"))
        self.assertFalse(valid_mutation_name("p_A1C", "p", "ACDE", "CADE"))
        self.assertFalse(valid_mutation_name("p_insG1", "p", "ACDE", "GACDE"))

    def test_full_preparation_and_loader(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw, output = root / "raw", root / "output"
            raw.mkdir()
            output.mkdir()
            rows = []
            fields = ["name", "WT_name", "aa_seq", "mut_type", "dG_ML", "ddG_ML", "deltaG_95CI_low", "deltaG_95CI_high"]
            def add(name, parent, seq, wt, dg, ddg=0):
                rows.append(dict(zip(fields, (name, parent, seq, "wt" if wt else "mut", dg, ddg, 0, 6))))
            for parent, seq in (("p", "ACDE"), ("q", "WWWW"), ("r", "GGGG")):
                add(parent, parent, seq, True, 5)
            add("p_dmutv5_1A:2C_A1F:C2G", "p", "FGDE", False, 2)
            add("p_dmutv5_1A:2C_A1F:C2G", "p", "FGDE", False, 4)  # measured replicate mean=3
            add("p_dmutv5_1A:2C_A1G:C2F", "p", "GFDE", False, 1)
            add("p_A1C_C2A", "p", "CADE", False, 4)
            add("p_A1F_C2A", "p", "FADE", False, 3)
            add("p_C2F", "p", "AFDE", False, 2)
            add("p_A1V", "p", "VCDE", False, ">5")
            add("p_A1L", "p", "LCDE", False, 3, "-")
            add("p_A1C", "p", "CCDF", False, 3)  # unreported fourth-site mutation
            text = io.StringIO()
            writer = csv.DictWriter(text, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            with zipfile.ZipFile(raw / "Processed_K50_dG_datasets.zip", "w") as archive:
                archive.writestr(MEMBER, text.getvalue())
            with zipfile.ZipFile(raw / "AlphaFold_model_PDBs.zip", "w") as archive:
                for parent, seq in (("p", "ACDE"), ("q", "WWWW"), ("r", "GGGG")):
                    archive.writestr(parent + ".pdb", pdb_text(seq))
            connection = sqlite3.connect(":memory:")
            refs = stage1(raw, output, connection)
            report = json.loads((output / "stage1_labels.json").read_text())
            self.assertEqual(report["counts"]["excluded_filtered_censored_or_nonfinite"], 2)
            self.assertEqual(connection.execute("SELECT dg FROM measurements WHERE sequence='FGDE'").fetchone()[0], 3)
            parents, _ = map_structures(raw, output, refs)
            homology = group_parents(parents, connection)
            self.assertEqual(set(homology["parent_counts"].values()), {1})
            stage3(output, parents, homology, connection)
            manifest, _, pairs = load_dataset(output / "stability_pairs.json")
            swaps = [p for p in pairs if p["pair_kind"] == "measured_swap"]
            self.assertEqual(len(swaps), 2)
            fg = next(p for p in swaps if p["reference_sequence"] == "FGDE")
            self.assertEqual(fg["target"], 2)  # reference DG3 minus variant DG1
            self.assertTrue(any(p["reference_sequence"] == "AFDE" and p["variant_sequence"] == "FADE" for p in swaps))
            self.assertEqual(len(manifest["pairs"]), 7)
            self.assertEqual(sum(p["is_swap"] for p in pairs), 3)
            connection.close()

    def test_measured_sequence_overlap_merges_before_splitting(self):
        parents = {p: {"reference_sequence": s} for p, s in (("a", "ACDE"), ("b", "CDEA"), ("c", "WWWW"), ("d", "GGGG"))}
        connection = sqlite3.connect(":memory:")
        connection.execute("CREATE TABLE measurements(parent TEXT,sequence TEXT)")
        connection.executemany("INSERT INTO measurements VALUES (?,?)", [("a", "CADE"), ("b", "CADE")])
        audit = group_parents(parents, connection, threshold=0.99)
        self.assertEqual(parents["a"]["split"], parents["b"]["split"])
        self.assertEqual(parents["a"]["homology_group"], parents["b"]["homology_group"])
        self.assertEqual(len(audit["groups"]), 3)
        connection.close()


if __name__ == "__main__":
    unittest.main()

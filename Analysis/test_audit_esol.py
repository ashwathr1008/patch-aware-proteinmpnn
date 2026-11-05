"""Check eSOL parsing, mapping, exclusion precedence and the empty-input guards."""
import unittest

import audit_esol as audit


def esol_row(b_number, solubility, jw=None, gene="geneA"):
    return {"JW_ID": jw if jw is not None else "JW" + b_number[1:],
            "B number": b_number, "Gene name K-12": gene,
            audit.SOLUBILITY_COLUMN: solubility}


def uniprot_row(accession, locus, pdb=""):
    return {"Entry": accession, "Gene Names (primary)": "g",
            "Gene Names (ordered locus)": locus, "PDB": pdb}


class NumberTest(unittest.TestCase):
    def test_ordinary_values_parse(self):
        self.assertEqual(audit.number("42.5"), 42.5)
        self.assertEqual(audit.number(" 7 "), 7.0)

    def test_blank_and_junk_return_nothing(self):
        for value in ("", "   ", None, "-", "NA"):
            self.assertIsNone(audit.number(value))

    def test_non_finite_values_are_rejected(self):
        """float() accepts these, and they would break the JSON write after the CSV is out."""
        for value in ("nan", "inf", "-inf", "NaN", "Infinity"):
            self.assertIsNone(audit.number(value))


class DistributionTest(unittest.TestCase):
    def test_an_empty_input_returns_nothing_rather_than_raising(self):
        self.assertIsNone(audit.distribution([]))

    def test_summary_statistics_are_correct(self):
        result = audit.distribution([10.0, 20.0, 30.0, 40.0])
        self.assertEqual(result["count"], 4)
        self.assertAlmostEqual(result["median"], 25.0)
        self.assertAlmostEqual(result["min"], 10.0)
        self.assertAlmostEqual(result["max"], 40.0)

    def test_the_band_fractions_count_the_right_records(self):
        result = audit.distribution([5.0, 25.0, 50.0, 80.0, 95.0])
        self.assertAlmostEqual(result["fraction_below_30"], 2 / 5)
        self.assertAlmostEqual(result["fraction_above_70"], 2 / 5)

    def test_a_single_value_does_not_break_the_quartiles(self):
        result = audit.distribution([12.0])
        self.assertEqual(result["count"], 1)
        self.assertAlmostEqual(result["p25"], 12.0)


class LocusIndexTest(unittest.TestCase):
    def test_every_locus_token_is_indexed(self):
        index = audit.build_locus_index([uniprot_row("P1", "b0001 JW0001")])
        self.assertIn("b0001", index)
        self.assertIn("JW0001", index)

    def test_entries_sharing_a_token_are_collected(self):
        index = audit.build_locus_index([uniprot_row("P1", "b0001"),
                                         uniprot_row("P2", "b0001")])
        self.assertEqual(len(index["b0001"]), 2)


class AuditTest(unittest.TestCase):
    def run_audit(self, esol, uniprot, maximum=100.0):
        return audit.audit(esol, uniprot, maximum)

    def test_a_complete_record_is_usable(self):
        records, counts = self.run_audit([esol_row("b0001", "55")],
                                         [uniprot_row("P1", "b0001", "1ABC;")])
        self.assertEqual(counts["usable"], 1)
        self.assertEqual(records[0]["excluded_reason"], "")
        self.assertEqual(records[0]["pdb_entry_count"], 1)

    def test_a_missing_measurement_is_excluded_before_mapping(self):
        _, counts = self.run_audit([esol_row("b0001", "")],
                                   [uniprot_row("P1", "b0001", "1ABC;")])
        self.assertEqual(counts["missing_solubility"], 1)
        self.assertEqual(counts["usable"], 0)

    def test_an_unmapped_identifier_is_counted(self):
        _, counts = self.run_audit([esol_row("b9999", "55")],
                                   [uniprot_row("P1", "b0001", "1ABC;")])
        self.assertEqual(counts["unmapped"], 1)

    def test_the_jw_identifier_is_a_fallback(self):
        records, counts = self.run_audit([esol_row("bZZZZ", "55", jw="JW0001")],
                                         [uniprot_row("P1", "JW0001", "1ABC;")])
        self.assertEqual(counts["usable"], 1)
        self.assertEqual(records[0]["uniprot_accession"], "P1")

    def test_exclusion_categories_are_disjoint_and_ordered(self):
        """An over-100% record with no structure counts once, under the solubility rule."""
        _, counts = self.run_audit([esol_row("b0001", "147")],
                                   [uniprot_row("P1", "b0001", "")])
        self.assertEqual(counts["above_maximum_solubility"], 1)
        self.assertEqual(counts["no_structure"], 0)

    def test_the_usable_count_reconciles_with_the_category_totals(self):
        esol = [esol_row("b0001", "55"), esol_row("b0002", "147"),
                esol_row("b0003", "40"), esol_row("b0004", "")]
        uniprot = [uniprot_row("P1", "b0001", "1ABC;"), uniprot_row("P2", "b0002", "2ABC;"),
                   uniprot_row("P3", "b0003", ""), uniprot_row("P4", "b0004", "")]
        _, counts = self.run_audit(esol, uniprot)
        mapped = len(esol) - counts["missing_solubility"] - counts["unmapped"]
        self.assertEqual(mapped, counts["above_maximum_solubility"]
                         + counts["no_structure"] + counts["usable"])

    def test_ambiguous_mappings_are_counted_but_still_scored(self):
        _, counts = self.run_audit([esol_row("b0001", "55")],
                                   [uniprot_row("P1", "b0001", "1ABC;"),
                                    uniprot_row("P2", "b0001", "2ABC;")])
        self.assertEqual(counts["ambiguous_mapping"], 1)
        self.assertEqual(counts["usable"], 1)


class SummaryTest(unittest.TestCase):
    def test_an_all_structured_set_leaves_an_empty_subset_without_raising(self):
        """Guards the case where no mapped protein lacks a structure."""
        records, counts = audit.audit([esol_row("b0001", "55")],
                                      [uniprot_row("P1", "b0001", "1ABC;")], 100.0)
        summary = audit.summarize(records, counts, 100.0, ["1UBQ"])
        self.assertIsNone(summary["solubility_distribution"]["without_structure"])
        self.assertIsNotNone(summary["solubility_distribution"]["with_structure"])

    def test_a_set_with_no_measurements_summarizes_to_nothing(self):
        records, counts = audit.audit([esol_row("b0001", "")], [uniprot_row("P1", "b0001")],
                                      100.0)
        summary = audit.summarize(records, counts, 100.0, [])
        self.assertIsNone(summary["solubility_distribution"]["all_measured"])
        self.assertEqual(summary["coverage"]["records_with_measurement"], 0)

    def test_pilot_overlap_is_recorded_as_zero(self):
        records, counts = audit.audit([esol_row("b0001", "55")],
                                      [uniprot_row("P1", "b0001", "1ABC;")], 100.0)
        summary = audit.summarize(records, counts, 100.0, ["1UBQ", "1CRN"])
        self.assertEqual(summary["pilot_overlap"]["overlapping_records"], 0)
        self.assertEqual(summary["pilot_overlap"]["pilot_structures"], ["1UBQ", "1CRN"])


if __name__ == "__main__":
    unittest.main()

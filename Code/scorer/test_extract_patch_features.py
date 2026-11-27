"""Check contrast eligibility, orientation and order averaging in the patch extractor.

The eligibility rule is the part worth asserting. `check_pair` is what stops a mis-mapped
site from being featurized silently: if `site_i` resolved to the wrong residue, the two
arrangements would still differ at two positions and still have matching composition, but
they would not be *these* two positions, and nothing downstream would notice. The tests below
pin each way that can fail.

Orientation is the other silent failure. The contrast table canonicalises A and B
alphabetically, so `row_a` must be the arrangement carrying `aa_i` at `site_i`, matching the
`stability_forward` the measured difference was computed from. A reversal here would flip the
sign of every label without changing any other observable.
"""
import csv
import json
import tempfile
import unittest
from pathlib import Path

import torch

import extract_patch_features as extract
import features
from test_features import StubModel, make_features


NATIVE = "ACDEFGHIKL"


class TestArrangement(unittest.TestCase):

    def test_places_each_identity_at_its_own_site(self):
        self.assertEqual(extract.arrangement(NATIVE, 2, 5, "W", "Y"), "ACWEFYHIKL")

    def test_reverse_exchanges_only_those_two(self):
        a = extract.arrangement(NATIVE, 2, 5, "W", "Y")
        b = extract.arrangement(NATIVE, 2, 5, "Y", "W")
        self.assertEqual([k for k, (x, y) in enumerate(zip(a, b)) if x != y], [2, 5])


class TestCheckPair(unittest.TestCase):

    def valid(self, pos_i=2, pos_j=5, aa_i="W", aa_j="Y"):
        return (extract.arrangement(NATIVE, pos_i, pos_j, aa_i, aa_j),
                extract.arrangement(NATIVE, pos_i, pos_j, aa_j, aa_i))

    def test_accepts_a_genuine_two_site_exchange(self):
        a, b = self.valid()
        self.assertIsNone(extract.check_pair(NATIVE, a, b, 2, 5))

    def test_refuses_when_the_recorded_sites_are_not_the_differing_ones(self):
        a, b = self.valid()
        # Mapping slipped by one: the sequences are fine, the claimed sites are not.
        self.assertIn("not the two swapped sites",
                      extract.check_pair(NATIVE, a, b, 3, 5))

    def test_refuses_identical_identities(self):
        a, b = self.valid(aa_i="W", aa_j="W")
        self.assertIsNotNone(extract.check_pair(NATIVE, a, b, 2, 5))

    def test_refuses_a_third_changed_position(self):
        a, b = self.valid()
        b = b[:8] + ("P" if b[8] != "P" else "G") + b[9:]
        self.assertIn("not the two swapped sites",
                      extract.check_pair(NATIVE, a, b, 2, 5))

    def test_refuses_unexchanged_identities(self):
        a = extract.arrangement(NATIVE, 2, 5, "W", "Y")
        b = extract.arrangement(NATIVE, 2, 5, "M", "P")
        self.assertIn("not exchanged", extract.check_pair(NATIVE, a, b, 2, 5))

    def test_refuses_a_length_mismatch(self):
        a, b = self.valid()
        self.assertIn("length mismatch", extract.check_pair(NATIVE, a[:-1], b, 2, 5))


class TestLoadContrasts(unittest.TestCase):

    def write(self, rows):
        handle = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, newline="")
        writer = csv.DictWriter(handle, fieldnames=[
            "domain", "site_i", "site_j", "aa_i", "aa_j", "stability_forward",
            "stability_reverse", "difference", "uncertainty", "resolved"])
        writer.writeheader()
        writer.writerows(rows)
        handle.close()
        return Path(handle.name)

    def row(self, **kw):
        base = {"domain": "1ABC", "site_i": 3, "site_j": 6, "aa_i": "A", "aa_j": "W",
                "stability_forward": "4.0", "stability_reverse": "3.0",
                "difference": "1.0", "uncertainty": "0.1", "resolved": "True"}
        base.update(kw)
        return base

    def test_groups_by_site_pair_and_parses_types(self):
        path = self.write([self.row(), self.row(aa_i="C", aa_j="D"),
                           self.row(site_j=7)])
        grouped, total = extract.load_contrasts(path)
        self.assertEqual(total, 3)
        self.assertEqual(sorted(grouped), [("1ABC", 3, 6), ("1ABC", 3, 7)])
        self.assertEqual(len(grouped[("1ABC", 3, 6)]), 2)
        first = grouped[("1ABC", 3, 6)][0]
        self.assertIsInstance(first["site_i"], int)
        self.assertEqual(first["difference"], 1.0)
        self.assertIs(first["resolved"], True)

    def test_resolved_false_is_parsed_as_false(self):
        path = self.write([self.row(resolved="False")])
        grouped, _ = extract.load_contrasts(path)
        self.assertIs(grouped[("1ABC", 3, 6)][0]["resolved"], False)

    def test_difference_sign_is_preserved_not_normalized(self):
        path = self.write([self.row(difference="-2.5")])
        grouped, _ = extract.load_contrasts(path)
        self.assertEqual(grouped[("1ABC", 3, 6)][0]["difference"], -2.5)


class TestHomology(unittest.TestCase):

    def test_maps_domains_to_family_and_split(self):
        handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        json.dump({"membership": {"1ABC": "1ABC", "2DEF": "1ABC", "3GHI": "3GHI"},
                   "splits": {"assignment": {"train": ["1ABC", "2DEF"],
                                             "test": ["3GHI"]}}}, handle)
        handle.close()
        membership, split_of = extract.load_homology(Path(handle.name))
        self.assertEqual(membership["2DEF"], "1ABC")
        self.assertEqual(split_of["2DEF"], "train")
        self.assertEqual(split_of["3GHI"], "test")

    def test_absent_file_yields_empty_maps(self):
        self.assertEqual(extract.load_homology(None), ({}, {}))


class TestMeanLogprobs(unittest.TestCase):
    """One seed must pass values straight through; several must average in log space."""

    def setUp(self):
        self.entry = {"features": make_features(len(NATIVE))}
        self.sequences = [extract.arrangement(NATIVE, 2, 5, "W", "Y")]
        self.positions = [2, 5]

    def test_single_seed_matches_the_underlying_call(self):
        model = StubModel()
        got = extract.mean_logprobs(model, self.entry, self.sequences, self.positions,
                                    torch, [137], 64, "1ABC")
        direct = features.conditional_logprobs_at(
            StubModel(), self.entry["features"], self.sequences, self.positions,
            torch, seed=137, batch_size=64, tag="1ABC")
        for position in self.positions:
            self.assertEqual(got[0][position], direct[0][position])

    def test_two_seeds_average_elementwise(self):
        a = features.conditional_logprobs_at(
            StubModel(), self.entry["features"], self.sequences, self.positions,
            torch, seed=1, batch_size=64, tag="1ABC")
        b = features.conditional_logprobs_at(
            StubModel(), self.entry["features"], self.sequences, self.positions,
            torch, seed=2, batch_size=64, tag="1ABC")
        got = extract.mean_logprobs(StubModel(), self.entry, self.sequences,
                                    self.positions, torch, [1, 2], 64, "1ABC")
        for position in self.positions:
            expected = [(x + y) / 2 for x, y in zip(a[0][position], b[0][position])]
            for value, want in zip(got[0][position], expected):
                self.assertAlmostEqual(value, want, places=12)

    def test_averaging_does_not_mutate_a_single_seed_result(self):
        got = extract.mean_logprobs(StubModel(), self.entry, self.sequences,
                                    self.positions, torch, [137], 64, "1ABC")
        self.assertEqual(len(got[0][2]), len(features.ALPHABET))


class TestPositionsArePartOfFeatureIdentity(unittest.TestCase):
    """Why the extractor's feature cache is scoped to one site pair.

    The same full sequence belongs to several site pairs of one domain: the native sequence
    belongs to every site pair whose measured identities happen to match the wild type, and
    one real domain had a single arrangement claimed by seven site pairs. A cache keyed by
    sequence alone reused those rows and returned log-probabilities read at the wrong
    residues -- valid-looking features for the wrong patch, which no other check catches.
    """

    def test_one_sequence_gives_different_features_at_different_positions(self):
        entry = {"features": make_features(len(NATIVE))}
        sequence = NATIVE
        here, there = [2, 5], [3, 8]
        a = extract.mean_logprobs(StubModel(), entry, [sequence], here,
                                  torch, [137], 64, "1ABC")
        b = extract.mean_logprobs(StubModel(), entry, [sequence], there,
                                  torch, [137], 64, "1ABC")
        va = features.patch_feature_vector(a[0], sequence, here)
        vb = features.patch_feature_vector(b[0], sequence, there)
        self.assertNotEqual(va, vb)

    def test_identity_block_follows_the_positions_not_the_sequence(self):
        entry = {"features": make_features(len(NATIVE))}
        size = len(features.ALPHABET)
        for positions in ([2, 5], [3, 8]):
            logprobs = extract.mean_logprobs(StubModel(), entry, [NATIVE], positions,
                                             torch, [137], 64, "1ABC")
            vector = features.patch_feature_vector(logprobs[0], NATIVE, positions)
            placed = features.ALPHABET[vector[size:2 * size].index(1.0)]
            self.assertEqual(placed, NATIVE[positions[0]])


class TestFeatureVectorOrientation(unittest.TestCase):
    """row_a must carry aa_i at site_i, matching stability_forward."""

    def test_a_and_b_vectors_differ_in_their_identity_blocks(self):
        entry = {"features": make_features(len(NATIVE))}
        positions = [2, 5]
        a = extract.arrangement(NATIVE, 2, 5, "A", "W")
        b = extract.arrangement(NATIVE, 2, 5, "W", "A")
        logprobs = extract.mean_logprobs(StubModel(), entry, [a, b], positions,
                                         torch, [137], 64, "1ABC")
        va = features.patch_feature_vector(logprobs[0], a, positions)
        vb = features.patch_feature_vector(logprobs[1], b, positions)
        self.assertEqual(len(va), extract.PATCH_SIZE * 2 * len(features.ALPHABET))
        self.assertNotEqual(va, vb)
        # The identity one-hot for member 0 sits straight after its log-probability block.
        size = len(features.ALPHABET)
        self.assertEqual(va[size:2 * size].index(1.0), features.ALPHABET.index("A"))
        self.assertEqual(vb[size:2 * size].index(1.0), features.ALPHABET.index("W"))


if __name__ == "__main__":
    unittest.main()


class TestNearestNeighbours(unittest.TestCase):
    """The region must come from the backbone alone, identically for A and B."""

    def line(self, n, spacing=1.0):
        """n residues on a straight line, so distances are exactly known."""
        return [[(0.0, 0.0, 0.0), (i * spacing, 0.0, 0.0),
                 (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)] for i in range(n)]

    def test_picks_the_closest_by_minimum_distance_to_either_site(self):
        # Sites at 0 and 9; the nearest non-patch residues are 1 and 8, then 2 and 7.
        got = extract.nearest_neighbours(self.line(10), 0, 9, 4)
        self.assertEqual(sorted(got), [1, 2, 7, 8])

    def test_excludes_the_patch_positions_themselves(self):
        got = extract.nearest_neighbours(self.line(10), 0, 9, 8)
        self.assertNotIn(0, got)
        self.assertNotIn(9, got)

    def test_ties_break_by_sequence_index(self):
        # Residues 2 and 4 are both distance 1 from the site at 3; 2 must come first.
        got = extract.nearest_neighbours(self.line(7), 3, 6, 1)
        self.assertEqual(got, [2])

    def test_count_zero_returns_nothing(self):
        self.assertEqual(extract.nearest_neighbours(self.line(10), 0, 9, 0), [])

    def test_short_chain_returns_all_it_has(self):
        got = extract.nearest_neighbours(self.line(5), 0, 4, 8)
        self.assertEqual(sorted(got), [1, 2, 3])

    def test_region_is_deterministic_across_calls(self):
        coords = self.line(30)
        first = extract.nearest_neighbours(coords, 5, 20, 8)
        second = extract.nearest_neighbours(coords, 5, 20, 8)
        self.assertEqual(first, second)

    def test_neighbour_identities_are_shared_by_both_arrangements(self):
        """A and B differ only at the patch, so the neighbour block is identical in both --
        it can still change q, because the head that reads it is nonlinear."""
        coords = self.line(12)
        neighbours = extract.nearest_neighbours(coords, 2, 9, 4)
        a = extract.arrangement(NATIVE + "AA", 2, 9, "W", "Y")
        b = extract.arrangement(NATIVE + "AA", 2, 9, "Y", "W")
        self.assertEqual([a[n] for n in neighbours], [b[n] for n in neighbours])

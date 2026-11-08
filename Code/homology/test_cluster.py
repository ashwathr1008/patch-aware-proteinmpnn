"""Check that clustering separates unrelated short sequences and merges genuine relatives."""
import random
from pathlib import Path
import os
import subprocess
import sys
import unittest

import cluster


def mutate(sequence, fraction, seed):
    """Change a fraction of positions, producing a genuine relative of the input."""
    rng = random.Random(seed)
    letters = list(sequence)
    positions = rng.sample(range(len(letters)), int(len(letters) * fraction))
    for position in positions:
        letters[position] = rng.choice("ACDEFGHIKLMNPQRSTVWY")
    return "".join(letters)


def random_sequence(length, seed):
    rng = random.Random(seed)
    return "".join(rng.choice("ACDEFGHIKLMNPQRSTVWY") for _ in range(length))


# The pilot structures: seven small domains, none homologous to another.
PILOT = {
    "1UBQ": "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",
    "2CI2": "NLKTEWPELVGKSVEEAKKVILQDKPEAQIIVLPVGTIVTMEYRIDRVRLFVDKLDNIAQVPRVG",
    "1SHG": "KELVLALYDYQEKSPREVTMKKGDILTLLNSTNKDWWKVEVNDRQGFVPAAYVKKLD",
    "1ENH": "RPRTAFSSEQLARLKREFNENRYLTERRRQQLSSELGLNEAQIKIWFQNKRAKIK",
    "1CRN": "TTCCPSIVARSNFNVCRLPGTPEAICATYTGCIIIPGATCPGDYAN",
    "1VII": "MLSDEDFKAVFGMTRSAFANLPLWKQQNLKKEKGLF",
    "1L2Y": "NLYIQWLKDGGPSSGRPPPS",
}


class IdentityTest(unittest.TestCase):
    def test_a_sequence_matches_itself_completely(self):
        self.assertAlmostEqual(cluster.identity(PILOT["1UBQ"], PILOT["1UBQ"]), 1.0)

    def test_identity_is_symmetric(self):
        a, b = PILOT["1UBQ"], PILOT["2CI2"]
        self.assertAlmostEqual(cluster.identity(a, b), cluster.identity(b, a), places=6)

    def test_an_empty_sequence_is_rejected(self):
        with self.assertRaises(ValueError):
            cluster.identity("", PILOT["1UBQ"])

    def test_short_unrelated_sequences_reach_misleading_identity(self):
        """The failure this module exists to avoid: chance identity on small domains."""
        value = cluster.identity(PILOT["1UBQ"], PILOT["1VII"])
        self.assertGreater(value, 0.30)


class NullTest(unittest.TestCase):
    def test_the_null_is_reproducible(self):
        first = cluster.null_statistics(PILOT["1UBQ"], PILOT["2CI2"], shuffles=40, seed=3)
        second = cluster.null_statistics(PILOT["1UBQ"], PILOT["2CI2"], shuffles=40, seed=3)
        self.assertEqual(first, second)

    def test_too_few_shuffles_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least two"):
            cluster.null_statistics(PILOT["1UBQ"], PILOT["2CI2"], shuffles=1)

    def test_an_identical_pair_stands_far_above_its_null(self):
        record = cluster.relatedness(PILOT["1SHG"], PILOT["1SHG"], shuffles=60)
        self.assertGreater(record["z"], 8.0)


class RelatednessTest(unittest.TestCase):
    def test_a_close_relative_is_detected(self):
        relative = mutate(PILOT["1UBQ"], 0.20, seed=1)
        record = cluster.relatedness(PILOT["1UBQ"], relative, shuffles=80)
        self.assertGreater(record["z"], cluster.DEFAULT_Z)
        self.assertGreater(record["identity"], cluster.DEFAULT_IDENTITY)

    def test_chance_identity_does_not_survive_the_null(self):
        """1UBQ and 1VII exceed 30% identity but are unrelated."""
        record = cluster.relatedness(PILOT["1UBQ"], PILOT["1VII"], shuffles=200)
        self.assertGreater(record["identity"], 0.30)
        self.assertLess(record["z"], cluster.DEFAULT_Z)


class ClusterTest(unittest.TestCase):
    def test_the_seven_pilot_structures_are_all_independent(self):
        result = cluster.cluster(PILOT, shuffles=200)
        self.assertEqual(result["cluster_count"], 7)
        self.assertEqual(result["related_pairs"], [])

    def test_a_family_merges_into_one_cluster(self):
        family = {"base": PILOT["1UBQ"],
                  "rel1": mutate(PILOT["1UBQ"], 0.15, seed=5),
                  "rel2": mutate(PILOT["1UBQ"], 0.25, seed=6),
                  "other": PILOT["1SHG"]}
        result = cluster.cluster(family, shuffles=120)
        sizes = sorted(len(v) for v in result["clusters"].values())
        self.assertEqual(sizes, [1, 3])

    def test_single_linkage_merges_through_an_intermediate(self):
        names = ["a", "b", "c"]
        groups = cluster.single_linkage(names, [("a", "b"), ("b", "c")])
        self.assertEqual(len(groups), 1)
        self.assertEqual(sorted(next(iter(groups.values()))), ["a", "b", "c"])

    def test_clusters_do_not_depend_on_input_order(self):
        forward = cluster.cluster(PILOT, shuffles=60)
        reversed_input = {k: PILOT[k] for k in reversed(list(PILOT))}
        backward = cluster.cluster(reversed_input, shuffles=60)
        self.assertEqual(forward["clusters"], backward["clusters"])

    def test_duplicate_identifiers_are_impossible_to_pass_silently(self):
        with self.assertRaises(ValueError):
            cluster.related_pairs({"a": ""})

    def test_screened_pairs_are_recorded_and_never_counted_as_related(self):
        result = cluster.cluster(PILOT, shuffles=40)
        screened = [e for e in result["evidence"] if e["screened"]]
        self.assertTrue(screened)
        self.assertTrue(all(not e["related"] for e in screened))


class SplitTest(unittest.TestCase):
    def setUp(self):
        self.clusters = {"c1": ["a", "b", "c"], "c2": ["d", "e"], "c3": ["f"], "c4": ["g"]}

    def test_every_sequence_lands_in_exactly_one_split(self):
        assignment = cluster.assign_splits(self.clusters, {"dev": 0.5, "test": 0.5})
        placed = sorted(m for members in assignment.values() for m in members)
        self.assertEqual(placed, list("abcdefg"))

    def test_no_cluster_is_divided(self):
        assignment = cluster.assign_splits(self.clusters, {"dev": 0.6, "test": 0.4})
        self.assertTrue(cluster.verify_split(assignment, self.clusters))

    def test_a_divided_cluster_is_caught(self):
        broken = {"dev": ["a", "b"], "test": ["c", "d", "e", "f", "g"]}
        with self.assertRaisesRegex(ValueError, "split across"):
            cluster.verify_split(broken, self.clusters)

    def test_an_unassigned_sequence_is_caught(self):
        partial = {"dev": ["a", "b", "c"], "test": ["d", "e"]}
        with self.assertRaisesRegex(ValueError, "Unassigned"):
            cluster.verify_split(partial, self.clusters)

    def test_assignment_is_reproducible(self):
        first = cluster.assign_splits(self.clusters, {"dev": 0.5, "test": 0.5}, seed=7)
        second = cluster.assign_splits(self.clusters, {"dev": 0.5, "test": 0.5}, seed=7)
        self.assertEqual(first, second)

    def test_invalid_fractions_are_rejected(self):
        with self.assertRaises(ValueError):
            cluster.assign_splits(self.clusters, {})
        with self.assertRaises(ValueError):
            cluster.assign_splits(self.clusters, {"dev": 0.0, "test": 0.0})


class TestNullIsReproducibleAcrossProcesses(unittest.TestCase):
    """The null must not depend on per-process string hash randomisation.

    Seeding from `hash()` of a tuple containing sequence text made every z-score differ
    between runs, so clusters and the splits built on them could not be regenerated. A
    same-process check cannot catch that, because one process has one hash seed; these run
    the computation in subprocesses with PYTHONHASHSEED set differently on purpose.
    """

    SEQUENCES = ("MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ", "MKTAYIAKQRQISFVKSHFSRQAAARLGLIEVQ")

    def _in_subprocess(self, hashseed):
        script = (
            "import sys; sys.path.insert(0, %r)\n"
            "import cluster\n"
            "print(repr(cluster.null_statistics(%r, %r, shuffles=25, seed=0)))\n"
            % (str(Path(__file__).resolve().parent), self.SEQUENCES[0], self.SEQUENCES[1]))
        env = dict(os.environ, PYTHONHASHSEED=hashseed)
        out = subprocess.run([sys.executable, "-c", script], capture_output=True,
                             text=True, env=env, check=True)
        return out.stdout.strip()

    def test_identical_under_different_hash_seeds(self):
        first = self._in_subprocess("0")
        second = self._in_subprocess("12345")
        third = self._in_subprocess("98765")
        self.assertEqual(first, second)
        self.assertEqual(first, third)

    def test_relatedness_is_stable_too(self):
        """The z-score is what thresholds act on, so it is the value that must be stable."""
        script = (
            "import sys; sys.path.insert(0, %r)\n"
            "import cluster\n"
            "r = cluster.relatedness(%r, %r, shuffles=25, seed=0)\n"
            "print('%%.12f' %% r['z'])\n"
            % (str(Path(__file__).resolve().parent), self.SEQUENCES[0], self.SEQUENCES[1]))
        values = set()
        for seed in ("1", "777", "31337"):
            env = dict(os.environ, PYTHONHASHSEED=seed)
            out = subprocess.run([sys.executable, "-c", script], capture_output=True,
                                 text=True, env=env, check=True)
            values.add(out.stdout.strip())
        self.assertEqual(len(values), 1, "z-score varied with PYTHONHASHSEED: %s" % values)


if __name__ == "__main__":
    unittest.main()

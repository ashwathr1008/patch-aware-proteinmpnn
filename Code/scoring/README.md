# Controlled frozen ProteinMPNN scoring

This runner compares full original and composition-matched permuted sequences on
the same complete backbone. It performs inference with frozen vanilla and/or soluble
`v_48_020` checkpoints. These are compatibility scores, not aggregation, solubility,
folding or experimental measurements. "Original" is the sequence in the deposited
structure; for a designed scaffold this is not necessarily a naturally evolved sequence.

From the repository root in the existing project environment:

```powershell
.\.venv\Scripts\python.exe Code/scoring/score_pairs.py --pairs Data/first_pair/pairs.json --mpnn-dir Runs/psc-smoke-20261104/vendor/ProteinMPNN --output Results/my-paired-scores --device cuda --orders 8 --order-seed 137 --models vanilla,soluble
.\.venv\Scripts\python.exe -m unittest discover -s Code/scoring -p test_*.py -v
```

Use a new output directory. CPU mode is `--device cpu`. The input schema is:

```json
{
  "schema_version": 1,
  "purpose": "patch_compatibility_pilot",
  "structures": [{
    "structure_id": "1UBQ", "chain": "A", "pdb_path": "raw/1UBQ.pdb",
    "sha256": "required-exact-input-sha256", "split": "development", "group": "ubiquitin"
  }],
  "pairs": [{
    "pair_id": "ubq_pair_001", "structure_id": "1UBQ", "chain": "A",
    "patch_id": "ubq_patch_001", "original_sequence": "FULL_ORIGINAL_SEQUENCE",
    "permuted_sequence": "FULL_PERMUTED_SEQUENCE", "patch_indices": [6, 7, 8, 9, 10, 11, 12, 13]
  }]
}
```

PDB paths are relative to the pair manifest. Patch indices are **one-based observed
sequence positions**, not PDB residue numbers. Pair sequences must be canonical,
the same length, and different; patch compositions must match exactly and all outside
identities must be unchanged. Every structure must have at least one pair. Input
hashes, original sequences, complete backbone masks and upstream parser mappings
are checked before inference. Numbering gaps and incomplete backbones fail explicitly.
Structure staging reuses the baseline parser and retains original residue IDs. Only
the selected chain's N/CA/C/O coordinates are supplied to the model.

## What is controlled

- Model/code/checkpoint hashes are pinned to the same upstream revision as the baseline.
- `model.eval()`, disabled gradients and zero backbone-coordinate noise apply to all calls.
- Each structure has a reproducible set of unique explicit residue permutations. The same
  permutations are shared by all its pairs and both checkpoints. No order is chosen after
  seeing results. The seed controls their generation, not implicit model order.
- Every primary model call sets `use_input_decoding_order=True`. A scoped observer
  intercepts the **actual tensor used to construct the upstream autoregressive decoding
  mask**, requires it to equal the recorded order, and restores the original PyTorch
  function. Merely passing equal seeds is not considered order verification.
- Sequence presentation order is deterministically randomized. The first pair of every
  structure/checkpoint is rescored in reversed presentation order at every decoding order.
  Its repeated original sequence supplies an identical-sequence null check. Per-residue
  differences must remain within an absolute tolerance of `1e-6`.
- For the first order of that audited pair, an additional call sets
  `use_input_decoding_order=False` and supplies distinct positive random-input ranks that
  reconstruct the recorded permutation. The observed mask order and per-residue scores
  must match the explicit-order call. This validates both upstream order-selection paths.

All positions are scored under full-chain autoregression. Outside-patch identities
are unchanged, but they are not universally revealed before the patch: which identities
are available depends on the explicit order. This is not a conditional score with all
outside residues fixed as already known. Sequence effects can propagate into the NLL
contributions of unchanged positions.

## Scores and interpretation

The primary score is the mean negative log probability (natural-log units per residue)
over the entire chain. **Gap = permuted NLL minus original NLL**; positive means the
model prefers the original sequence under that order. Scores use unscaled model
probabilities over the upstream 21-symbol alphabet, including its unknown-symbol
category; there is no sampling-temperature transformation or amino-acid bias.

The secondary patch score averages the same full-chain autoregressive terms only at
patch positions. It is not an isolated patch likelihood, pseudo-likelihood, causal
attribution, or aggregation metric. Arithmetic means over orders are means of NLLs;
they do not compute an order-marginalized probability. Differences between checkpoints
are descriptive comparisons between different learned distributions.

`fraction_native_preferred` in `pair_summary.csv` is the fraction of **decoding orders**
that prefer the original, assigning half-credit to gaps within `±1e-6`. The field uses
"native" for continuity with the original baseline but includes deposited designed
sequences. Pair-level decisions based on order-averaged scores are a separate analysis.
Overlapping patches and many permutations of one protein are not independent biological
replicates. The order standard deviation uses the population formula (`ddof=0`) and is
descriptive order sensitivity, not a standard error across proteins.

## Output

- `pair_scores.csv`: one row per pair/checkpoint/order, with original/permuted whole-chain
  and patch NLLs, signed gaps, order hash, presentation order and mapping identifiers.
- `pair_summary.csv`: per-pair/checkpoint means, order spread and order preference fraction.
- `per_residue_nll.npz`: per-residue values for inspection; keys contain pair, model,
  order index and member. No raw side-chain features are used.
- `inputs/`: exact staged backbone inputs.
- `metadata/`: exact source pair manifest, configuration/checkpoint hashes, explicit
  one-based orders, residue mappings, environment and completed/failed summary. Failures
  preserve diagnostics and earlier score rows. A directory's existence is not success;
  check `metadata/summary.json` for `status: complete`.

The scorer does not choose new structures, submit cluster jobs, or write to GitHub.
Keep full runs local; export reviewed small artifacts separately.

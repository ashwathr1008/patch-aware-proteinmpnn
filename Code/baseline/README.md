# Small ProteinMPNN development baseline

This offline runner generates sequences with the pinned official vanilla and soluble
`v_48_020` checkpoints. It does not train a model. Its small development inputs test
the pipeline; they are not a benchmark or evidence of reduced aggregation.

Run from `Project/`, using a Python environment with NumPy and PyTorch:

```powershell
.\.venv\Scripts\python.exe Code/baseline/run_baseline.py --manifest Data/development_manifest.json --mpnn-dir Runs/psc-smoke-20261104/vendor/ProteinMPNN --output Results/baseline-20261104 --samples 8 --seed 37 --temperature 0.1 --device cuda
```

If the vendor directory is absent in a fresh clone, prepare the exact pinned files:

```powershell
python Compute/psc/prepare_smoke.py --output Runs/psc-smoke-20261104 --commit 8907e6671bfbfc92303b5f79c4b5e6ce47cdef57
```

This preparation downloads files and does not execute models or submit jobs. It
requires a new output directory. See [the preparation documentation](../../Compute/psc/README.md)
for details; the baseline itself needs no PSC account.

Use `--device cpu` for CPU execution; the runner hides CUDA from itself and child
processes. CUDA execution requires a successful tensor-operation preflight, including
matrix multiplication and synchronization, before any model runs. There is no silent
fallback. OMP and MKL threads are set to 2. Both devices run batch size 1 with no
backbone noise. Each structure/model subprocess receives the same explicit seed.

The output directory must not exist. Failed runs retain their logs and a failed
`metadata/summary.json`; use a new output path when rerunning. Already validated
runs remain in `candidates.csv` if a later run fails. The runner neither downloads
files nor submits cluster jobs.

## Manifest

Paths are relative to the manifest's directory. Input SHA256 checksums are required;
`expected_length` is optional. Structure/chain pairs must be unique.

```json
{
  "schema_version": 1,
  "purpose": "development_only",
  "structures": [
    {
      "structure_id": "1UBQ",
      "chain": "A",
      "pdb_path": "raw/1UBQ.pdb",
      "sha256": "replace-with-the-64-character-input-sha256",
      "expected_length": 76
    }
  ]
}
```

The runner stages the first PDB model and the requested chain, using canonical
residues with complete N/CA/C/O coordinates. Original residue numbers and insertion
codes are retained and recorded. It chooses a complete alternate conformer per
residue by summed backbone occupancy (lexical altloc breaks ties); blank altloc atoms
are shared. Incomplete backbones and noncanonical ATOM residues fail explicitly.
HETATM records are excluded. Numbering gaps that ProteinMPNN would insert as missing
residues are unsupported by this starter and cause a validation error. The staged
structure contains only backbone atoms and is not a repaired all-atom structure.

## Outputs and interpretation

- `candidates.csv`: one native and `--samples` generated rows per structure/model;
  full sequences, stable record IDs, sequence hashes, length, exact identity count
  and fraction against native, model scores, sampling settings and runtime.
- `results/`: unmodified upstream FASTA and NPZ outputs.
- `inputs/`: exact first-model backbones passed to ProteinMPNN.
- `logs/`: stdout/stderr for every model subprocess.
- `metadata/`: config, input/model/code hashes, environment and package versions,
  exact command argument lists, residue mappings, validation results and status.

Generated `score` and `global_score` come from the float32 NPZ arrays; native scores
are available only rounded to four decimal places in upstream FASTA and are tagged
accordingly. Scores are mean negative log probabilities from the generation run.
Native and sampled sequences use different decoding orders, and the two checkpoints
define different distributions. These raw scores are **not** a controlled rescoring
experiment or a calibrated comparison between model families. For these isolated
single-chain complete-backbone inputs, score and global_score should agree.

Recovery is sequence identity to the native sequence; a higher value does not
establish improved protein quality. No folding, stability, solubility, aggregation,
or experimental validation is performed here. No patch method is applied in this
baseline. Validate any later patch-method comparison with a separate evaluation.

The upstream commit is `8907e6671bfbfc92303b5f79c4b5e6ce47cdef57`; exact code and
checkpoint SHA256 values are enforced. Passing POSIX-style path strings avoids
upstream path parsing problems on Windows without editing vendor code.

Run fixture checks without model execution:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s Code/baseline -p "test_*.py" -v
```

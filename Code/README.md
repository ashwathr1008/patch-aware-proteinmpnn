# Shared implementation

The [baseline runner](baseline/README.md) runs pinned, frozen vanilla and soluble ProteinMPNN checkpoints and validates every output. The November 4 local pilot is complete; see [actual results](../Analysis/README.md).

The [patch starter](patches/README.md) constructs one reproducible, backbone-only, composition-preserving patch permutation example and records its geometry, assignments, residue mapping, and invariant checks. It is a development demonstration; training and evaluation are not implemented.

Keep experiment parameters and dataset manifests separate from implementation. Existing compute/transfer utilities live in `Compute/psc/`.

The [controlled scorer](scoring/README.md) compares existing original/permuted sequences under explicit shared decoding orders. The [multi-patch builder](patches/DATASET.md) expands the starter to several patches and distinct composition-preserving permutations, with a fixed spatial descriptor and full validation. These are compatibility diagnostics, without aggregation labels.

## Local setup

Run from the repository root. The verified local environment is Python 3.12.6, PyTorch 2.7.0+cu128, NumPy 2.0.2 and Matplotlib 3.9.4 on an RTX 4090 Laptop GPU. Create a project environment rather than modifying a homework environment:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe Code/prepare_development_data.py --data-dir Data --fetch-only
.\.venv\Scripts\python.exe Compute/psc/prepare_smoke.py --output Runs/local-vendor
```

The CUDA build above is the one tested on this laptop. For CPU-only Windows/Linux, install `torch==2.7.0` from `https://download.pytorch.org/whl/cpu` instead and pass `--device cpu`. Other hardware needs its own compatible installation; see [official version-specific packages](https://pytorch.org/get-started/previous-versions/). Linux/macOS environment executables use `.venv/bin/python`. CPU execution is supported by the wrapper but was not timed in this pilot.

`--fetch-only` downloads missing raw structures and verifies the committed manifest hashes without changing it. The model-preparation command downloads pinned code/weights and creates a smoke-test directory; it does not execute a model or submit a job.

```powershell
.\.venv\Scripts\python.exe Code/baseline/run_baseline.py --manifest Data/development_manifest.json --mpnn-dir Runs/local-vendor/vendor/ProteinMPNN --output Results/my-baseline --samples 8 --seed 37 --temperature 0.1 --device cuda
.\.venv\Scripts\python.exe Code/patches/starter.py --pdb Data/raw/1UBQ.pdb --chain A --output Results/my-patch --seed 7 --min-patch-size 8
.\.venv\Scripts\python.exe Code/patches/visualize.py --manifest Results/my-patch/manifest.json --output Results/my-patch-figure
.\.venv\Scripts\python.exe Analysis/summarize_baseline.py --run-dir Results/my-baseline --output Results/my-baseline-summary
```

All output directories must be new. The runner verifies a real CUDA operation, checks input/model hashes and preserves failures. It redesigns the complete selected chain; no residue identities or disulfides are fixed. The example permutation is a separate input for later controlled scoring, not part of baseline generation.

## Checks

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s Code/baseline -p test_*.py -v
.\.venv\Scripts\python.exe -m unittest discover -s Code -p test_*.py -v
.\.venv\Scripts\python.exe -m unittest discover -s Analysis -p test_*.py -v
.\.venv\Scripts\python.exe -m unittest discover -s Compute/psc -p test_bundle.py -v
```

These validate parsing, output accounting, patch invariants, report integrity and archive retrieval without launching inference or Slurm jobs.

# Bridges-2 project runs and local retrieval

Prepared November 4, 2026. User confirmed course-allocation permission extends beyond the mid-report. The outage, not the mid-report, limits later use. Mid-report results should still be finished November 4–5.

Use PSC when an available slot or its existing environment saves time. Keep preprocessing, report work, and small-model fallback runnable locally. Useful additional PSC work could include larger frozen-model batches, preprocessing independent structures, or a limited structural-prediction subset if the required software is already working. Do not start a new folding-stack installation solely to use available allocation.

## Cutoffs (America/New_York)

| Work | Compute finish target | Verified local copy target |
|---|---|---|
| Mid-report | November 5, 18:00 | November 5, 20:00 |
| Optional final-project preparation | November 7, 18:00 | November 8, 12:00 |

Download each finished batch immediately. These are internal targets, not PSC shutdown times. [PSC](https://www.psc.edu/b2-migration-2026/) schedules the full outage from November 9 for approximately 4–6 weeks. Do not rely on being able to transfer files on November 9.

## Verified existing access

- Login alias: `bridges2`; username: `raghuram`.
- Transfer alias: `b2-data`, for `data.bridges2.psc.edu`.
- Use only course account `cis260266p` for this project. The November 4 live check showed 11,116 remaining GPU service units in its shared balance. This is not a personal allocation and service-unit cost depends on GPU type.
- Existing Python: `/ocean/projects/cis260266p/raghuram/envs/progen/bin/python` (Python 3.9.16, PyTorch 2.7.0+cu128, NumPy 2.0.2). It is reused without changing the homework environment.
- New isolated project area: `/ocean/projects/cis260266p/raghuram/patch_aware_project`.
- Heavy work runs through Slurm, never on a login node. File staging and small archive operations are separate from model inference.

These aliases and paths describe Ashwath's setup. Each teammate must use their own authorized PSC identity and storage paths. Before using the scripts from another account, configure the SSH aliases and update `TASK_PYTHON` in `smoke.sbatch` and the remote paths/aliases in `Collect-PscRun.ps1`; those scripts currently contain Ashwath's settings. Sharing this repository does not share an account or environment access.

## Initial engineering check

Run name: `psc-smoke-20261104`. Job **50214872** ran on November 4 around **12:02–12:03 Eastern** and **failed**. It requested one `v100-16` GPU with a 15-minute limit and November 5 18:00 deadline. The existing PyTorch 2.7.0+cu128 binary requires CUDA capability 7.5 or newer, while this V100 has capability 7.0. The first vanilla-model run failed with `no kernel image is available for execution on the device`; no successful sequence-generation result was produced. The collector subsequently downloaded and verified all **17 files** from the failed run.

Use the template's **L40S default** with this environment rather than overriding it to V100. A V100 requires a separately verified compatible PyTorch build. CUDA availability alone is insufficient to establish that a particular GPU can execute the installed binary.

On November 4, replacement job **50216628** was submitted with `--gpus=h100-80:1`, using run `psc-smoke-h100-20261104` and a real CUDA-operation preflight. It was still **pending** at the last check while the local pilot completed. This is one eight-sequence engineering test, separate from the 48-sequence baseline. A hidden local PowerShell collector watches this job and retrieves/checks it when terminal, with a November 5 20:00 Eastern deadline. Its logs are under `.local/h100-50216628-collector.log` and `.err`; it requires this laptop to remain awake, online and authenticated. This is a local process, not a hosted scheduled service. Verify the receipt and run summary before counting any PSC output as successful.

The local RTX 4090 Laptop GPU has now successfully completed both checkpoints on three structures: 48 sequences, 39.4 seconds across six model subprocesses (42.8 seconds end to end). A same-machine 1UBQ repeat reproduced 16 sequences and scores exactly. See [actual baseline results](../../Analysis/README.md). The midpoint baseline therefore has no outstanding cluster dependency.

This test uses public ubiquitin structure [1UBQ](https://www.rcsb.org/structure/1UBQ), chain A, to exercise vanilla and soluble ProteinMPNN checkpoints, four samples each. It is a software/compute check, not a held-out evaluation, developability result, or chosen project dataset. The exact upstream commit and file hashes are in `config.json`. Only the two required source files, two checkpoints, license and upstream README are downloaded.

The prepared run is under `Runs/psc-smoke-20261104` in Ashwath's local copy. It includes input structure, code, model weights, configuration, and locations for logs, environment metadata and outputs. After successful inference, the runner checks expected sequence count/length and finite numerical scores; the failed run did not reach those checks. The upstream FASTA may report an unknown Git hash because a full Git checkout is unnecessary; `config.json` records the pinned commit. Generated run directories are not needed in Git; teammates can prepare their own using the command below.

At setup, `sbatch --test-only` estimated November 8 starts, but the submitted job actually started on November 4. Scheduling estimates are uncertain; keep the midpoint runnable locally. The job's stored November 5 deadline was verified. [Slurm deadlines](https://slurm.schedmd.com/sbatch.html) constrain eligibility but do not guarantee successful execution or downloads.

## Prepare, stage and submit another test

From the repository root (the `Project/` directory in the original course workspace) in PowerShell, use a new run name each time. These remote commands use Ashwath's configuration; teammates must substitute their own authorized paths and aliases:

```powershell
python Compute/psc/prepare_smoke.py --output Runs/NEW_RUN
scp -r Runs/NEW_RUN b2-data:/ocean/projects/cis260266p/raghuram/patch_aware_project/runs/
ssh bridges2 'cd /ocean/projects/cis260266p/raghuram/patch_aware_project/runs/NEW_RUN && sbatch --test-only smoke.sbatch'
ssh bridges2 'cd /ocean/projects/cis260266p/raghuram/patch_aware_project/runs/NEW_RUN && sbatch --parsable smoke.sbatch'
```

`NEW_RUN` is a placeholder. Do not submit duplicates while a job is pending. This preparation script is intentionally a fixed small test; actual experiments require their own agreed inputs and configuration. It refuses an existing output directory. For later November 6–7 jobs, explicitly update both the Slurm deadline and execution-time cutoff to the later approved target; the midpoint script deliberately expires November 5.

## Retrieve and verify a completed run

```powershell
pwsh -NoProfile -File Compute/psc/Collect-PscRun.ps1 -RunName NEW_RUN -JobId JOB_ID
```

Replace `NEW_RUN` and `JOB_ID` with the run directory name and numeric Slurm job ID. The initial job `50214872` has already been collected locally; the collector refuses to overwrite an existing extraction directory.

Add `-Wait` to poll every minute until terminal, bounded by the collection deadline. The default is November 5 20:00 Eastern. A later approved job can use `-Deadline '2026-11-08T12:00:00-05:00'`. Waiting requires this computer to remain awake, connected and able to authenticate through SSH; it is not a hosted scheduled task.

The collector:

1. Checks Slurm state and verifies that the job's working directory matches the requested run.
2. Bundles a terminal run with its code, inputs, weights, configuration, logs and outputs. It also recovers failed-run files without presenting them as successful results.
3. Downloads through the data-transfer host, using partial filenames until transfer completes.
4. Verifies the archive SHA256 and every embedded file's SHA256/size, then extracts into a new directory.
5. Writes a local receipt recording state, paths, archive hash and verification time.

Final files go to `Results/PSC/<run>-job<id>`; archives and sidecars go to `Results/PSC/archives`. The receipt, not successful `scp` output alone, is the completion criterion. Nothing deletes the remote originals. Check that the scientific summary says `complete` as well as checking Slurm state.

The initial local collector finished its recovery of job `50214872`. Its process metadata, `collector-50214872.*.log` files and receipt are under `Results/PSC` in Ashwath's local copy. For a new job, check the corresponding logs and receipt to confirm retrieval. If a future collector stops or authentication changes, rerun collection. No future experiment batches are automatically submitted by these scripts.

## Transfer test and local fallback

The November 4 staging snapshot was packed on PSC, downloaded through `b2-data`, verified and extracted locally: `Results/PSC/psc-staging-check-20261104`. All 11 staged files passed integrity checks. The subsequent failed-run archive in `Results/PSC/psc-smoke-20261104-job50214872` also passed verification for all 17 files. These establish transfer and recovery, not successful inference or biological results. The current wrapper additionally creates missing empty output directories when rerun.

For a local run, use a Python environment with compatible PyTorch and NumPy, then run `python code/run_smoke.py --root .` from a new prepared directory. CUDA is used when available; otherwise the upstream runner supports CPU. A full Conda environment is not copied between Linux and Windows; pinned source, model weights and `pip-freeze.txt` support recreation instead. Local PyTorch was not installed during PSC setup.

Archive utility tests cover roundtrip integrity, corrupt archives, traversal, links, duplicate names and non-overwrite behaviour. Run with `python -m unittest discover -s Compute/psc -p test_bundle.py` from the repository root.

Official references: [PSC user guide](https://www.psc.edu/resources/bridges-2/user-guide/), [ProteinMPNN source](https://github.com/dauparas/ProteinMPNN), [Slurm sbatch](https://slurm.schedmd.com/sbatch.html).

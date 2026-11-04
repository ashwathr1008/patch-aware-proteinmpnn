#!/bin/bash
#SBATCH --job-name=patch-mpnn-smoke
#SBATCH --account=cis260266p
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --gpus=l40s-48:1
#SBATCH --time=00:15:00
#SBATCH --deadline=2026-11-05T18:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Run with sbatch from the prepared run directory}"
test -n "${SLURM_JOB_ID:-}"
# Bridges-2's America/New_York clock was verified before submission.
if [[ $(TZ=America/New_York date +%Y%m%d%H%M) -ge 202611051800 ]]; then
    printf '%s\n' 'Past the midpoint compute cutoff; use the local fallback.' >&2
    exit 1
fi
TASK_PYTHON=/ocean/projects/cis260266p/raghuram/envs/progen/bin/python
test -x "$TASK_PYTHON"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
hostname
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
"$TASK_PYTHON" -u code/run_smoke.py --root "$PWD" --require-cuda

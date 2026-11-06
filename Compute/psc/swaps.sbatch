#!/bin/bash
# Full autoregressive scoring of the measured arrangement swaps.
#
# Configure before first use. TASK_PYTHON and the run paths below belong to whoever submits
# this; they are not shared with the repository. Set TASK_PYTHON to a Python that has a
# PyTorch build this GPU can actually execute. The November 4 V100 failure is the reason that
# sentence is here: CUDA being available did not mean the installed binary could run.
#
# Submit from a prepared run directory holding code/, inputs/ and logs/:
#   sbatch --test-only swaps.sbatch      # check it is accepted and see an estimated start
#   sbatch --parsable  swaps.sbatch      # submit
#
#SBATCH --job-name=patch-mpnn-swaps
#SBATCH --account=cis260266p
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --gpus=l40s-48:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Run with sbatch from the prepared run directory}"
test -n "${SLURM_JOB_ID:-}"

# Bridges-2 begins a migration outage on November 9, 2026. A job that cannot finish and be
# collected before then should not start: partial output that cannot be retrieved is worse
# than no output, because it looks like progress.
OUTAGE=202611090000
if [[ $(TZ=America/New_York date +%Y%m%d%H%M) -ge ${OUTAGE} ]]; then
    printf '%s\n' 'Past the Bridges-2 migration cutoff; run elsewhere.' >&2
    exit 1
fi

: "${TASK_PYTHON:?Set TASK_PYTHON to your own Python with a working GPU PyTorch build}"
test -x "$TASK_PYTHON"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

hostname
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

MODEL_TYPE="${MODEL_TYPE:-vanilla}"
ORDERS="${ORDERS:-8}"
BATCH="${BATCH:-256}"

# --require-cuda makes the job fail loudly rather than silently falling back to CPU, where
# this workload would take days rather than hours.
"$TASK_PYTHON" -u code/score_swaps_full.py \
    --contrasts inputs/swap_contrasts.csv \
    --structures inputs/structures \
    --mpnn-dir inputs/ProteinMPNN \
    --output "results/${MODEL_TYPE}" \
    --model-type "$MODEL_TYPE" \
    --device cuda --require-cuda \
    --orders "$ORDERS" --batch-size "$BATCH"

printf '%s\n' "Finished ${MODEL_TYPE}. Collect results/ before the November 9 outage."

#!/bin/bash
# Bounded CPU-only CP2K state/element smoke suite for xtbflow.
# Submit from the repository root on the configured Slurm host.
#SBATCH --job-name=xtbflow-cp2k
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
# Slurm opens these files before the shell can create a per-job directory.
# The JSON result itself is written below runs/cp2k-${SLURM_JOB_ID}/.
#SBATCH --output=cp2k-%j.out
#SBATCH --error=cp2k-%j.err

set -euo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate xtbflow
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export XTBFlow_CP2K_EXECUTABLE="${XTBFlow_CP2K_EXECUTABLE:-$HOME/miniconda3/envs/xtbflow-cp2k/bin/cp2k.psmp}"
mkdir -p "runs/cp2k-${SLURM_JOB_ID}"
python scripts/cp2k_calibration_smoke.py \
  --threads "${SLURM_CPUS_PER_TASK}" \
  --timeout 600 \
  --artifact-dir "runs/cp2k-${SLURM_JOB_ID}/artifacts" \
  --output "runs/cp2k-${SLURM_JOB_ID}/calibration.json"

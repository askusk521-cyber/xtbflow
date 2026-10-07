#!/bin/bash
# Run one M0 command inside a Slurm allocation, from the checkout it is submitted in.
# Example (from the source checkout on n2):
#   sbatch --partition=main --gres=gpu:pro6000:1 --cpus-per-task=8 --time=01:30:00 \
#     --output=<run-root>/slurm-%j.log scripts/m0_slurm.sh \
#     python scripts/m0_train.py --config configs/m0/base.json --role joint --seed 0 --out <new-dir>
# m0_train.py opens its --out with exist_ok=True and would overwrite logs, so a training
# --out that already exists is refused here: earlier runs and failures must be kept.
set -euo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate xtbflow
cd "${SLURM_SUBMIT_DIR:-.}"
export XTBFLOW_DATA="${XTBFLOW_DATA:-$HOME/data}" PYTHONPATH=src:vendor/mechai_reusable
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
args=("$@")
for ((i = 0; i < ${#args[@]} - 1; i++)); do
  if [ "${args[i]}" = "--out" ] && [ "${args[0]}" = python ] && [[ "${args[1]}" == *m0_train.py ]]; then
    test ! -e "${args[i+1]}" || { echo "refusing to reuse existing run dir ${args[i+1]}" >&2; exit 2; }
  fi
done
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} git=$(git rev-parse HEAD) start=$(date -Is)"
"$@"
echo "end=$(date -Is)"

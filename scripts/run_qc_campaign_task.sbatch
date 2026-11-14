#!/usr/bin/env bash
set -euo pipefail
QC_CODE="$1"
QC_SHA="$2"
QC_MANIFEST="$3"
QC_OUTPUT="$4"
QC_GPU="$5"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate xtbflow-qc
source "$HOME/.config/n2-qc/runtime.env"
export LD_LIBRARY_PATH="$GQC_OPENBLAS_DIR:$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
if [[ "$QC_GPU" == 0 ]]; then
    export CUDA_VISIBLE_DEVICES=
fi
cd "$QC_CODE"
test "$(git rev-parse HEAD)" = "$QC_SHA"
test -z "$(git status --porcelain)"
exec "$GQC_PYTHON" scripts/qc_campaign_task.py --manifest "$QC_MANIFEST" \
    --index "$SLURM_ARRAY_TASK_ID" --out "$QC_OUTPUT"

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
export QC_GRAPH_PYTHON="$HOME/miniconda3/envs/xtbflow/bin/python"
QC_RUN_ROOT="$(dirname "$QC_OUTPUT")"
export PYTHONPATH="$QC_RUN_ROOT/deps${PYTHONPATH:+:$PYTHONPATH}"
if [[ "$QC_GPU" == 0 ]]; then
    export CUDA_VISIBLE_DEVICES=
fi
cd "$QC_CODE"
test "$(git rev-parse HEAD)" = "$QC_SHA"
test -z "$(git status --porcelain)"
set +e
"$GQC_PYTHON" scripts/qc_campaign_task.py --manifest "$QC_MANIFEST" \
    --index "$SLURM_ARRAY_TASK_ID" --out "$QC_OUTPUT"
QC_EXIT=$?
set -e
"$GQC_PYTHON" - "$QC_OUTPUT" "$SLURM_ARRAY_JOB_ID" "$SLURM_ARRAY_TASK_ID" "$QC_EXIT" <<'PY'
import json, pathlib, sys, time
out, job, index, status = sys.argv[1:]
p = pathlib.Path(out) / ("launcher-" + job + "_" + index + ".json")
p.parent.mkdir(parents=True, exist_ok=True)
with p.open("x") as f:
    json.dump({"array_job_id": job, "index": int(index), "worker_exit_code": int(status), "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, f)
PY
exit "$QC_EXIT"

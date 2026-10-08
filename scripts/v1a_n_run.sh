#!/bin/bash
# V1a-N step-count control (exploratory). From a clean clone at a fixed commit:
#   sbatch scripts/v1a_n_run.sh smoke        10 development parents, then the smoke check
#   sbatch scripts/v1a_n_run.sh seed <0|1|2> formal screen parents, one training seed
# Only the unguided arms A0 (25 event + 25 geometry steps) and B0 (25 joint steps) run,
# with the frozen V1a checkpoints, cost table and 'efficiency' noise namespace. Outputs go
# to a new run root; the sealed V1a directories are only read.
#SBATCH --partition=main
#SBATCH --gres=gpu:pro6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --job-name=v1a-n
#SBATCH --output=/home/lhshen/xtbflow-runs/v1a-n-20261008/logs/%x-%j.log
set -euo pipefail
export PYTHONPATH=src:vendor/mechai_reusable PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8
py=/home/lhshen/miniconda3/envs/xtbflow/bin/python
v1a=/home/lhshen/xtbflow-runs/v1a-20261007
root=${V1A_N_ROOT:-/home/lhshen/xtbflow-runs/v1a-n-20261008}
costs=$v1a/evidence/cost_calibration_512.json
nfe=(--arms A0 B0 --nfe-event 25 --nfe-geometry 25 --nfe-joint 25)
stage=$1; seed=${2:-}
case "$stage" in
smoke)
  mkdir -p "$root/smoke/queries"
  head -n 10 "$v1a/data/development_queries.jsonl" > "$root/smoke/queries/development_queries.jsonl"
  "$py" -B scripts/v1a_stream_generate.py --queries "$root/smoke/queries/development_queries.jsonl" \
    --run-root "$v1a" --costs "$costs" --path sync --alpha-x .3 --alpha-b .25 --seed 0 --batch-size 512 \
    "${nfe[@]}" --out "$root/smoke/streams"
  "$py" -B scripts/v1a_n_smoke.py --run "$root/smoke/streams" --catalogue "$v1a/data" --v1a-root "$v1a" \
    --out "$root/smoke/smoke.json" ;;
seed)
  mkdir -p "$root/screen"
  "$py" -B scripts/v1a_stream_generate.py --queries "$v1a/data/screen_queries.jsonl" --run-root "$v1a" \
    --frozen "$v1a/formal/freeze.json" --costs "$costs" --seed "$seed" --batch-size 512 \
    "${nfe[@]}" --out "$root/screen/nfe_s$seed"
  "$py" -B scripts/v1a_stream_evaluate.py --run "$root/screen/nfe_s$seed" --catalogue "$v1a/data" \
    --frozen "$v1a/formal/freeze.json" --out "$root/screen/nfe_s$seed.json" ;;
*) echo "unknown stage $stage" >&2; exit 2 ;;
esac

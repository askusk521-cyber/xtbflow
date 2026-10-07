#!/bin/bash
# V1a formal screen driver. Usage (from a clean checkout at the frozen commit):
#   sbatch scripts/v1a_formal.sh <stage> [seed]
# Stages, in order: plan, freeze, seed (0/1/2), controls, rarity, report.
# Every stage writes into new directories and refuses to overwrite earlier output.
#SBATCH --partition=main
#SBATCH --gres=gpu:pro6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --job-name=v1a-formal
#SBATCH --output=/home/lhshen/xtbflow-runs/v1a-20261007/evidence/formal-%x-%j.log
set -euo pipefail
export PYTHONPATH=src:vendor/mechai_reusable PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
py=/home/lhshen/miniconda3/envs/xtbflow/bin/python
root=${V1A_ROOT:-/home/lhshen/xtbflow-runs/v1a-20261007}
ev=$root/evidence; data=$root/data; freeze=$root/formal/freeze.json; screen=$root/screen
stage=$1; seed=${2:-}
case "$stage" in
plan)
  "$py" -B scripts/v1a_power_plan.py --a2-report "$ev/streams_a030_early.json" \
    --b1-report "$ev/streams_sync_a030.json" --window-report "$ev/selected_pulse_s0_final.json" \
    --extra-a2-reports "$ev/streams_sync_s1.json" "$ev/streams_sync_s2.json" \
    --extra-b1-reports "$ev/streams_sync_s1.json" "$ev/streams_sync_s2.json" \
    --extra-window-reports "$ev/selected_pulse_s1_final.json" "$ev/selected_pulse_s2_final.json" \
    --time .5 --sd-multiplier 1 --icc-increment 0 --catalogue "$data" --out "$ev/power_plan_formal.json" ;;
freeze)
  mkdir -p "$root/formal"
  "$py" -B scripts/v1a_freeze.py --catalogue "$data" --run-root "$root" --power-plan "$ev/power_plan_formal.json" \
    --superseded-power-plan "$ev/power_plan_final.json" --costs "$ev/cost_calibration_512.json" \
    --baseline-report "$ev/baseline.json" --controls-report "$ev/continuous_controls.json" \
    --drift-reports "$ev/streams_sync_a030.json" "$ev/streams_sync_s1.json" "$ev/streams_sync_s2.json" \
    --pulse-reports "$ev/selected_pulse_s0_final.json" "$ev/selected_pulse_s1_final.json" "$ev/selected_pulse_s2_final.json" \
    --decision docs/v1a/FREEZE_DECISION_ZH.md --out "$freeze" ;;
seed)
  q=(--queries "$data/screen_queries.jsonl" --run-root "$root" --frozen "$freeze")
  "$py" -B scripts/v1a_stream_generate.py "${q[@]}" --costs "$ev/cost_calibration_512.json" --seed "$seed" \
    --batch-size 512 --out "$screen/efficiency_s$seed"
  "$py" -B scripts/v1a_stream_evaluate.py --run "$screen/efficiency_s$seed" --catalogue "$data" --frozen "$freeze" \
    --out "$screen/efficiency_s$seed.json"
  "$py" -B scripts/v1a_window_generate.py "${q[@]}" --seed "$seed" --out "$screen/pulses_s$seed"
  "$py" -B scripts/v1a_window_evaluate.py --run "$screen/pulses_s$seed" --catalogue "$data" --frozen "$freeze" \
    --out "$screen/pulses_s$seed.json" ;;
controls)
  "$py" -B scripts/v1a_control_probe.py --queries "$data/screen_queries.jsonl" --run-root "$root" --frozen "$freeze" \
    --out "$screen/controls"
  "$py" -B scripts/v1a_control_evaluate.py --run "$screen/controls" --catalogue "$data" --frozen "$freeze" \
    --out "$screen/controls.json" ;;
rarity)
  "$py" -B scripts/v1a_stream_generate.py --queries "$data/screen_queries.jsonl" --run-root "$root" --frozen "$freeze" \
    --costs "$ev/cost_calibration_512.json" --rarity --batch-size 512 --out "$screen/rarity"
  "$py" -B scripts/v1a_stream_evaluate.py --run "$screen/rarity" --catalogue "$data" --frozen "$freeze" \
    --out "$screen/rarity.json" ;;
report)
  "$py" -B scripts/v1a_report.py --frozen "$freeze" \
    --efficiency "$screen"/efficiency_s{0,1,2}.json --pulses "$screen"/pulses_s{0,1,2}.json \
    --controls "$screen/controls.json" --rarity "$screen/rarity.json" --out-dir "$root/final" ;;
*) echo "unknown stage $stage" >&2; exit 2 ;;
esac

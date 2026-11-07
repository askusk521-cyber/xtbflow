# Comparison asset replay — 2026-11-07, job 4102

**Passed: both original analyses reproduced byte for byte.** This is a replay of
historical catalogue-event recall, not a new physical experiment or a full-cost
method ranking. The reference-cost adapter independently recovered the successful
and interrupted job-4096 attempts without issuing new calculations.

Source `912931245c7349cc651f354b8660088c09f597b3`, branch
`codex/w138-comparison-accounting-20261107`, based on PR #148 at
`e07ac8512fa987e0fad2e37b14c576c534a7df87`. Execution host `node2`,
Conda `xtbflow`, Python 3.10.19, NumPy 2.2.6, SciPy 1.15.3,
PyTorch 2.10.0+cu128. [Full audit](comparison_assets_replay_20261107.json)
and [scheduler/test identities](comparison_assets_replay_20261107_envelope.json)
retain exact source/config/input/output hashes and limitations.

## Historical paired replay

| Asset | Original exact head | Reproduced output SHA256 |
| --- | --- | --- |
| PR113 | `3db499ca1ae956dc572ee87439fe654b0192d8f3` | `ea2237f229c718069abef225cd0c76c068357c653b6829b3275b74603d8be0b4` |
| PR120 | `b6db3df15eefd85a64eed329674a48d3ecba301e` | `7d4a3372791b4b4f727e305f4331fa6f91cd790e24c255e38e06cbe5002c03b3` |

Both source checkouts were clean and their pinned files verified before and after
execution. The audit verified **1571 unique files, 1,109,778,606 bytes**, including
enumerated events/scores, the 342-parent physical-acquisition outputs, source
catalogue, split/freeze, model/scorer inputs and generator streams. Raw data,
weights and full reconstructed legacy analyses remain on n2.

Both analyses cover 342 historical screen parents; their original primary
comparisons retain 335 parents and 66 formula groups after the original censoring
rules. Their values are unchanged:

| Historical proxy comparison | Candidate-axis AUC difference | Original two-sided 95% interval |
| --- | ---: | --- |
| `S_N - B0` | −0.1765458422 | [−0.2304110844, −0.1226806001] |
| `S_N_thermo - B0` | −0.0554371002 | [−0.1069630915, −0.0039111089] |

These are the old enumeration-minus-generation **reference-event hit** statistics,
conditional on the historical models and candidate domain. They are not
certified-channel effects or equal-total-cost results. Their omissions and
censoring are preserved in the JSON; the new failure-inclusive discovery
statistics do not retroactively change them.

The five legacy generator arms each contain 1026 parent/seed cells. Their recorded
proxy units are retained as proxy units, not converted into seconds or calculator
calls. The old thermodynamic screen reports 27,451.454 measured CPU seconds, but
complete per-method training/tuning, generator/enumeration resources and scheduler,
retry/cache overhead are not jointly available. Consequently
`full_cost_comparison_available=false`.

## Reference cost reconstruction

The new adapter verified raw artifact hashes, requests, results, settlements,
ledger events and committed tokens, and recomputed feedback from retained
evidence. The two **reference-only** vectors reconcile exactly:

| Job-4096 case | Evidence | SCF / gradient / Hessian | Total primitives |
| --- | --- | --- | ---: |
| First | `valid_alternative` | 99 / 99 / 4 | 202 |
| Second | `attempted_unresolved`, execution failed | 132 / 131 / 2 | 265 |

The original 467 calls and measured costs remain retained, including the
software-interrupted IRC. These are not new calls and not a paired method
experiment. Generator, training/tuning, shared setup and cache-lookup costs are
explicitly excluded from each reference-only vector.

## Resource envelope and exact execution

The replay used **81.661 s runner wall / 80.684 s measured CPU**, zero GPU and zero
new quantum-chemistry calls. Slurm 4102 finished `COMPLETED`, exit `0:0`, in
84 seconds. The request was one CPU per task and 8 GiB; Slurm actually allocated
two logical CPUs. Allocated CPU capacity and measured CPU use are distinct.
The stage cap was 600 seconds, one analyzer thread, 2 GiB audited input and
64 MiB output; outer allocation 12 minutes. Raw analyzer outputs/logs total
644,715 bytes. Separate software tests at the exact source passed **338 tests,
68 subtests and 56 reusable tests**; bootstrap verified 362 files.

Matrix SHA256: `8705f104c5fc4394ff4dbe8b3e404703f9a7f0ac8d0e963c1fe1497272fcb429`.
Replay config SHA256: `b3d343ba5676a0c918a735dabd3f48992598d83dffba16e31aba7d66319aeeef`.
Final report SHA256: `11c3dc9d88dde17b9e0c35500d62178c53ed04b0f9a0f7d10acf11d9fd8c1847`.
Job script SHA256: `1b9d7f0c0f0c779cd466f38b3a06dc8936a09b9b11de8ba672aa471337f149e1`.

The preserved job script was submitted from the new named n2 execution directory:

```bash
ssh n2 '
  cd "$HOME/xtbflow-runs/w138-20261107-9129312"
  sbatch --partition=main --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=8G \
    --time=00:12:00 --no-requeue --job-name=w138-cost-replay \
    --output=scheduler-%j.log --error=scheduler-%j.log xtbflow-w138-9129312-job.sh
'
```

Its numerical-free analysis invocation was:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate xtbflow
cd "$HOME/xtbflow-runs/w138-20261107-9129312/source"
test "$(git rev-parse HEAD)" = 912931245c7349cc651f354b8660088c09f597b3
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=src:vendor/mechai_reusable PYTHONDONTWRITEBYTECODE=1
timeout 700 python scripts/replay_comparison_assets.py \
  --enum-source "$HOME/xtbflow-runs/w138-assets-20261107/enum-source" \
  --thermo-source "$HOME/xtbflow-runs/w138-assets-20261107/thermo-source" \
  --runs-root "$HOME/xtbflow-runs" \
  --reference-panel "$HOME/xtbflow-runs/w137-20261107-931d81f/panel" \
  --out "$HOME/xtbflow-runs/w138-20261107-9129312/replay"
```

These are the original recorded commands. Re-execution requires another clean
named checkout and a fresh output directory; the script rejects existing output.
The scheduler script also captures source, environment, start/exit state and
return code. `sacct` remains disabled; process accounting and captured scheduler
state are not presented as a Slurm accounting database.

## Remaining work and claim limits

The semantic matrix can dispatch the existing functional joint/serial batch.
It correctly remains `full_experiment_ready=false`. Strong serial backtracking,
separately trained one-way controls, matched terminal/inflow feedback, complete
method costs, new fixed-budget paired certification, mechanism interventions and
W13 independent confirmation are still required. No issue is closed by this
increment; the full #129 roadmap remains active.

两份旧分析已经逐字节重放成功，原有候选数指标和删失规则都保留了下来。
真实认证费用也从原始产物与账本中重新核对，成功和中断案例共 467 次调用均未丢失。
这些结果证明重放和计费入口可用，但不能代替强串联、同总预算的新实验或独立确认。

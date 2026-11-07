# W02 real task-view reconstruction — 2026-11-07

Source: `887bf8635e34c80a423e054526c1ba7a9576e944`, branch
`codex/w132-task-views-20261107`. Base is W07 / PR #145 source
`931d81f53a85669e137600c62b343120ff0de3d0`, itself based on PR #144.
No previous evidence or raw source file was modified.

## Observed result

The existing Diels–Alder development source reconstructs 201 rows in 41 parent
and family groups. The event-only, geometry-only and paired-joint projections
each contain the same split counts; **they overlap and must not be added**.

| Split | Rows per projection | Parent groups | Family groups |
| --- | ---: | ---: | ---: |
| Train | 147 | 29 | 29 |
| Validation | 9 | 3 | 3 |
| Development test | 45 | 9 | 9 |

Energy/force eligibility is zero for this adapter; no missing label is filled
with zero. Confirmation eligibility is zero. Existing source admission and
family/parent splits remain unchanged. Environment is unresolved for
matched-geometry multi-fidelity pairing; no solvent model is inferred.

Three distinct training parents passed hidden-label substitution. Each used
the same reactants, fixed random model parameters, seed 132 and two candidates
with two flow steps. Product coordinates/graphs, event identity and TS
coordinates/identity changed, while model inputs and complete generated attempt
records remained equal. Total: 24 flow evaluations, zero training steps and
zero calculator/GPU calls. This verifies software input isolation, not predictive
accuracy, successful reaction discovery or physical certification.

## Host, costs and identities

Host `node2`, conda `xtbflow`, Python 3.10.19, PyTorch 2.10.0+cu128,
NumPy 2.2.6, RDKit 2024.3.3. Observed audit wall time 2.206492 s;
process CPU time 3.250724 s (includes interpreter/module initialization).
Caps were 600 s wall / 600 s CPU / 16 MiB per output file, one CPU thread,
three cases and no training, quantum chemistry or GPU allocation.

- Source manifest SHA256: `6b84937d3cc75bea44faa7608d690618ce22f7ab75c51b8e25515f05a193cc90`.
- Task index SHA256: `2d4a20aea9c64abce7d76753e3a38bde940d19e68a736397b8bc1270159d3aad`.
- Audit report SHA256: `6e223fb939e0824fc8ed6394abf6ceb04cb2163d90198533312f8212b2040c4e`.
- [Machine-readable audit](task_views_20261107.json) records the script/protocol
  hash, source asset hash, per-case identities, batch shapes and limitations.
- [Training-readable task index](../../data/manifests/task_views_dft_da_20261107.json)
  contains identifiers/hashes only; source coordinates stay in the n2 cache.

Exact-source n2 software validation: 307 pytest tests and 68 subtests passed in
16.67 s; one existing PyTorch scalar-conversion warning. Bootstrap matched 342
files before this separate evidence commit. The real `load_task_views` consumer
is also checked against the generated index; training labels remain a separate
view. No source/schema test result is described as chemistry validation.

## Reproduction

Configure `XTBFlow_RUN_ROOT` and `XTBFlow_DFT_DA_CACHE` on n2 to an operator-owned
run directory and the already-audited source cache. They are not private paths
stored in Git. Use a new execution/output directory for every replay:

```bash
ssh n2 'bash -s' <<'SH'
set -euo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate xtbflow
task_run="${XTBFlow_RUN_ROOT:?}/task-views-887bf86"
mkdir "$task_run"
git clone --no-checkout https://github.com/askusk521-cyber/xtbflow.git "$task_run/source"
git -C "$task_run/source" switch -c replay-w132-887bf86 887bf8635e34c80a423e054526c1ba7a9576e944
cd "$task_run/source"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=src:vendor/mechai_reusable
git rev-parse HEAD
python --version
python scripts/validate_bootstrap.py
# 600 s wall/CPU; 16 MiB per output; 24 flow evaluations; zero QC/GPU/training.
timeout 600 python scripts/audit_task_views.py \
  --cache "${XTBFlow_DFT_DA_CACHE:?}" --out "$task_run/audit" > "$task_run/audit.log" 2>&1
SH
```

Remaining #132 work includes partial-label source admission, energy/force and
same-geometry Delta data, declared conditional completion, cross-source/reverse
reaction grouping and certified-discovery promotion with confirmation isolation.
This is a usable first increment, not closure of the full work package.

## 中文说明

本次把现有 201 条开发数据变成可直接读取的事件、几何和联合监督视图，并保留原来的
41 个母组与训练、验证、测试划分。三个视图使用同一批记录，不能把数量相加后说成
更多独立数据；当前来源没有可用的同构型能量／力标签，清单明确记为零。三个母组
的隐藏标签替换检查都通过，说明这些隐藏结果不会改变本次固定生成器的输入和候选。
这项检查不代表模型预测准确或反应已得到量化认证，后续仍需接入多保真数据、合法
条件输入和经过隔离审计的新发现回流。

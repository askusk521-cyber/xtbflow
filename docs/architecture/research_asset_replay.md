# Research asset replay / 研究资产重放

This is the first integration increment for #131, with handoffs to #132 and
#137. The registry `configs/research/assets_20261107.json` pins main and the
five research PR heads observed on 2026-11-07, including configuration,
implementation and evidence hashes. It does not merge those branches or grant
training admission. Historical evidence remains exploratory.

本入口先验证旧研究产物的来源，再重新计算已有认证报告，并把同一候选的事件、
坐标、端点和费用关联起来。它为后续统一接口提供可核验的真实记录，避免仅凭
PR 摘要判断资产可用。新反应物生成、重新运行物理认证和同预算比较仍须分别执行；
这个入口通过不代表 #131 或 #137 已全部完成。

## Pinned assets and interface handoffs

| PR | Exact head | Available interface | Handoff and remaining work |
|---|---|---|---|
| #111 | `9cc15e03672264fddda0804bb8c2cb4eb31da8f0` | `v1/clocks.py`, `v1/sampler.py`, three generator hashes | #133/#136: use the research model API; main's `JointEventGeometryFlow` is a different implementation. Rehash weights on n2 before loading. |
| #113 | `3db499ca1ae956dc572ee87439fe654b0192d8f3` | frozen enumeration analysis and raw-file manifest | #138: retain candidate-axis results; add full search/physics cost separately. |
| #114 | `98720d8e358904f72b76df0a40b81e969769cabd` | oracle configuration, acquisition identities, ranking evidence | #135/#136: ranking is not TS certification; AIMNet training overlap is unresolved. |
| #119 | `e84e330390a519bf4ac12048b297403c09717646` | O2 analysis, DFT protocol, archived verdicts and dispatcher ledgers | #132/#137/#139/#142: replay first, then adapt live calls to shared hard budgets and explicit alternate-channel records. |
| #120 | `b6db3df15eefd85a64eed329674a48d3ecba301e` | frozen thermodynamic enumeration and training threshold | #138: preserve domain/coverage limitations and separate complete-cost evaluation. |

`main` at `d1aee8f7e18bfa30fd53308064f55b2a34fa42b7` supplies the stable
`data/records.py`, `data/task_views.py`, `data/track_b.py`, `runtime/ledger.py`,
`models/joint_flow.py`, and `validation/ts_search.py` contracts. Research
`m0/` and `v1/` modules are not on main. A source checkout and a weight file
with a matching hash establish availability only; they do not establish
API compatibility or chemical accuracy.

## Replay contract

`scripts/replay_research_o2.py` accepts an exact, clean PR #119 checkout,
the n2 runs root, and a new output directory. It checks all 159 entries in
the pinned O2 manifest before executing the existing analyzer. Raw paths
are relocated under the supplied runs root; escapes and missing files fail.
Neither checkout nor historical run output is edited. The output directory
must not exist. This reads real archived data and therefore runs only on n2.

The fixed cap is 600 wall seconds, 2 GiB of hashed inputs, 64 MiB output,
one CPU thread, zero GPU and zero new calculator calls. The analyzer has
600 CPU seconds and 16 MiB per-file OS limits, and a wall timeout covering
the remaining stage budget. Its two outputs are `analysis.log` and
`results.json`. `report.json` records status, measured CPU/wall time, hashes,
environment, source revisions and candidate links. Hash failures, unavailable
inputs, execution failures, analysis timeouts and comparison failures remain
distinct. Keep every failed output directory. Report-write failures surface as
process errors and must be retained with scheduler stderr; they are not replay
successes. External Slurm limits bound the whole process including reporting.

All geometry is Angstrom, energy Hartree and frequency cm^-1. Charge and
multiplicity are read from the frozen plan, never inferred. The archived
method is gas-phase electronic wB97X/6-31G(d), `v1b-h-gpu4pyscf-3`; this is
not CP2K evidence. The replay imports no QC backend. It preserves the archived
strict status and the historical protocol's endpoint gray-zone rule, and
does not strengthen those into a new scientific certificate.

Candidate records distinguish the planned product matrix hash from the
observed endpoint event ID. An unavailable observed event remains null.
Geometry hashes use sorted, compact JSON with nonfinite numbers rejected;
they identify the archived values and atom order, not a new atom-map audit.
The plan, per-attempt manifest and analysis cost row must agree on item,
parent and job identity. Recorded failures retain their historical costs.

## Reproduction on n2

Set the private run/source paths locally; do not commit them. Use a new
clean execution clone at the integration commit (record its full SHA),
and keep the research checkout at the pinned #119 SHA above. After loading
`~/miniconda3/etc/profile.d/conda.sh` and activating `xtbflow`, run from the
integration clone:

```bash
git rev-parse HEAD
python --version
srun --partition=main --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=4G \
  --time=00:12:00 bash -lc '
    set -e
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate xtbflow
    cd "$XTBFlow_REPLAY_SOURCE"
    test "$(git rev-parse HEAD)" = "$XTBFlow_REPLAY_COMMIT"
    python scripts/replay_research_o2.py \
      --research-source "$XTBFlow_O2_SOURCE" \
      --runs-root "$XTBFlow_RUN_ROOT" --out "$XTBFlow_REPLAY_OUTPUT"
  '
```

Export the five `XTBFlow_*` variables before submission. `XTBFlow_O2_SOURCE`
must be a clean checkout of `e84e330390a519bf4ac12048b297403c09717646`.
The output directory must be new, outside both checkouts, with at most
64 MiB retained. Budget is zero new physical calls; historical costs stay
historical. The Slurm envelope is 12 minutes, 1 CPU, 4 GiB and no GPU.

## Evidence still required

- #131: fresh reactant-only joint/serial proposal → common physical tools →
  certification → shared budget in one candidate chain, and all other raw
  research assets/weights rehashed. Archived replay alone is insufficient.
- #132: task counts, data admission and independent grouped splits; archived
  candidate labels are validation feedback, not inference inputs.
- #137: live service, full cache identity, bounded retries/recovery and
  explicit valid-alternative semantics. The existing protocol's internal
  meter is not the shared hard reservation ledger.
- #138: separately trained strong serial/backtracking controls, equal full
  costs, paired effects and independently confirmed claims.

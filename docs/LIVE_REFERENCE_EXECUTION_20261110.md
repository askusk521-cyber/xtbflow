# Live feedback handoff to the independent reference service

The release candidate and this scientific-software integration are separate.
The MVP release is reviewed in #182, with implementation/evidence in #174 and
the one-file inventory repair in #175. This integration is #181, based on
`codex/w133-dual-clock-20261108@5544a4ae837b61cce1ef645704a36ea2c48e8f8c`.
No branch or PR was merged.

## Frozen execution

The reference execution source is
`2772c7bedb9c674144d7dab702a29178eeee699a`, in a clean named checkout on node2.
The input producer is separately pinned to corrected #179 source
`a8e4a88069ba363f8e76f8378a58795800718648`. The source context was prepared by
`cae68937c00a6bf6061b44a2cc977d6fe66a60c9`. None of these SHAs is relabeled as
the later evidence-publication commit.

| Item | Frozen value |
| --- | --- |
| Parent | `reactant-da-r0-approach-v1` |
| Seed | `20261111` |
| Arms, in order | `no_feedback`, `physical_feedback` |
| Reference configuration SHA-256 | `3a89632d2b711620c407cebc27dfee4d82738bf8fd71430a3b75fa83d721563e` |
| Live configuration SHA-256 | `6aa438af414600d97997a7c823bda0a803a90de90d9a5a1536d4e7b5c769641f` |
| Live manifest SHA-256 | `4f0393bbf88f3805bfb480531dbc2a43c046a5eca9e3c282fae2fbb493518de7` |
| Reactant context SHA-256 | `1f6685075ddd3f0bd2c1d33ade912656d0cbac2586500aea143d47210e269bea` |
| Reference protocol | `xtbflow-reference-certification/1` |
| Candidate contract | `mapped-candidate-optional-proposal/v2` |
| Attempt | n2 Slurm `4223`, one GPU, 8 CPU cores, 64 GiB, 120 minutes |

The configuration is
[`live_feedback_reference_20261110.json`](../configs/validation/live_feedback_reference_20261110.json).
Each arm has the same 3,000-primitive, 3,300-second, 128-MiB raw-artifact cap.
The combined stage is capped at 6,000 primitives, 2 GPU-hours, 16 CPU-core-hours
and 1 GiB, with no external retry. The protocol's existing internal gray-zone
and endpoint-mode recovery remains unchanged and charged within those caps.
Earlier xTB guidance and path-diagnostic costs remain in the upstream ledger.

Only each immutable `snapshot_after.geometry` enters the reference calculation.
The input has 16 mapped atoms and 64 verified reactant automorphisms. Charge,
multiplicity and the explicit gas-phase modeling environment are checked.
No product, TS, endpoint or force label is used to construct this handoff.
The original archive describes a reactant endpoint: this is development data,
not independently prepared blind input or held-out confirmation.

There is no chemical product decoder for the four-dimensional guidance scores.
`planned_product_be`, planned event ID and prediction-match outcome remain null.
The unchanged reference protocol may identify an observed channel only after
TS refinement, full final modes, bidirectional IRC, both endpoint minimum/frequency
checks and mapped reactant identity. `pipeline_completed` alone does not certify
a channel. Same endpoints, unresolved results and protocol failures stay in the
candidate denominator and costs.

## Environment and software checks

The clean execution source passed 39 tests on node2 in the `xtbflow` environment
(Python 3.10.19), and all 395 tracked files passed the bootstrap inventory check:

```bash
PYTHONPATH=src:vendor/mechai_reusable python -m pytest -q \
  tests/test_certification_service.py tests/test_live_reference_handoff.py \
  tests/test_runtime_budget.py
python scripts/validate_bootstrap.py
git diff --check
```

The `xtbflow-qc` environment used for the reference job reports Python 3.11.17,
NumPy 2.4.6, SciPy 1.17.1, PySCF 2.14.0, GPU4PySCF CUDA12x 1.8.1,
geomeTRIC 1.1.1, RDKit 2026.3.6 and CuPy CUDA12x 14.2.0. It does not have
pytest; the software tests above used the separate `xtbflow` environment.
Installed package RECORD hashes and observed GPU/CUDA identity are included in
the actual reference protocol/cache identity.

## Commands and retained data

Private raw inputs and run paths remain on n2. In an n2 shell, set
`XTB_REFERENCE_CODE` to the clean named execution checkout at `2772c7b`, and
`XTB_REFERENCE_RUN_ROOT` to the retained hash-checked run directory. It contains
`live_input/`, `live_config.json`, `reactant_context.json`, `source_freeze.json`,
the original job script/log and `reference/`. The equivalent one-shot launch
below uses the same finite allocation; the original run used a stored `sbatch`
script with these limits and invoked the same Python command through `srun`:

```bash
cd "$XTB_REFERENCE_CODE"
test "$(git rev-parse HEAD)" = 2772c7bedb9c674144d7dab702a29178eeee699a
test -z "$(git status --porcelain)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate xtbflow-qc
export PYTHONPATH=src:vendor/mechai_reusable
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
srun --partition=main --gres=gpu:pro6000:1 --nodes=1 --ntasks=1 \
  --cpus-per-task=8 --mem=64G --time=02:00:00 \
  python scripts/certify_generated_candidates.py \
    --live-feedback "$XTB_REFERENCE_RUN_ROOT/live_input" \
    --generation-config "$XTB_REFERENCE_RUN_ROOT/live_config.json" \
    --reactant-context "$XTB_REFERENCE_RUN_ROOT/reactant_context.json" \
    --config configs/validation/live_feedback_reference_20261110.json \
    --out "$XTB_REFERENCE_RUN_ROOT/reference"
```

The runner refuses to overwrite an existing output. Do not rerun this physical
command to inspect results. For read-only replay, point `XTB_REFERENCE_AUDITOR`
at the published `scripts/audit_live_reference_cache.py` file outside the exact
execution checkout, then run from that same clean checkout/environment:

```bash
cd "$XTB_REFERENCE_CODE"
python "$XTB_REFERENCE_AUDITOR" \
  --run-root "$XTB_REFERENCE_RUN_ROOT" \
  --config configs/validation/live_feedback_reference_20261110.json \
  --out "$XTB_REFERENCE_RUN_ROOT/cache-audit-new.json"
```

The audit verifies source/config/protocol/build/candidate identity, every raw
artifact hash, durable request/result/settlement hashes, primitive sequence,
resource events, reconstructed feedback, channel index and the runner's original
zero-call cache replays. It cannot call `service.run` or a calculator. Its own
driver hash is recorded separately from the original execution source. Existing
receipts are never overwritten. Cache replay is `software_demo`, not a fresh
physical replication.

## Execution outcome

Slurm job 4223 completed with exit `0:0` after 59 minutes 33 seconds. Both arms
completed the frozen TS/gradient/mode checks, bidirectional IRC and two endpoint
minimum/frequency checks. Both endpoints in each arm map to the reactant graph:
the verdict is `same_endpoint / both_ends_reactant`, with **zero unique strict
reaction channels**. A completed calculation is not a certified reaction channel.

| Arm | Reference primitives | Candidate wall seconds | Measured CPU hours | Candidate GPU allocation hours | Verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| no-feedback | 500 | 1934.972291 | 0.614920 | 0.537492 | `same_endpoint` |
| physical-feedback | 402 | 1636.052427 | 0.518358 | 0.454459 | `same_endpoint` |

The parent denominator is one; the candidate denominator is two. There are no
unfinished reference primitives or exhausted budgets. Time/cost to the first
reaction certification remain null. The difference in calls is an accounting
observation from this one fixture, not evidence of method superiority or a
chemical failure-rate estimate. The live physical feedback changed geometry and
guidance scores; this did not produce a distinct certified reaction channel.
The declared-source-endpoint input remains development data.

The service ledger retains 902 reference primitives, 1.133278 measured child
CPU-core-hours and 0.991951 candidate GPU allocation hours. The terminal
`scontrol` snapshot records 0.9925 whole-job GPU allocation hours and 7.94 reserved
CPU-core-hours (eight allocated cores times job wall time). CPU reservation is
not measured CPU utilization. The difference between whole-job wall time and
summed candidate wall time is approximately 1.98 seconds, with one-second
scheduler precision. Earlier xTB guidance and diagnostic costs remain separately
recorded; xTB energy/force calls are not added to reference primitives as one
interchangeable call unit.

Both original cache replays returned identical records, zero new primitives and
zero new channels. The separate read-only audit passed for all 38 raw artifacts,
request/result/settlement hashes, candidate identities, primitive sequences,
reconstructed verdicts and settled resource events. Its guard checks also rejected
an incomplete run and a receipt overwrite with structured errors. These replay
and software checks are `software_demo`; they are not a new physical rerun.
The complete frozen reference verdicts are `strict_validation` within this
development protocol, with no positive reaction-channel result. Earlier xTB
diagnostics remain `proxy_metric`; their withdrawn certification claims are not
restored by this downstream run.

Evidence, copied without rewriting the program-produced records:

- [Terminal report](evidence/live_reference_exact_2772c7b_20261110.json).
- [Raw/cache/ledger audit](evidence/live_reference_cache_audit_2772c7b_20261110.json).
- [Execution, costs, source assets and artifact index](evidence/live_reference_execution_receipt_2772c7b_20261110.json).
- [Exact-source environment and input preflight](evidence/live_reference_preflight_exact_2772c7b_20261110.json).
- [Audit failure and overwrite guards](evidence/live_reference_cache_guards_2772c7b_20261110.json).

The reference protocol/build SHA-256 is
`6dc84cd79c7223e6e05126d1211d70075f5401d9f5f9fc45c396ea77a2d25a39`.
The outcome identity is
`03cb95ce9e99423f621710a449cde07b5369ed6c2d6397e72052b38d3ee5414a`.
The receipt records all input/config/source/result hashes and separates the
original execution source from the later audit-driver and publication commits.

### Remaining accounting limitation

`sacct` returned `Slurm accounting storage is disabled`. Historical scheduler
accounting, whole-job measured CPU usage and maximum RSS are therefore unavailable
and remain null. The completed `scontrol` snapshot, job start/exit receipts,
original job script hash and measured service ledger are retained. No scheduler
configuration was changed, and no extra quantum calculation was used to fill this
gap. Enabling historical accounting is a separate administrator action for future
jobs; it cannot recreate missing historical measurements.

## 中文说明

两条固定几何都完成了 TS（过渡态）优化、频率、双向 IRC（反应路径）和端点检查，
但两端都映射回反应物图，因此严格反应通道数为零。baseline 和物理反馈分别消耗
500 与 402 次参考计算原语，全部费用和两个候选的分母都保留；一次样例中的费用差
不能证明方法优势。原始坐标、路径和 Hessian（二阶导数）保留在 n2，38 个原始
artifact、调用账本及判定已经完成只读重放检查，没有新增计算调用。缺失化学产物
图时仍保持 null，不虚构预测事件，也不把程序正常退出写成发现了反应。Slurm 未
启用历史资源记账，所以缺失指标明确留空；已有程序实测成本和终态调度记录仍然可查。
这些结果限于一个开发样例的冻结协议，不支持化学精度、独立泛化或方法优势结论。

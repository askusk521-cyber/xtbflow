# Narrowed study: bounded evidence-insufficient result

**S4 does not justify S5. #188 is not launched; this report follows path A of
#189.** The result is “no strict-channel advantage demonstrated”, not “useful
physical information does not exist” or “practical effects have been excluded”.
All new chemistry budgets remain zero. The work advances #183–#187 through
contracts, focused repairs, fake/cached replay and finite frozen-model diagnosis.
It does not close their remaining scientific acceptance criteria automatically.

## Scientific question, gap, hypothesis and falsifier

Within a fixed neutral closed-shell CHNO competing-event domain, with only a
mapped reactant packet and current generated state visible at inference, can
geometry-side physical information usefully change event selection while the
event remains editable? If so, does using that information during generation
increase independently connected channels at matched actual cost relative to
terminal treatment and a strong adaptive serial method?

The distinction from related work is the **timing/intervention and total-cost
question**, not the mere combination of flow models and chemistry. FlowER uses
mass/electron-conserving BE flow matching for reaction prediction
([primary paper](https://www.nature.com/articles/s41586-025-09426-9)). React-OT
generates TS structures conditioned on both reactants and products
([authors' paper](https://arxiv.org/abs/2404.13430)), a different input permission
from reactant-only event discovery. The 2026 MolGEN paper addresses product and
TS generation with a shared framework and reaction-network exploration
([primary paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC13493859/)). These adjacent
capabilities make a broad “joint generation is new/better” claim inappropriate.
Our inference from these sources is that a defensible contribution would need
the narrower causal, input-matched, full-cost comparison; no novelty or
publishability claim is established by this audit.

The proposed mechanism requires useful present-state physical information,
adequate endpoint transport, a change in decoded event/candidate geometry and
improved independent endpoint evidence. Its efficacy falsifier would be a
qualified paired study whose family-aware upper uncertainty bound excludes the
prespecified useful effect, after a fair adaptive serial comparison. Current
evidence cannot execute that falsifier. A missing gate or an unresolved
certification is not itself proof of no chemical effect.

## Frozen implementations and information

| Line | Exact identity | Evidence and allowed claim |
| --- | --- | --- |
| Trained science | research e84e330; joint EMA step 60000 seed 0, checkpoint 570bc0d1… | Real frozen BE+x model; dual-time architecture and original decoder |
| First new diagnostic | ID63623 / dft-da:R0, sampling seed 0, three states, S0 packet hashes | Exposed reactant-endpoint calibration; no fresh discovery/generalization |
| Modern runtime/rule lanes | producer a8e4a88; S2 fixes in #191/#192 | Rule scores are not decoded chemical events |
| Independent reference chain | execution 2772c7b, original auditor/evidence 721a9f2 | Frozen TS/mode/IRC/endpoint protocol; old hashes/verdicts preserved |
| MVP/release | #182/#175 at 6b38af8 | Software/inventory only; not the trained scientific line |

Full SHAs, all 87 remote heads, 34 source/evidence file hashes, checkpoint/status
hashes, input mapping/electronic state and exposure history are in
[S0](NARROWED_SCOPE_20261111.md) and its machine-readable manifests. The source
reactant endpoint's original environment and cross-dataset overlap remain
unresolved. A new gas-phase modeling condition is explicitly declared, not
inferred from hidden reference labels. Product/TS/event/energy targets and
future commitment are never inference-visible. ReactionQM remains quarantined.

The 62-parent geoinfo/clock/pg-1/SMC pool, 342-parent unblinded V1a formal pool,
nine-parent DA diagnostic pool and known calibration candidates are exposed.
The former five-parent W08 plan had no completed frozen panel and cannot be
relabelled as sealed. No verified eligible unexposed pool currently exists.

## What the new evidence establishes

1. **Amplitude confounding is real.** All 27 retained paired steps reproduce a
   240.74× median displacement difference; standard steps improve energy/force
   RMS and normalized steps worsen them. This supports overshoot without proving
   a sign error or better saddle/channel behavior.
2. **Endpoint transport is not an identity map.** Three frozen trained probes
   give gain .411–.497 and directional cosine .155–.534. FD/JVP agree to ≤1.51e-6
   relative error; BE symmetry/conservation and centering hold. Event response
   occurs on the next forward. These numerical directions are not physical
   forces, and the probed intermediate BE states do not establish a decoded
   final chemical change. No J-transpose saddle “repair” is promoted.
3. **Demonstrated identity and accounting defects are repaired in draft.**
   SO(3) parity/current snapshot checks use labelled and recording-fake tests.
   Started backend failures retain costs; unknown measurement/termination stays
   explicit and blocks further work. None of these fixes is causally linked to
   the old zero-channel outcomes by this study.
4. **Cached trained chemical handoff is verified.** Both historical trained
   candidates exactly match archived reference input, including decoded BE and
   mapped geometry. This is exposed calibration, not a new live feedback loop.
5. **The old information result remains inconclusive.** Matched G−N ΔSpearman
   is +.02714 with 95% interval [-.11096,+.16525]. The .70 xTB information time
   leaves 14.546% of valid-final trajectories uncommitted under an offline rule.
   All 5,952 attempts remain represented: 1,016 invalid, 2,600 outside catalogue,
   876 known-event geometry misses and 1,460 proxy matches. Catalogue labels are
   available for only 2,336 trajectories; they are not strict channel labels.

See [S1 evidence](../evidence/narrowed_scope_20261111/numerical_diagnostics.md),
[S4 evidence and reproducible figure](../evidence/narrowed_scope_20261111/incremental_signal.md),
[S3 raw-stage/cached bridge audit](https://github.com/askusk521-cyber/xtbflow/blob/99c82f2c137eac464a5788e13cdf826273170fdc/docs/evidence/narrowed_scope_20261111/failure_audit.md),
and the [claim–evidence manifest](../../configs/research/narrowed_claims_20261111.json).

## Failures, resources and compatible effects

The new frozen-model wrapper initially failed on clock-array conversion after
50 completed forwards and one started forward. Its original receipt/tensors
remain intact. A separately registered recovery used those exact saved states,
not a replacement parent or regeneration, within the original total budget.
Totals are **90 started trained-model forwards**, 89 completed, four JVP attempts,
six completed reverse passes/eight conservatively reserved, 2.903416 measured
process CPU seconds and 3.131729 measured diagnostic wall seconds. All new
training, GPU, xTB, HVP/Hessian and reference primitives are zero. Software test
fixtures and read-only audits have separate receipts and are not new chemical
samples. No stage budget was silently increased.

The first cached bridge audit also stopped on an incorrect protocol selection
in the audit wrapper. Its failure is retained; the corrected version uses the
hash-verified historical handoff protocol. The original reference cache auditor
still passes from exact source 2772c7b with zero new calls.

Historical outcomes and costs remain: two rule `same_endpoint` attempts at
500/402 primitives; trained joint unresolved at 468 (endpoint bond perception),
trained serial connected at 638, and trained feedback unresolved at 526
(frequency gray-zone cap). Same_endpoint is distinct from protocol failure;
unresolved does not mean chemical nonexistence. Mode/endpoint differences are
quantified without assigning an unobserved torsional mechanism. Program timing
is retained; disabled historical Slurm accounting is not filled with estimates.

The catalogue interval permits harm and moderate improvement. For the main
strict-channel endpoint, **no causal effect estimate or confidence interval is
available**. Logical probability-difference bounds [-1,+1] are only support
bounds. The planned useful-effect threshold (+0.10 probability of at least one
strict connection per parent) is neither demonstrated nor excluded. Seeds,
trajectories and cache replays are not independent chemical systems.
The old formula clusters are uncertainty-accounting groups, not verified
reaction-family independence; no reaction-family extrapolation is claimed.

## Gate, recommendation and review boundary

S4 supplies no qualified exact-endpoint incremental signal, validated conditional
shuffle or new discriminating pilot. Full live trained chemical intervention,
independently prepared/provenance-qualified inputs and a strong adaptive serial
comparator remain open. S5 is therefore **not_launched_gate_not_met**, with zero
new attempts/cost. S6 completion here means bounded synthesis, not independent
confirmation or a finished efficacy paper.

Stop promotion of the current hypothesis at this evidence boundary. A future
continuation must first document a specific remaining causal question and a
frozen protocol: legal independent inputs, compatible current-state physical
signal, decoded final event/geometry/endpoint chain, restart/reproposal/backtrack
serial baseline, and all actual model/physics/reference costs. Data acquisition
must independently audit provenance, license and overlap; no exposed pool is
reused as confirmation. This report does not schedule that work or train a new
architecture/correction potential to obtain a positive result.

Draft deliverables are #190 (S0), #194 (S1), #191/#192 (S2), #193/#195 (S3) and
this S4/S6 synthesis. The review order is main→#190→#194→synthesis;
#179→#191→#193; old PVO summary branch→#192; #181→#195. These separate
implementation lineages have **not** been merged or silently integrated.
Bootstrap inventories can conflict during future stack integration and require
normal review/refresh, not bypassing the human merge gate.

The [current-main CI snapshot](../evidence/narrowed_scope_20261111/main_ci_snapshot.json)
records successful Python 3.10/3.11 checks for
`ac36394571a48491cbbfe860653bc604532b0e88`. Exact-source n2 tests for the new
branches are recorded separately; main CI is not their CI. Stacked PR workflows
may not trigger. No PR/issue is automatically merged/closed, and no external
paper submission, deployment or confirmation computation is authorized by this
writing. Future merging requires human review and refreshed exact main/head CI.

中文结论：本轮完成了合同冻结、有限数值诊断、身份与失败记账修复，以及旧证据只读
审计。已发现几何到事件的局部数值响应，但没有证明流内物理反馈带来严格通道收益。
因此不启动 #188，按 #189 保留有边界的证据不足结论、全部历史结果、失败费用和
原始哈希。尚未验证的输入条件、化学干预链、强串联对照和独立数据继续明确列为缺口。

# V1a v2.1 execution ledger

## Formal screen result (2026-10-07) — supersedes the development hold below

Decision **`GO_V1b`** from the single frozen analysis (`docs/evidence/v1a/formal/`).
Before any screen output, the project lead chose to freeze on the unguarded
development power estimate (`docs/v1a/FREEZE_DECISION_ZH.md`); the pre-written
plan rule gave N=342 parents / G=70 formula groups (freeze
`afc5f55d4592d5da972deca9fa4fe0126b65bbb9b9ef0a62831f120a1fa94ea6`).

| Quantity (3 training seeds, sampling seed 0) | Estimate | One-sided 95% L / U |
|---|---:|---:|
| Primary AUC B1−A2 | +0.0931 | +0.0723 / +0.1139 |
| M0 = U(F_S)−U(F0) | +0.0212 | +0.0178 / +0.0246 |
| MR = U(F_S)−U(F_R) | +0.0211 | +0.0179 / +0.0243 |

Interpretation limits that must travel with the decision: unguided B0 has the
highest AUC (0.4945) and AUC B1−B0 is −0.229, because guidance costs leave B1
far fewer completed candidates; A0 (0.382) also exceeds B1 (0.265). The GO is
B1 versus the adaptive cascade A2 only, on a finite T1x catalogue proxy
(`PROXY_MATCH` is not a certified TS); no quantum calculation was run. The
gain concentrates in common/intermediate best-event strata; rare and
pilot-unseen strata are small (+0.020 / +0.018).

Jobs 2606 (plan), 2607 (freeze), 2608/2614–2619 (screen, report) ran from
`xtbflow-dev` at `21ec44e`/`966ee5b`; 2608's pulse step failed on a CLI
argument before producing output and was rerun as 2614 after the fix. The
V1b source population (1,710 bundles, budget 16, seed 0) and all raw
candidates stay in the n2 run root; hashes are in `SHA256SUMS`.

## Final development-stage disposition (2026-10-07)

`HOLD_POWER` before formal screening. All 10 generator trainings and all six
score members completed. Five-arm development, three-seed nuisance measurement,
paired pulses, real continuous replay controls and the full power scenarios are
complete. No screen method outcomes or quantum calculations were produced.
The final scientific implementation was verified at `5aa6f96`: 341 pytest tests,
68 subtests, 56 reusable unittest tests, and the bootstrap inventory pass.

The three-seed mean parent AUC difference has SD 0.21708 and estimated formula
ICC 0.15948. At the maximum reserved inventory (344 parents / 72 groups), target
joint GO power is 0.81485 under estimated Gaussian nuisance, 0.81925 under the
empirical-skew working model. The planning guard (SD x1.10, nonnegative ICC +0.05)
reduces these to 0.72785 / 0.73635. These numeric guard increments are analyst
choices specified before the first power scan, not literal guide thresholds.
The target effects and final statistical gate were not changed. The hold means
the planning decision is sensitive to nuisance uncertainty, not that 80% power is
mathematically impossible or that H1 has failed.

Authoritative final planning artifacts in the run root are
`evidence/power_plan_final.json`, `power_limits_final.json`,
`resource_forecast_N300.json`, `continuous_controls.json`, and
`selected_pulse_s{0,1,2}_final.json`. Complete raw candidates and immutable source
snapshots remain on n2. The user delivery includes the report, decision, evidence
archive and Git bundle. Formal freeze/run/report/export wrappers are not claimed
complete: those stages were not entered following this supported pre-screen hold.

Objective: complete the provided V1a protocol, including a supported GO/HOLD/NO-GO
decision and all applicable evidence. No V1b quantum work is authorized by this run.
The protocol is preserved in `protocol-v2.1.md`; hypothetical simulation outputs
are kept separate from experimental results.

## Source and isolation

- Base: M0 stabilized snapshot `335dd9b8b95a8680520f9c8afd8fb28c2bae56ed`.
- Local branch: `codex/v1a-20261007`; existing M0 checkouts are not modified.
- Execution root: `/home/lhshen/xtbflow-runs/v1a-20261007`.
- Source cache: `/home/lhshen/data/t1x/t1x_m0_v1.npz`, SHA-256
  `7eb681a252a93fc5f2627edbb4cd371385b7f5a29b3d71ae3b7fda45a5fdf0f1`.
- Existing M0 weights are not eligible for the new holdout. All roles are retrained.
- Official validation/test formulas remain excluded. Earlier M0 diagnostics and
  architecture choices are background development exposure, not new confirmation.
- Reserve selection uses only catalogue inventory and a fixed hash order. Formal
  screen N and the gate power plan remain unfrozen until development is complete.
- Screen cap 12 balances retained training data and independent groups; development
  cap 4 obtains more groups for variance estimation. All parents of a reserved
  formula, including those outside the cap, are excluded from generator/score training.
- The actual grouped holdout removes substantially more training data than the
  guide's rough one-third estimate. Report this and compare B0 only to B_M0re
  trained on exactly the same reduced training set.

## Requirements and evidence still to produce

| Stage | Required evidence | Status |
|---|---|---|
| 0 | Environment/assets/provenance, grouped counts, cost calibration, resource forecast | Development audit complete; inherited raw-HDF5 limitation retained |
| 1 | Catalogue, split/anchor identities, input firewall, joint matcher and adversarial tests | Built/reproduced on n2; adversarial tests pass |
| 2 | 3 g/h/f seeds, B_M0re, 3 h_E/h_X members, real rollout calibration, baseline check | Complete; baseline and real rollout score checks pass |
| 3A | Shared guidance, input gradients, full replay, RNG invariance, drift check | Real three-seed controls and drift checks pass |
| 3B | Five development windows; paired three-amplitude pulses; M0/MR, controls | Both paths scanned; selected sync t=0.50 audited across three seeds |
| 4 | A2 first4/later8 adaptive search; all five logical cost streams and failures; rarity | Development streams complete; formal streams/rarity not entered |
| 5 | Development nuisance estimates; complete-gate power and null scenarios; frozen N/G | Complete planning assessment: conservative HOLD_POWER; no formal N/G frozen |
| 6 | Single final analysis, integrity audit, decision, V1b source export and report | Pre-screen hold report delivered; formal analysis/export not applicable |

Do not mark any stage complete from a smoke test alone. If a protocol HOLD occurs,
perform its allowed concrete repair and document the remaining uncertainty. Never
convert missing runs, infrastructure interruptions or unresolved mappings to negatives.

## Verified first training batch

Remote data: 1,110 parents / 8,520 references; official train has 994 K>=2 parents
in 147 formula groups. No exclusions occurred in the new mapping audit. The selected
split has training 386 parents / 62 groups / 2,603 records, development 62 parents /
21 groups, and screen reserve 344 parents / 72 groups. The latter is NOT the final N.
Split hash: `a0225e6cb773049d94593f0d93bf6c753db125deeda3ade8899da6922835367f`.

Seed-0 generator source is immutable `d20321facf804a7dea8341801935446c34ee5738`.
Baseline, joint, event and geometry each completed 60,000 steps; loop elapsed times
were respectively 1511.36, 1577.00, 1476.85 and 1427.46 seconds. Training checkpoints
use final-60,000 selection throughout. The same source is used for seeds 1 and 2.
Scores use immutable `a9cca4d`; baseline generation uses `775c85c`.

Development baseline, 62 parents x 8 proposals (seed 0):

| Model/path | Any-reference joint recall | Best-reference recall | Valid rate |
|---|---:|---:|---:|
| B_M0re / sync | 0.790323 | 0.290323 | 0.852823 |
| B0 / sync | 0.774194 | 0.306452 | 0.838710 |
| B0 / event_lead2 | 0.806452 | 0.370968 | 0.864919 |

The synchronized B0 drops 0.016129 in joint recall and 0.014113 in validity, below
the protocol's 0.05 / 0.10 diagnostic limits: `BASELINE_DEVELOPMENT_PASS`.
This is neither an equivalence test nor the formal B1-A2 outcome. Initial window
development used event_lead2; subsequent sync calibration retained a wider event
response window at t=0.50, so sync was selected for the completed power assessment.
Authoritative evidence is `evidence/baseline.json` and the sealed source manifest
under `development/baseline` in the execution root.

CPU verification on n2 at the initial training snapshot: 313 pytest tests, 68 subtests,
and 56 reusable unittest tests pass. Later focused score/replay/cost/statistics tests
also pass. Final development implementation verification is recorded above.
The batch-64 cost measurement remains a preflight; completed common-batch-512
weights were used throughout the development cost streams. No formal freeze was
created following the power hold.

## 中文说明

本轮按 V1a v2.1 完成代理筛查，所有正式结果都要有冻结清单和真实日志支持。
数据按分子式分开，已有 M0 权重不能作为未见这些母体的新模型直接使用。
正式人数和功效计划必须等待开发结果，不能把附带的假设模拟结果当实测证据。

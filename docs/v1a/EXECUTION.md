# V1a v2.1 execution ledger

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
| 0 | Environment/assets/provenance, grouped counts, cost calibration, resource forecast | In progress |
| 1 | Catalogue, split/anchor identities, input firewall, joint matcher and adversarial tests | Implemented; remote reproduction pending |
| 2 | 3 g/h/f seeds, B_M0re, 3 h_E/h_X members, real rollout calibration, baseline check | Implementing |
| 3A | Shared guidance, input gradients, full replay, RNG invariance, drift check | Pending |
| 3B | Five development windows; paired three-amplitude pulses; M0/MR, controls | Pending |
| 4 | A2 first4/later8 adaptive search; all five logical cost streams and failures; rarity | Pending |
| 5 | Development nuisance estimates; complete-gate power and null scenarios; frozen N/G | Pending |
| 6 | Single final analysis, integrity audit, decision, V1b source export and report | Pending |

Do not mark any stage complete from a smoke test alone. If a protocol HOLD occurs,
perform its allowed concrete repair and document the remaining uncertainty. Never
convert missing runs, infrastructure interruptions or unresolved mappings to negatives.

## 中文说明

本轮按 V1a v2.1 完成代理筛查，所有正式结果都要有冻结清单和真实日志支持。
数据按分子式分开，已有 M0 权重不能作为未见这些母体的新模型直接使用。
正式人数和功效计划必须等待开发结果，不能把附带的假设模拟结果当实测证据。

# M0 diagnostic round handoff — experiments remain stopped

## English summary
The owner authorized diagnosis with “诊断”, then requested closing this round and publishing detailed progress for other contributors. Four independent bounded Slurm replays reproduced all original failed steps. All failures involve training cache index 7451, with finite inputs and parameters, but explosive forward activations. Three fail at squared-error overflow; joint seed 2 first overflows the EventHead scalar pair product. No official data/model/hyperparameter/evaluation change was made. Formal training remains stopped; test sampling has not started. This is not a GO/NO-GO decision or a completed M0 result.

## 中文说明
本轮完成了四个失败运行的独立诊断，均在原来的训练步数复现异常，证据已经保存供其他人审查。问题是网络内部数值逐层放大，最终超过浮点数能表示的范围，并不是已经发现输入含有坏数值。样本 7451 的特殊氢原子构型确实来自官方数据，不能未经审批删除样本或替换 TS 标签。正式训练和测试仍然暂停，下一位执行者应先审阅证据、提出具体修复方案并获得负责人批准。

## Deliverable/evidence status
| Requirement | Current evidence/status | Remaining work |
|---|---|---|
| Separate task branches/PRs | #101 task0, #103 task1, #102 task2, #104 task3, #105 task4; unmerged at prior handoff; #104 OPEN verified at publication | Recheck all PR heads/state and stacked-branch divergence; human merge approval |
| CPU/GPU environment and implementation | Prior task evidence and tests; see docs/evidence/m0 in task branches | Exhaustive guide-to-artifact audit still required; prescribed pytest PYTHONPATH discrepancy disclosed previously |
| Official data/cache | Official HDF5 hash 6a20f8a3f49c50d462270d10d4c44ca102e788072e2096a91d70b5a0f598b629; cache hash 7eb681a252a93fc5f2627edbb4cd371385b7f5a29b3d71ae3b7fda45a5fdf0f1 | Prior approval covers retention 84.58% and 35 test parents only, not numerical failure waiver |
| Validation noise selection | noise_selection.json: joint (1.0,0.5), cascade (0.5,0.5) | No test-based tuning |
| Three-seed 60k formal training | numerical_stop.json; joint_s1/event_s0/event_s1/geometry_s0 previously completed | Failed seeds and canceled runs prevent completion; no resumed formal runs |
| Diagnostic round | docs/evidence/m0/diagnostics/DIAGNOSIS.md and files listed below | Concrete stabilization proposal and owner approval |
| Frozen selection / one-time test / real gate report | NOT COMPLETE; test sampling not started | Validation checkpoint selection, committed freeze, one-time test, paired-parent bootstrap and real gate report |
| Resource accounting / completion audit | Diagnostic loop timings below; sacct unavailable because accounting storage disabled | Include failed/retry/diagnostic allocations in conservative resource ledger, check 12h/run and 150h total; full requirement audit |

## Exact replay results
| Job | Original run | Reproduced step | Bad sample / t | First overflow | Loop seconds |
|---|---|---:|---|---|---:|
| 2547 | joint_s0 | 7559 | 7451 / 0.8604222536 | residual square; both losses inf | 288.85 |
| 2548 | joint_s2 | 10893 | 7451 / 0.8604839444 | EventHead `si * sj`, then Linear NaN | 414.73 |
| 2549 | geometry_s1 | 11447 | 7451 / 0.9130203724 | residual square; geometry loss inf | 395.47 |
| 2550 | geometry_s2 | 10095 | 7451 / 0.9581016898 | residual square; geometry loss inf | 352.42 |

Each job used one GPU. Limits: #2547 10 minutes; remaining jobs 12 minutes each. Total requested upper bound for these four diagnostics: 46 GPU-minutes (0.767 GPU-hours), not actual billed allocation time. Prior GPU work is additional. Replays performed diagnostic-only parameter updates from initialization to reproduce failed states; they did not resume formal runs or produce candidate checkpoints. All four finished before the final handoff. No further jobs started after owner closed the round.

## Evidence map
Root: `docs/evidence/m0/diagnostics/`.
- `DIAGNOSIS.md`: detailed findings and limits.
- `artifact_manifest.json`: SHA-256/size for every retained diagnostic artifact, including excluded weights.
- `diagnose_7451.py`, `initial-forward-7451.json`: CPU initial-weight sweep.
- `replay_joint0.py`, `replay_failures.py`: bounded replay logic against remote formal source `b8d135d26518021f6dc7aa72a5a7a7f0e7a96857`.
- `replay-*/result.json`: exact failure, layer/module trace, finite input/parameter checks.
- `replay-*/sample_visits.jsonl`: sample-specific prior visits (may contain nonstandard JSON NaN/Infinity; parse with Python json, not strict JSON tools).
- `replay-2547/loss_operator_check.json`: float32 overflow versus enormous finite float64 loss; float64 is not a fix.
- `replay-2547/update_term_localization.json`: joint_s0 amplification localized to zero-based H atom 7 and multiplicative PaiNN updates; other samples moderate.
- `replay-2547/atom7_geometry.json`, `raw_h5_provenance.json`, `raw_trajectory_check.json`: official `/train/C6H14O/rxn10025`, cache agreement <2.02e-7 Å, named TS exact trajectory frame 3912, nearest atom to H7 8.509 Å.
- `replay-joint_s2/head_product_check.json`: finite trunk outputs but nonfinite scalar pair product.

Model weights (`diagnostic_state.pt`) are deliberately NOT committed, per AGENTS.md. They remain in `/home/lhshen/xtbflow-diagnostics/{replay-2547,replay-joint_s2,replay-geometry_s1,replay-geometry_s2}/` and n2 `/home/lhshen/xtbflow-runs/m0/diagnostics-owner-approved/{joint0-,replay-joint_s2,replay-geometry_s1,replay-geometry_s2}/`. These are diagnostic-only snapshots, never eligible model-selection candidates.

## Reproduction notes and limits
Use the recorded formal source, original `run_meta.json`, official cache, and n2 Conda `xtbflow`. Run GPU diagnostics only inside bounded Slurm allocations. Set `PYTHONPATH=<source>/src:<source>/scripts`; invoke the published replay script with `--source <source> --out <new-unique-directory>`, plus `--run joint_s2|geometry_s1|geometry_s2` for replay_failures.py. Both scripts protect output with mkdir(exist_ok=False); original formal run files must remain read-only. replay_joint0.py is hardcoded for the original joint seed 0 and stops at 7559; replay_failures.py uses the corresponding fixed bounds. The initial script's remote directory is `joint0-` because the submitting shell expanded SLURM_JOB_ID too early; no artifact was overwritten. Formal num_workers=4 is preserved. CPU confirmation checks are evidence artifacts; some ad hoc operator/provenance commands were run inline and are not standalone scripts.

This round did not rerun all software tests and did not qualify chemistry. SHA-256 matches were checked for backbone.py, flow.py, model.py, numerics.py between local task3 and remote formal source. Source TS validity remains unresolved; replacing the named TS with trajectory argmax is not justified by this inspection.

## Next contributor: required decision boundary
1. Read guide mandatory stops, AGENTS.md, numerical_stop.json and this diagnostic evidence before editing.
2. Propose an explicit common-arm stabilization change targeting scalar/vector multiplicative amplification; explain interface, parameter-count, equivariance and prescribed-code implications. Do not silently clamp losses or delete sample 7451.
3. Obtain owner approval for the exact code/protocol change and rerun scope BEFORE implementation/resumption. Diagnosis authorization was not approval for a fix or stop waiver.
4. After approval: independent branch/run directories, CPU numerical/shape/equivariance/gradient tests, bounded GPU diagnostic verification, then fair formal reruns with all failures and costs retained.
5. Audit upstream/downstream source divergence before any committed freeze. Task4 branch does not automatically contain newer task3 evidence.
6. Complete validation-only selections and commit freeze before test sampling. No post-test tuning. Produce real gate evidence and exhaustive completion audit, then seek owner confirmation.

Do not merge any PR without explicit human approval. Do not mark the goal complete while formal training, freeze, test, report, accounting or audit remains missing.

# T2a numerical and evidence audit

## Requirement-to-artifact map
- Frozen definition/config: configs/explore/u_diagnostic.json at ef3690e, before physical production; clean acquisition source-a861037.
- 558 unrelaxed dev TS, anchored x_ts/b_p: followup_u_acquire.py reads exact check_a_rows reference IDs, verifies 558 unique IDs and development split membership. Event direction imports existing event_direction; no relaxed coordinates used.
- GFN2 analytical gradient finite differences: imported explore_geometry_information._force_job and gradient_to_force; columns from ±0.005 Å; explicit symmetry. AIMNet2 forces use documented eV/Å conversion and same difference rule.
- Rigid projection: followup_u_math.internal_basis obtains orthogonal complement by SVD, three translations and three rotations. Cartesian overlap uses lowest internal eigenvector. Mass-weighted frequencies use isotope masses specified in config and sqrt(mass)-weighted rigid basis.
- Physical water: validation/water.json records optimized coordinates, full Hessian, residual gradient, six removed modes and three positive frequencies. Synthetic tests alone were not accepted as this evidence.
- Raw outputs: u-gfn2/rows.jsonl (558 attempted,557 success), u-aimnet/rows.jsonl (558 success); 557/558 NPZ Hessians retained remotely. u-gfn2-hessians.SHA256SUMS and u-aimnet-hessians.SHA256SUMS hash all NPZ files.
- Failure: GFN2 C6H12O/rxn9718, gradient SCF failure after12 gradient calls. Not retried or imputed; all proportions use557 successful records and report this denominator.
- Statistics: scripts/followup_u_analyze.py includes frozen reading boundaries, event-size strata, per-parent averages then cluster_summary(split_group), Wilson intervals for proportions, all primary/secondary outputs in results.json. Three analysis tests cover boundaries, Wilson, coverage and parent-versus-record weighting.
- Deterministic replay: u-evidence/results.json and u-evidence-repeat/results.json byte-identical. Original raw dynamics timers are NOT reconstructed by analysis.
- Test surface: source-a188a61 full pytest336/reusable56; later source-8b007e1 full pytest341/reusable56/bootstrap536. These test counts are version-specific, not numerical re-acquisition proof.
- Resource: physical water+GFN2+AIMNet2 /usr/bin/time user+system =666.81 CPU seconds (0.185225 core-hours), including initialization; this is measured physical acquisition, excludes test/analysis commands. No GPU used.

## Remaining verification before claiming fully accepted
Independent10-record physical re-acquisition has not been performed by this agent; the raw Hessians enable independent numerical diagonalization but cannot independently establish gradient correctness. Need verify raw NPZ hashes and recompute stored metrics directly from NPZ, preserve exact commands and final evidence manifest. The handoff author plans a separate ten-record physical acceptance check. Full run-level clean-source and pushed-config chronology should accompany final cross-task audit. No completion is claimed from green tests alone.

## 中文说明
本文件把每项诊断要求对应到实际代码、原始矩阵和测试记录，没有把合成测试当成真实水分子的物理验证。GFN2 有一条梯度计算失败，报告明确使用成功的 557 条作为比例分母，没有补值或重试。原始 Hessian 矩阵全部保留并生成了哈希清单，但独立物理抽查及最终跨任务审计尚未完成，因此不能据此宣布整个交接已经验收通过。

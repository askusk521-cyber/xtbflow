# T1 final audit: unresolved implementation conformance

This is an audit finding, not a new scientific result. The goal remains active.

## Concrete requirement and observed implementation
Handoff section 5.2 T1b-2 requires one ETKDG embedding attempt, followed on failure by one useRandomCoords=True attempt. In source-dc47ab9/source-1b1afac scripts/followup_thermo.py, product_geometry() retries when EmbedMolecule returns a nonzero code, but does not catch RuntimeError thrown by the first EmbedMolecule call. Such exceptions propagate to single_energy(), which records a failed product without executing the prescribed embedding fallback.

The existing T1 results contain 74 product RuntimeError:Invariant Violation failures (BoundsMatrixBuilder.cpp, upper bound not greater than lower bound). These records do not preserve whether the exception arose on the first or fallback embedding attempt. Therefore it is not currently verified that all 74 followed the required fallback rule. Successful score byte replay and the passing bootstrap/tests do NOT prove this requirement.

## Boundary and next action
Do not silently edit existing acquisition code, overwrite raw rows/results, or retry screen events under the original version. The handoff permits only additive files and explicitly requires owner handling when existing definitions/files need change; result-dependent corrections require a new version with both results retained. No correction or retry has been performed.

First permissible verification: inspect the existing per-event failure records and the exact RDKit exception paths; without any scientific re-acquisition, determine whether the logs identify first/fallback attempt. If unresolved, owner decision is needed on an additive explicitly versioned fallback-conformance correction and limited re-acquisition, preserving all v1 results and freezing the correction before execution. Worst-case scope is at least the affected failures plus any training failures of the same kind; training threshold could change, so do NOT assume only 74 screen scores need updating. Re-evaluate training tau provenance before any proposed v2 screen analysis.

Options: (1) accept the existing experiment only as a documented protocol deviation, not full handoff acceptance; (2) authorize an additive v2 correction and training/screen impact audit within the original remaining resource budget. No option has been selected. T2 remains independent.

## 中文说明
最终审计发现，原嵌入程序在返回失败代码时会重试，但抛出异常时可能直接结束，不能确认所有事件都执行了交接规定的一次备用嵌入。已有结果里有 74 条此类异常，而且原始记录没有标明异常发生在第一次还是备用尝试，所以测试通过和结果重放不能证明这一要求已经满足。当前没有修改原脚本或重算数据，需要先保留这个差异，再由负责人决定是否批准独立的新版本修正；训练阈值也可能受同类问题影响，不能只补算这 74 条就宣称完成。

Coding-Agent: pi
pi-Version: 0.99.2
Model: gpt-6-astra
Reasoning-Effort: off

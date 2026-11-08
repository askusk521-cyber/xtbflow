# W10 离线预算调度合同

`xtbflow.runtime.scheduler.BudgetScheduler` 是 W10 的软件增量。它只在一个冻结候选池上选择下一项动作并记录结算结果，不调用 xTB、CP2K 或其他计算器，也不把调度结果当成化学证据。固定策略和启发式策略必须使用同一候选池、同一成本单位和同一预算，之后才能比较策略本身。

每个 `BudgetCandidate` 需要 `candidate_id`、`family_id`、`action_kind` 和正的 `estimated_cost_units`。可选的 `channel_key` 只用于去重；它表示调用方已经冻结的通道身份，不由调度器从几何或标签推断。`information_gain` 是输入侧的启发式提示，不能由结果反填，也不能当作准确率、能量或认证标签。候选元数据必须是有限的 JSON 数据。

`fixed` 按候选池原顺序选择，遇到估算成本超过剩余预算时停止；`heuristic` 在能放入剩余预算的候选中按 `information_gain / estimated_cost_units` 选择，随后按较小 `priority` 和候选 ID 稳定消歧。两种策略都自动把已接受 `channel_key` 的后续候选记成零成本 `duplicate`，保留重复尝试的分母和停止原因。

调用方通过 `next_action()` 取得一个动作，执行后必须调用 `settle()`。实际 `charged_cost_units` 由执行器提供，失败、超时、取消和未解析都要求 `reason`，并且同样计入已消耗预算。成功记录可带实际 `channel_key`；重复、失败和预算不足不会产生严格认证结论。`stop()` 会保留未结算动作，`resume()` 会返回同一个动作，`save()/load()` 以原子 JSON 快照支持恢复。

`summary()` 中的 `successful_channels`、`failure_records`、`failure_cost_units`、`duplicate_records`、`spent_units` 和 `pending_candidates` 仅表示软件账本状态。真实计算、参考协议、TS/IRC、端点和证据等级必须由后续验证服务另行记录；本模块没有独立的化学 claim。

示例：

```python
from xtbflow.runtime import BudgetCandidate, BudgetScheduler

pool = [
    BudgetCandidate("event-a", "family-1", "new_event", 3, channel_key="family-1:event-a"),
    BudgetCandidate("geom-a", "family-1", "new_geometry", 2, information_gain=1.0),
]
scheduler = BudgetScheduler(pool, budget_units=5, strategy="heuristic")
while (action := scheduler.next_action()) is not None:
    # 这里调用方执行已冻结的外部动作，并保留真实失败/费用。
    scheduler.settle(action, "unresolved", 0, reason="offline_planning_only")
```

上述示例只演示软件账本，不是量化计算或方法效果结果。候选池、协议、预算哈希和实际环境应在上游运行报告中分别保存。

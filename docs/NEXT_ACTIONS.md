# 下一个执行 agent 的首轮工作

已完成实际推送并核验远端 SHA；`main` 当前指向后续 handoff 状态提交。没有连接权限时继续做本地可执行工作，如实标注提交位置。项目权限已授予，不再要求负责人重填所有历史批准位。

## 交接验收

运行完整性脚本；确认 origin、当前分支和工作区；验证本地计算资源与所需软件。旧工程只通过真实可访问的路径导入，记录选取的代码、已有审计证据、依赖与数据哈希，保留未提交修改。P0 资产索引已写入 `docs/LEGACY_ASSET_INDEX.md`，机器可读公开清单已写入 `configs/public_data_manifest.yaml`；没有权限访问的材料不冒充已迁移。

## 第一轮方法工作

复用公开数据审计，建立真实公开子集清单和母反应分组；实现 xTBloom 适配器与独立数值检查。用同构型公开参考标签建立 GFN2 配对、小型标量校正模型，并比较裸半经验、直接学习势及物理引导／统一后处理。mechai 的守恒事件、有限水接力、任务 manifest 和未训练兼容原型已作为有来源快照放入 `vendor/mechai_reusable/`；标准库合同测试已通过，Torch 测试等待依赖环境。

最终独立反应物种子清单不阻塞这轮模块工作，但不得用公开端点诊断替代未来独立发现测试。未取得可确认电荷、自旋或其他必要状态的记录先隔离。

## 第一轮报告

给出真实 commit、数据规模与缺项、数值符合性、分组留出指标、预算和失败清单。指标为空时明确说明原因。先证明方法模块带来增益，再扩大数据和训练；实验合作不作为当前启动条件。

## 2026-09-27 n2 增量证据

- 直接 tblite GFN2 oracle 的单位适配已修复，需用新的入口 [`scripts/run_gfn2_diagnostic.py`](../scripts/run_gfn2_diagnostic.py) 重新资格化；历史证据 [`docs/evidence/gfn2_units_diagnostic_20260927.json`](evidence/gfn2_units_diagnostic_20260927.json) 已 superseded。该入口要求显式源／输入哈希和电荷／多重度，`--count` 不得超过 64，输出固定为 `diagnostic_quarantine`；本次仅补脚本和文档，不运行真实计算。
- 历史联合流证据完成 8 步诊断训练、权重保存、恢复和 4 步采样；恢复输出逐位一致，守恒残差小于 `6e-17`。该历史运行使用合成零速度目标，仅证明当时的软件链路，不代表当前非零目标 smoke，也不构成科学性能结果。
- 运行账本新增原子化 calculator-call reservation，可在 TS 搜索器调用前拒绝超预算计划。

## 2026-09-27 provenance 与联合流增量

- n2 已完成 UniTS-Lib 首批 64 条记录的可复现 provenance audit；64/64 条通过坐标、能量、力、原子清单、反应位点、RDKit 原子数和图特征结构检查，图特征中的全局电荷／多重度编码与原始字段一致。证据见 [`docs/evidence/units_provenance_audit_20260927.json`](evidence/units_provenance_audit_20260927.json)。
- 原始 object array 没有逐记录声明坐标、能量和力单位；B3LYP-D3(BJ)/def2-SVP 只能由 UniTS-Gen 论文及其 SI 作为数据集级方法证据追溯。因此该切片结论为 `diagnostic_only`，公开清单准入数继续保持 0，不生成带有假定单位的正式 E/F 训练缓存。
- 联合流 v2 烟测增加了真正独立的保存前／恢复后前向比较：事件速度、几何速度和两类消息的最大绝对差均为 0；4 步采样有限，守恒残差最大 `6.94e-18`。证据见 [`docs/evidence/joint_flow_smoke_20260927_v2.json`](evidence/joint_flow_smoke_20260927_v2.json)。
- 将当前源码同步到 n2 后，完整 pytest 为 `135 passed`、`68 subtests passed`；唯一输出是既有的测试代码 tensor-to-scalar 警告。下一项区分力最大的工作是继续追踪 UniTS 原始单位声明或转向明确单位的公开 E/F 来源，再运行 Delta／direct 留出对照。

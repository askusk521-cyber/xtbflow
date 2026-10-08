# SMC 补充工作统一索引与验收边界

全部结果为探索性。本补充响应用户“多做点实验，别破坏已有成果”，只读复用 V1a 开发集已有候选，不新增生成轨迹或量子化学计算。

## 交付索引

| 工作 | 冻结/实现 | 原始证据 | 仓库证据 |
|---|---|---|---|
| 原smc-1实验 | 2116d43；主交付0517ff9，PR116 | smc-2116d43/ | docs/evidence/explore_smc/ |
| raw终点排序消融 | 配置与代码726912a，计算前推送 | raw-ranking-726912a/results.json；raw-ranking-repeat-726912a/results.json | explore_smc_raw_ranking/REPORT_ZH.md、summary.json、SHA256SUMS |
| 可用性/排序损失分解 | 配置与代码8c98eac，计算前推送 | decomposition-8c98eac/；decomposition-repeat-8c98eac/ | explore_smc_decomposition/REPORT_ZH.md、results.json、SHA256SUMS |
| 独立候选指标验证 | 1443686 | supplement-independent-audit.json | explore_smc_decomposition/independent_audit.json |
| 独立统计区间验证 | 955f74d | supplement-interval-audit.json | explore_smc_decomposition/interval_audit.json |

运行根目录均为 n2 `/home/lhshen/xtbflow-runs/explore-smc-20261009`。每个source-短SHA克隆均保留；补充实验运行于推送后的干净源码。独立审计属于验证脚本而非新增科学实验。

## 要求到证据的映射

- **不破坏已有成果**：原dev-clone仍为0517ff9且干净；补充分支相对0517ff9仅新增文件及bootstrap库存；原SHA256SUMS全部复核通过，日志preservation-consolidated.log。原PR116保持open、未合并、base physics-guidance；补充PR117保持open、未合并、base smc-selection。
- **多做实验而非重复结果**：补充问题为终点打分器敏感性；全部13臂、全部24比较、两个排序器同时报告，不能把事后raw指标替换原主指标。第二项分解是机制诊断，不宣称新增采样。
- **先冻结后计算**：两项补充配置分别在726912a、8c98eac推送后执行。它们是在看过smc-1后提出，因此仍是事后探索，不能当作预注册确认性检验。
- **可重复性**：两项完整results分别在两个独立新输出目录计算，逐字节一致；哈希入库。raw完整结果1015766字节，按保守1000000字节阈值留n2；summary仅移除parent_metrics。
- **指标覆盖**：独立实现从77376候选、2418种群复算4030母体指标，最大误差0；没有调用原SMC指标函数。分解恒等式和可用性一致性误差0。
- **区间覆盖**：独立计算925个聚类区间，最大误差3.55e-15，覆盖绝对指标、配对差、排序替换及差中差、分解。没有调用项目统计函数；仍共享scipy的t分布实现。
- **资源/范围**：CPU单worker、CUDA隐藏、nice10；未申请补充GPU、未新增xTB、未使用screen/V1b/DFT、未安装包。已有成果读取不改变输入。
- **测试**：补充实验阶段320pytest通过；各提交bootstrap通过。原reusable套件56项已在主交付最终HEAD通过；补充最终分支另行复核日志见运行根目录。

## 结果要点（以机器可读结果为准）

raw终点排序下，主臂对none ΔH=-0.087814，区间[-0.205551,+0.029924]；对random ΔH=-0.050179，区间[-0.177674,+0.077316]。不能声称检测到生成中选择增益。

主臂对none的可用性下降0.198925，区间[-0.327006,-0.070843]；原排序下排序损失下降0.182796，区间[-0.311645,-0.053946]，二者相抵。代数恒等式不是因果分解，不能据此确认改善哪个组件会带来收益。

## 仍不满足严格原交接的历史项

原正式计算前的CPU冒烟只覆盖G1，克隆初态持久化在C1之后补做。现有计算和补充审计不能改变历史顺序。原目标不得仅凭这些补充实验被标记完成；若坚持原时序要求，需要负责人接受披露偏差或明确新实验的验收契约。无须为此反复重跑原实验或要求合并PR。

## 完成边界

已交付：原13臂科学实验、两个补充分析、点估计与区间独立复核、保全检查和报告。未交付：独立开发集之外的确认性证据、新生成种子实验或化学TS认证；它们不在当前补充范围。下一步只应针对明确的新科学问题或验收决定行动，不进行无界参数搜索。

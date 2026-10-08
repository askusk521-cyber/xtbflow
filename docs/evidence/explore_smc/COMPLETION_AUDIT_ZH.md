# 交接要求—证据核对（数值实验完成；历史流程差异待负责人验收）

## 具体交付
冻结 smc-1，在指定开发集完成13臂比较，所有复现门禁通过，确定性分析和中文报告，原始输出哈希、干净源码及资源记录，正确 base 的未合并 PR。不得将测试通过替代科学门禁或把偏差隐藏。

| 要求 | 证据 | 状态/边界 |
|---|---|---|
| §0–1 范围/问题 | smc_selection.json、REPORT_ZH.md、results.json | 仅62母体、21组、3seed；13臂，主臂预定 |
| §2.1 AGENTS/隔离分支 | STATUS.md；ffca40f→2116d43；PR116 | 原工作树只读，独立 clone |
| §2.2 GPU/CPU | slurm-2699.txt、slurm-2702.txt、各manifest、c2-time.txt | GPU Slurm47s；G2依赖2700；nice≥10，48worker，负载检查日志 |
| §2.3 不动旧文件/环境 | base...HEAD name-status | 仅新增，inventory例外；无安装包/DFT/screen/V1b |
| §2.4 干净已推源码 | source-2116d43、五阶段manifest dirty=false | 2116d43先推送再运行；辅助审计5fff33e |
| §2.4 提交/PR/大小 | Git trailers、PR116、文件大小检查、SHA256SUMS | 英中说明，未合并，数据单文件<1MB |
| §3 冻结/探索性/区间 | 初次冻结commit、报告首句、parent_metrics | 不改smc-1，母体seed先聚合，split_group聚类 |
| §4 输入/复用 | manifest inputs及导入源码 | pg1哈希匹配；现有采样/噪声/方向/反射/xTB/统计函数复用 |
| §5.1 种群/批次/检查点 | test_smc_pipeline、coverage_audit、24批G1张量 | 完整5952，批256，尾批192；k30/35/40/45/50 |
| §5.2 评分器/S0分支 | xtb_score.py、H2单位测试、S0记录 | raw/relaxed全参数；96/558负曲率触发无曲率层；额外raw诊断调用明确披露 |
| §5.3 克隆/随机 | smc.py、测试、plans.json、覆盖审计 | 2232方案，16幸存者且每个1克隆，噪声不含臂名 |
| §5.4 13臂 | 288个G2输出、2418完整种群 | 逻辑网络步数一致，前缀共享/续跑实际成本另报 |
| §5.5 指标/诊断 | results.json arms/comparisons/diagnostics/checkpoint_correlations/scorer_statistics | H、hit@K、hit@∞、效用、合法事件数/率、reference率；误剪/克隆/评分诊断 |
| §6 比较/读法 | 24比较，每个9项指标；readings | none区间含0，主结论NO_GAIN_OVER_POSTHOC；未按结果改分类 |
| §7 T | 六个指定新增文件，pytest/unittest/bootstrap | 320pytest、56unittest通过；最终源码与证据再次校验，日志在运行根目录 |
| §7 F | 冻结2116d43后smoke-2116d43/g1_manifest | 2母体32proposal seed0 CPU冒烟，无指标输出；只做G1冒烟，非全链路，作为流程覆盖限制明示 |
| §7 S0 | s0_manifest、s0_rows、独立audit | 558条raw误差0，rho精确复现；relaxed rho/曲率/降能报告 |
| §7 G1/C1 | g1/c1_manifest、none/checkpoint/plans原始输出 | none事件/status100%一致；有效raw差0；60成对失败单列 |
| §7 G2/C2 | g2/c2_manifest、endpoint_rows | 每臂2976幸存者100%事件一致，最大坐标偏差0；77376终点 |
| §7 analyze确定性 | results-first.json、analyze-repeat.sha256、cmp | 两次结果逐字节相同，均从冻结源码 |
| §7/9 D | evidence目录、PR116 | 报告、results、manifest、SHA256SUMS均已提交；PR116保留未合并并披露偏差 |
| §8 停止条件 | 门禁全部通过，资源低于限额 | G2排队约8分钟；无复现失败，无改已有科学定义 |
| §10 报告限制 | REPORT_ZH.md限制小节 | 重复开发集、0.7/lead3选择、闭世界、xTB/DFT排序差异均披露 |

## 已识别不足（不可用绿色测试掩盖）
- 原始冒烟只验证CPU轨迹，不是完整s0→analyze链路。交接要求“流程能否跑通”覆盖不充分；正式全链路已完成，但这不倒改历史冒烟范围。
- 初始阶段未收集进程CPU总核时，故只能精确报告C2进程时间和GPU墙钟；S0/C1采用48核×墙钟上界。禁止称为精确总CPU核时。
- C1 raw共有60条双边失败，不能声称23808条均有有限能量。失败配对状态一致，剩余有限值误差0。
- C1计划文件当时只持久化祖先与目标索引。现已从冻结输入按完全相同 G2 公式补存35712个克隆初态（288批），clone_initial_manifest记录全部哈希及跨臂同祖先一致性核查；这是事后证据补全，不能倒称C1当时就已存张量。
- 原始科学results与补充版都已两次逐字节复算；补充版954178字节，包含资源数字、覆盖计数、初态manifest。补充只增加证据，不改变任何科学指标。
- manifest中的Slurm号只对GPU阶段非空；CPU阶段明确null，而不是伪造调度号。

科学实验、核心复现、报告、资源数字补充和初态持久化均已完成；全部科学结论保持冻结版本不变。严格任务仍有无法事后修复的历史时间顺序差异：全流程冒烟并未在正式计算前完成，初态持久化晚于C1。这些差异必须由负责人验收接受；代理不能以事后补证据冒充原要求完全满足，因此不标记活动目标完成。

## 阻塞处理与选项
1. 推荐：负责人审阅已交付证据，明确接受以上历史流程差异；成本仅审阅，无新增科学计算。
2. 要求严格时序的新实验：另立新协议/运行目录，先全链路无指标冒烟和完整初态持久化，再重新执行全部阶段；会产生另一套探索性实验，不能抹去smc-1的历史差异，需先得到负责人授权。
3. 不得改写日志、首次推送时间或宣称原本做过完整冒烟。现无科学门禁失败、资源冲突或必须安装包的障碍。

所有要求仍按上表逐项列出。数值门禁通过不是流程偏差自动获准的依据。

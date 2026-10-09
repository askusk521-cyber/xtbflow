本报告全部结果为探索性；主比较 AUC(S_N_thermo) − AUC(B0) = -0.055437100，双侧 95% 区间 [-0.106963092, -0.003911109]，读法为 `GENERATION_BEATS_ENUMERATION`。

# T1b：产物热力学筛选枚举基线

使用全部 342 个冻结 screen 母体；训练集冻结阈值 τ = 141.295302852 kcal/mol。没有使用 M0 官方 val/test，没有重新生成生成侧候选。

## 配对比较

| 比较 | AUC 差 | 双侧 95% 区间 | 读法 |
|---|---:|---|---|
| S_N2-B0 | -0.105330490 | [-0.155926249, -0.054734732] | GENERATION_BEATS_ENUMERATION |
| S_N_thermo-A0 | -0.008902077 | [-0.059996295, 0.042192140] | INCONCLUSIVE |
| S_N_thermo-A1 | -0.004483431 | [-0.054217592, 0.045250731] | INCONCLUSIVE |
| S_N_thermo-A2 | 0.002923977 | [-0.044302478, 0.050150432] | INCONCLUSIVE |
| S_N_thermo-B0 | -0.055437100 | [-0.106963092, -0.003911109] | GENERATION_BEATS_ENUMERATION |
| S_N_thermo-B1 | -0.003323558 | [-0.055157917, 0.048510801] | INCONCLUSIVE |
| S_N_thermo-S_N2 | 0.050804094 | [0.031701105, 0.069907082] | THERMO_FILTER_HELPS |
| S_thermo-B0 | -0.313859275 | [-0.353555877, -0.274162673] | GENERATION_BEATS_ENUMERATION |

## 覆盖率与条件分析

b2f2 最佳通道覆盖率：0.690058480；目录通道覆盖率的母体均值：0.721076341。
只在最佳通道属于 b2f2 的母体上，主比较为 0.102824859，区间 [0.042126819, 0.163522898]，`ENUMERATION_BEATS_GENERATION`。
b2f2 事件总数 636320，每母体中位数 1638.5，95% 分位 4337.4。

最佳事件不属于 b2f2、但生成臂曾命中的母体数：
- A0: 92
- A1: 85
- A2: 80
- B0: 92
- B1: 83

## 流水线与失败

成功 599376 / 636320，成功率 0.941941162。
成功事件中 ΔE > τ 的数量为 458696；占所有事件的比例 0.720857430，占成功事件的比例 0.765289234。
每事件产物几何与单点的平均墙钟时间为 0.015281243 秒；该值不含每母体一次的反应物构建、初始化及测试开销。完整 CPU 用量见资源文件。

失败原因：
- `product:RuntimeError:Invariant Violation
	upper bound not greater than lower bound
	Violation occurred on line 240 in file Code/GraphMol/DistGeomHelpers/BoundsMatrixBuilder.cpp
	Failed Expression: ub > lb
	RDKIT: 2026.03.6
	BOOST: 1_85
;reactant:ok`: 74
- `product:ValueError:embedding_failed;reactant:ok`: 36867
- `product:ValueError:forcefield_not_converged;reactant:ok`: 1
- `product:collapsed;reactant:ok`: 2

## 冻结定义、验证和限制

排序规则及阈值来源见 configs/explore/enum_thermo.json 与 enum_thermo_tau.json。平局按通道 SHA256；失败事件按冻结规则放置，不删除。
指标先按母体与训练 seed 计算，再在母体内平均。配对差使用 split_group 分组的 cluster_summary 双侧区间。k 网格、删失规则与原枚举分析一致；每个比较的 retained/excluded k 和母体名单在 results.json 中。
T1b-0 与 B0 完成候选门禁逐字节复现；三个最小枚举母体的 b2f2 过滤与直接枚举通道集合完全一致。两次最终分析的逐字节核验记录随证据交付。
本次未采用 171 母体子集：前五个训练母体的计时预算判断选择了全 screen。预估和实际资源必须分别报告，不能把预估当成实际用量。
本实验只有代理事件召回，不证明提出了可认证过渡态，也不证明化学发现或机制优势。训练阈值和 screen 数据不构成独立保留集确认。
RDKit 未收敛或无法嵌入的情况计为失败；无 MMFF/UFF 参数但嵌入成功时按交接保留几何并记录 no_forcefield。不得根据结果修改阈值或规则。

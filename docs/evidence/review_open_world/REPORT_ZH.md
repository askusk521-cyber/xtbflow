本报告全部结果为探索性。目录外组在 GFN2-xTB 层面的严格认证率为 17/40（42.50%，95%区间 28.51%–57.80%）；这不是 DFT 认证，也不改变 V1a 的结论。

# T3：开放世界终点第一阶段

## 冻结样本与认证率

|组别|严格认证率及 Wilson 区间|
|---|---|
|PROXY_MATCH|17/20（85.00%，95%区间 63.96%–94.76%）|
|KNOWN_EVENT_GEOMETRY_MISS|12/20（60.00%，95%区间 38.66%–78.12%）|
|OUTSIDE_CATALOGUE_EVENT|17/40（42.50%，95%区间 28.51%–57.80%）|
|BASELINE|31/40（77.50%，95%区间 62.50%–87.68%）|

阳性对照闸门：通过。配置先于任何量化计算提交，样本重新抽取与冻结配置逐字节一致。所有候选按哈希排序选取，每组内每母体最多一条；组间可能重复母体，因此组间差异不是独立随机试验。

## 与同母体基准的势垒差

只对候选和基准链均严格通过的母体比较 TS 电子能量差，单位 kcal/mol。
- ΔEa_xTB < 0：1/12（8.33%，95%区间 1.49%–35.39%）
- ΔEa_xTB ≤ +5：5/12（41.67%，95%区间 19.33%–68.05%）
- 目录外候选虽通过、但基准未通过而排除的母体数：5

|母体|ΔEa_xTB kcal/mol|
|---|---:|
|8962b71ed3a7a171af9e|26.257367|
|a9460cfa7d3e4acccdb6|2.834676|
|854e36201fd5181b6d38|46.893734|
|8fe6d50aaad53c21d3c5|1.612403|
|4f5559117464139fee00|-0.599680|
|e193d2b0c22148150613|38.260903|
|7dc6922783059ead3fb0|23.725554|
|5b2d3d450552ae85d0f4|1.885950|
|13bf28b59fe70fb07791|51.818212|
|c0f8c54d5d73fed95b92|2.659227|
|a6ee19115420a88ad88e|23.729136|
|3e4899195c01fa43baa1|20.401106|

## 失败分布

|组别|状态|数量|
|---|---|---:|
|BASELINE|IRC_LIMIT|3|
|BASELINE|IRC_REACTANT_MISMATCH|1|
|BASELINE|MINIMUM_HAS_NEGATIVE_MODE|1|
|BASELINE|STRICT_EVENT_MISMATCH|3|
|BASELINE|STRICT_JOINT_GRAPH_VALID|31|
|BASELINE|TS_OPT_LIMIT|1|
|KNOWN_EVENT_GEOMETRY_MISS|EVALUATION_UNRESOLVED|1|
|KNOWN_EVENT_GEOMETRY_MISS|IRC_LIMIT|3|
|KNOWN_EVENT_GEOMETRY_MISS|NO_NEGATIVE_MODE|1|
|KNOWN_EVENT_GEOMETRY_MISS|STRICT_EVENT_MISMATCH|2|
|KNOWN_EVENT_GEOMETRY_MISS|STRICT_JOINT_GRAPH_VALID|12|
|KNOWN_EVENT_GEOMETRY_MISS|TS_OPT_LIMIT|1|
|OUTSIDE_CATALOGUE_EVENT|EVALUATION_UNRESOLVED|2|
|OUTSIDE_CATALOGUE_EVENT|IRC_LIMIT|3|
|OUTSIDE_CATALOGUE_EVENT|IRC_REACTANT_MISMATCH|5|
|OUTSIDE_CATALOGUE_EVENT|MINIMUM_HAS_NEGATIVE_MODE|1|
|OUTSIDE_CATALOGUE_EVENT|NO_NEGATIVE_MODE|1|
|OUTSIDE_CATALOGUE_EVENT|STRICT_EVENT_MISMATCH|9|
|OUTSIDE_CATALOGUE_EVENT|STRICT_JOINT_GRAPH_VALID|17|
|OUTSIDE_CATALOGUE_EVENT|TS_OPT_LIMIT|2|
|PROXY_MATCH|MINIMUM_HAS_NEGATIVE_MODE|1|
|PROXY_MATCH|STRICT_JOINT_GRAPH_VALID|17|
|PROXY_MATCH|TS_OPT_LIMIT|2|

优化或 IRC 达到步数上限、数值问题和图感知失败不能解释为通道不存在。严格认证仅要求一端为反应物、另一端为预测产物，并满足该 xTB 链的频率与收敛条件；不证明通道在更高层级方法中存在。

## 算力、实现与复现

全部为 CPU 直接运行（负责人追加授权），GPU 用量为零，没有提交 DFT 作业。每条链的墙钟和梯度调用见 verdicts.csv，汇总见下表。数值 Hessian 使用梯度中心差分并对称化，真实优化水分子的平动转动投影秩和正内部频率检查通过，证据见 hessian_check.json。

|组别|墙钟秒合计|梯度调用合计|
|---|---:|---:|
|BASELINE|902.607437|19416|
|KNOWN_EVENT_GEOMETRY_MISS|580.162309|11562|
|OUTSIDE_CATALOGUE_EVENT|1263.423405|22913|
|PROXY_MATCH|259.118598|9225|

纯函数 harmonic、classify_saddle、classify_minimum、endpoint_identity 直接来自 qc_protocol，没有改写，也未调用该模块的 DFT 引擎。数值设置、干净源码提交、配置及原始文件哈希在 manifest.json；完整环境见 pip-freeze.txt。

```bash
export CUDA_VISIBLE_DEVICES= PYTHONPATH=src:vendor/mechai_reusable OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python scripts/review_open_world_analyze.py --root /home/lhshen/xtbflow-runs/review-align-20261009/open-621b900 --out results.json
```

原始链保留于 n2，逐条 verdict 哈希列于 SHA256SUMS.remote；分析重跑必须与提交的 results.json 字节一致。

## 下一阶段

STAGE2_PROPOSAL_ZH.md 提供至多十条新通道的 DFT 认证提案、样本排序规则和算力预估。这里只提交提案，没有运行授权，也没有运行。

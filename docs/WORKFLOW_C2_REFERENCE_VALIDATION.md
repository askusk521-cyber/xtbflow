# 工作流 C2：CP2K 参考桥接与 TS 验证

工作流 C2 对应 #43、#17 和 #19。它把三种不同层级的证据分开：

1. **计算器正确性**：当前代码是否能用冻结 CP2K 身份稳定产生可审计的 E/F；
2. **参考桥接**：在完全相同的构型、电荷和多重度上，CP2K 与已资格化 CPU GFN2 的差异是多少；
3. **路径验证**：候选是否经过 TS 优化、频率/模态检查以及端点连通性验证。

前两层不能替代第三层。E/F 成功、有限差分通过或 CP2K–GFN2 差异较小，都不能单独证明一个结构是过渡态。

## 冻结运行身份

物理协议由 `configs/calculators/cp2k/protocol_v0.1.yaml` 给出。运行时加载器会在计算前检查：

- CP2K 版本和源码 revision 与协议中的 build identity 一致；
- basis、pseudopotential 和 D3 参数文件真实存在；
- 每个体系显式提供电荷和多重度；
- 运行时路径不写入公共证据，只公开可执行文件名和 SHA-256。

任一检查失败时，计算在启动 CP2K 前失败关闭。

## Gate C2.1：当前代码重放

```bash
export XTBFlow_CP2K_EXECUTABLE=/path/to/cp2k.psmp
export PYTHONPATH=src:vendor/mechai_reusable
export OMP_NUM_THREADS=4

python scripts/cp2k_calibration_smoke.py \
  --threads 4 \
  --timeout 600 \
  --artifact-dir "$RUN_ROOT/cp2k-calibration" \
  --output "$RUN_ROOT/cp2k-calibration.json"
```

三个校准体系分别覆盖中性闭壳层、带电开壳层和含硫闭壳层。所有 input、output、stdout 和 stderr 必须持久化。该 gate 只证明当前 adapter/runtime 可执行，不授予正式参考标签。

## Gate C2.2：解析力的独立有限差分检查

```bash
python scripts/cp2k_force_fd_smoke.py \
  --output "$RUN_ROOT/cp2k-force-fd.json" \
  --ledger "$RUN_ROOT/cp2k-force-fd-ledger.json" \
  --artifact-dir "$RUN_ROOT/cp2k-force-fd-artifacts" \
  --threads 4 \
  --step 1e-3 \
  --threshold 5e-4
```

脚本对一个非平衡水分子构型执行三次有预算令牌的 CP2K 调用：基准 E/F、正位移能量和负位移能量，并比较

\[
F_{i\alpha}^{\mathrm{analytic}}
\quad\text{与}\quad
-\frac{E(R+h e_{i\alpha})-E(R-h e_{i\alpha})}{2h}.
\]

通过只验证一个体系、一个坐标分量和一个冻结运行时；它不等于基组/截断收敛或广泛化学精度验证。

## Gate C2.3：有界同构型参考桥接

冻结配置位于 `configs/experiments/cp2k_bridge.yaml`。首个 pilot 只包含四个预先声明的体系：

- 非平衡中性水；
- 非平衡中性 H₂S；
- 非平衡闭壳层 H₃O⁺；
- 有限显式水环境中的水二聚体。

它们分别检查中性、含硫、带电和有限水环境，不能代表完整目标域。

```bash
python scripts/run_reference_bridge.py \
  --config configs/experiments/cp2k_bridge.yaml \
  --output "$RUN_ROOT/reference-bridge.json" \
  --ledger-dir "$RUN_ROOT/reference-bridge-ledgers" \
  --artifact-dir "$RUN_ROOT/reference-bridge-artifacts"
```

运行器强制串行执行，并分别记录 CP2K 和 GFN2 calculator calls。每个比较要求两种后端具有相同的 `input_hash`，即原子顺序、坐标、电荷、多重度和环境合同完全一致。

允许汇总的主要量是力分量差：

- component MAE；
- component RMSE；
- component maximum absolute difference。

绝对能量只保留在单体系记录中。不同化学计量之间不得汇总绝对能量差，也不得把四个小体系的均值写成广泛精度结论。

## Gate C2.4：截断与协议收敛

当前 600 Ry 协议仍标记为 provisional。600→700 Ry 的既有结果显示力差较小，但严格绝对能量阈值没有通过。完成新的持久化重放并审阅预先冻结的能量、力和成本阈值之前，不得把该协议升级为正式参考标签。

## Gate C2.5：真实 TS/模态/连通性

#19 的软件验证器已经能够记录候选身份、预算、模式证据和端点证据，但物理准入仍要求至少一个真实成功通道和一个失败对照：

1. 使用独立参考协议进行 TS 优化；
2. 计算 Hessian/频率并确认恰好一个相关虚频；
3. 检查虚频模态与目标反应事件的一致性；
4. 运行 IRC 或等价双向路径；
5. 根据端点重新推导观察到的事件，而不是复用目标标签；
6. 保留所有 calculator calls、失败、重试和原始 artifact。

在此 gate 完成前，`path_status` 必须保持 `not_requested`、`not_run` 或 `not_validated`，不得写成 `validated`。

## 证据发布规则

公共证据可以包含：

- 物理协议和运行脚本的 SHA-256；
- CP2K 版本、revision、可执行文件 SHA-256；
- 体系输入、input hash、数值结果和预算计数；
- 原始 artifact 的文件名和内容哈希。

公共证据不得包含私有主机绝对路径、凭据或未经许可的大型原始输出。完整输入/输出保留在受控运行目录中，公共记录只保存经过清理的有界摘要。

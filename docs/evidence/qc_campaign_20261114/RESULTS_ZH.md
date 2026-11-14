# GQC 准确性与第二批量化计算：2026-11-14

**结论：当前编译版的 B3LYP/def2-SVP 在本轮分子上通过数值核验；
不能因此说 GQC 的所有方法都准，也不能把 DFT 的化学预测当成精确答案。**
本轮实测发现 M06-2X 的 GPU 力明显不正确，def2-TZVP 的部分 GPU 力积分尚不支持。
后续结构搜索只使用通过核验的 B3LYP/def2-SVP；大基组、频率、TS 精修和 IRC
明确由 CPU PySCF 提供。

## 已经完成的准确性检查

| 检查 | 实测结果 | 可支持的结论 |
|---|---|---|
| 六个扰动构型，GQC B3LYP/def2-SVP 对 CPU PySCF | 最大能量差 4.75×10⁻¹² Hartree；最大笛卡尔力差 7.23×10⁻⁵ Hartree/Bohr | 这六个构型在记录的协议下数值一致 |
| 同六个构型，力对两档能量有限差分 | 最大方向误差 1.32×10⁻⁸ Hartree/Bohr | 这些点的 GQC B3LYP 力与自身能量导数一致 |
| 水簇和含氮反应复合物 | 两项新增核验均通过原定阈值 | 扩展了本轮的测试范围，仍非全化学空间认证 |
| 两个 R6 构型，GQC M06-2X/def2-SVP | 能量一致；方向力误差 0.170–0.195 Hartree/Bohr，笛卡尔力差最高 0.744 | 该版本的这一力接口不能用于优化或路径计算 |
| 六项 GQC B3LYP/def2-TZVP 力检查 | 全部报 `CUDA gradient integral descriptors admit s/p/d bases only` | 功能不支持；不是六个化学体系“算错能量” |

力有限差分步长为 0.001 和 0.0003 Bohr；预先设定的方向力、跨程序能量、
跨程序笛卡尔力阈值依次为 2×10⁻⁵ Hartree/Bohr、10⁻⁴ Hartree、
3×10⁻⁴ Hartree/Bohr。失败后没有放宽阈值。

## 化学能量差是否可靠

下表均为**烯醇式减此前选定的羰基式**，单位 kcal/mol；正值偏向该羰基式。
这是固定几何上的电子能比较，不是溶液自由能或完整互变异构体种群。

| 体系 | GQC B3LYP/SVP，气相（此前完成） | CPU B3LYP/TZVP，气相 | CPU CCSD(T)/cc-pVDZ，气相 | CPU B3LYP/TZVP，C-PCM 水 | 同左，C-PCM 乙腈 |
|---|---:|---:|---:|---:|---:|
| R6 | 3.014 | 2.541 | 7.116 | 2.158 | 2.155 |
| R30 | 3.416 | 3.047 | 7.424 | 2.791 | 2.785 |
| R51 | 12.028 | 13.234 | 15.272 | 14.334 | 14.306 |

这三对结构的排序一致，但方法间差异达到数 kcal/mol，因此现阶段不能相信能量差的
精确数值。CCSD(T)/cc-pVDZ 也有基组、单参考和几何选择误差，不能把它与 B3LYP
之差直接称为 B3LYP 的真实误差。已另行提交 R6 两个结构的 CCSD(T)/cc-pVTZ
检查。在本次提交快照中，它们尚未完成。

同方法、同基组的 PCM 水与气相比，R6/R30/R51 的相对能量差分别移动约
−0.383、−0.256、+1.100 kcal/mol。PCM 未优化溶液几何，也未提供溶剂化自由能的
完整模型。这组单点不等于溶液种群、化学速率或水促进机理。
此前的代表构象、频率和热修正见
[第一轮证据](../gqc_tautomer_20261114/RESULTS_ZH.md)。

## 已提交的任务与依赖

本轮共 **106 个登记任务**：首批 32 个、第二批 74 个；每个任务仅提交一次。
登记任务包括条件任务，前置物理检查不通过时将记录 blocked，不消耗路径计算预算。

| 阶段 | 个数 | Slurm 作业/数组 | 状态与依赖 |
|---|---:|---|---|
| 首批 GQC 数值核验 | 14 | 4650、4651 | 已结束：6 通过、2 力核验失败、6 功能不支持 |
| CCSD(T)/cc-pVDZ | 6 | 4652、4653 | 已完成 |
| PCM 水/乙腈单点 | 12 | 4654、4655 | 已完成 |
| 含氮/水簇新增 GQC 核验 | 2 | 4682 | 已通过 |
| 同基组气相单点 | 6 | 4683 | 已完成 |
| CCSD(T)/cc-pVTZ | 2 | 4684 | 提交快照：1 运行、1 排队 |
| 补充构象/异构体/路径端点优化 | 28 | 4685（探针）、4693 | 探针通过；其余运行/排队 |
| 对应频率 | 28 | 4694（探针）、4698 | 探针通过；其余等待优化阶段 |
| 反应路径链 | 8 | 4699 | 已入队，等待频率阶段及逐端点资格检查 |
| 最终汇总，无新增量化计算 | — | 4700 | 等待路径、频率、CCSD(T) 任务结束 |

八条路径假设是：R6 直接互变异构、一个显式水参与的互变异构、R6 与乙烯的 DA；
R51 与乙烯的 DA/逆 DA、该加合物的半缩醛开环；氨基呋喃/马来酰亚胺的 DA 与
中性 aza-Michael 竞争；一个显式水参与的 R51 加合物开环。它们使用明确构造的
中性单重态模型和双端点假设，尚不是已经确认的反应机理。

28 个结构包括 14 个路径端点、6 个固定新种子的补充构象、8 个此前未覆盖的
羰基连接方式/立体异构体。前后端点通过显式图编辑保持原子身份，包括转移的氢。
第一次非量化水摆放 smoke 中有两个结构被键识别判为不同分子；保留该失败后，
改为有界的无碰撞摆放。最终 28 个输入全部通过初始结构检查才提交量化任务。
没有从参考 TS、密封标签或计算结果选择初始反应通道。

路径链为 GQC NEB 候选 → CPU PySCF TS 精修 → 恰好一个显著虚频 → 双向 IRC
→ 两端优化、频率及结构身份。Hessian 另有梯度有限差分方向检查。
**本次快照尚无已经验证的 TS 或连接路径；完整路径入口尚未在本轮实际启动。**
单条搜索失败也不能证明该反应不存在。氨基反应仅覆盖一个中性 Michael 假设，
水模型只有一个显式水及一个确定的初始摆放，未穷尽逐步反应、水解或溶液路径。

## 精确身份、资源和状态

- 主机 `node2`；分区 `main`；每个 GPU 任务 `gpu:pro6000:1`。
- 第一批源码 `cb28d5c765ce2026395e149d91fd5662845b18d8`；第二批源码
  `bc13c63f1d3bb602e41982df349f6800b67d3043`，均为独立干净执行目录。
- GQC 上游 `2bf3fd98da9c6cfba46ca83345e63e1d934e4780`；native SHA-256
  `67bc98a4549fe8854ed6ae65791c824024bd77a9e8e498c6891dea3b8de670c4`。
- 冻结配置 SHA-256 `e2d870a1fd71706970236b74313d7e5f69e209640dbb61b199d9fcc48a622bd9`。
- 首批输入 manifest SHA-256
  `9f89ec5680089e1333b219ab99804deb84630fadb43ca5ec9b5376817af0c60b`；第二批
  `f1421d6b8870157b93c42f3dd89f23f874ed8fe247d5dfb8906191fc8fcd9c66`。
- Quantum Python 3.11.17，NumPy 2.4.6、SciPy 1.17.1、PySCF 2.14.0、
  geomeTRIC 1.1.1；ASE 3.29.0 纯 Python 包隔离在本轮目录中。
  结构处理/软件检查环境是 `xtbflow`：Python 3.10.19、NumPy 2.2.6、
  PyTorch 2.10.0+cu128、RDKit 2024.03.3。
- 登记任务上限：40.834 GPU 小时、451.334 CPU 核小时、18,156 个量化操作；
  授权封顶分别为 48、512、20,000。最多同时运行 2 个 GPU、2 个 CPU 量化 worker，
  不自动重试。每任务还有单独调用/时长/内存限制；路径每条最多 150 分钟、1,600 操作。
- 第一批实际消耗 84 个操作，其中 6 个失败；程序累计墙钟时间 1,165.12 秒，
  进程 CPU 时间 5,101.93 秒。不能把这些数字当作 Slurm 分配的 GPU 时间。
- 14:30:55 UTC 任务快照的总账为 201 个已开始、192 个已完成、6 个失败、
  3 个进行中尚未结算的操作。运行中的日志会继续增加，历史快照不覆盖。
- 14:32:30 UTC 保留数据约 2.258 GiB；4 GiB/任务与 32 GiB/本轮的磁盘检查发生在
  量化操作之间（全局每十次检查），不是文件系统硬配额。
- Slurm accounting storage 未启用；精确已分配 GPU 秒数为 `unknown`。
  worker 结果、已完成 launcher 退出码和当前队列分开保留。作业超时或杀死后的
  `started`/未结算操作不得改写成零成本的 `not_run`。

[任务快照及逐项哈希](submission_snapshot.json)、
[提交回执和依赖](submission_receipts.json)、
[调度器快照](scheduler_snapshot.json)、
[首批预算](wave1_manifest_summary.json)、[第二批预算](wave2_manifest_summary.json)、
[软件验证](software_checks.json) 均为可公开摘要。原始坐标、完整轨迹和 scratch 仍在 n2。

## 复现与取回结果

以下为命令模板。`N2_HOME` 在 n2 设为当前用户家目录；代码目录名称对应固定提交。
原运行目录不可重跑覆盖；复现时选择新的输出目录，并重新统计预算。
完整逐任务 `sbatch` 参数、数组索引、时限和输出位置已保存在提交回执中。

```bash
ssh n2 'N2_HOME="$HOME"; \
  QC_CODE="$N2_HOME/xtbflow/xtbflow-exec-campaign-bc13c63-20261114"; \
  QC_RUN="$N2_HOME/xtbflow-runs/qc-campaign-20261114"; \
  cd "$QC_CODE"; \
  test "$(git rev-parse HEAD)" = bc13c63f1d3bb602e41982df349f6800b67d3043; \
  source "$N2_HOME/miniconda3/etc/profile.d/conda.sh"; conda activate xtbflow; \
  squeue -u "$USER"; \
  test ! -e "$QC_RUN/final-summary.json" || \
    python -m json.tool "$QC_RUN/final-summary.json"'
```

准备命令（n2，最多 60 秒，不运行量化；`QC_FRESH` 必须不存在）：

```bash
cd "$QC_CODE"  # bc13c63f1d3bb602e41982df349f6800b67d3043
source "$N2_HOME/miniconda3/etc/profile.d/conda.sh"
conda activate xtbflow
timeout 60s python scripts/prepare_qc_paths.py \
  --config configs/research/qc_campaign_20261114.json \
  --wave1 "$QC_RUN/wave1/tasks.json" --out "$QC_FRESH"
```

单个数组任务模板（n2，示例 opt index 2；只在新的 results 路径中复现）：

```bash
cd "$QC_CODE"  # bc13c63f1d3bb602e41982df349f6800b67d3043
sbatch --partition=main --nodes=1 --ntasks=1 --cpus-per-task=4 \
  --mem=16G --time=30 --gres=gpu:pro6000:1 --array=2%1 \
  --output="$QC_RUN/replay-%A_%a.out" --error="$QC_RUN/replay-%A_%a.err" \
  scripts/run_qc_campaign_task.sbatch "$QC_CODE" \
  bc13c63f1d3bb602e41982df349f6800b67d3043 \
  "$QC_RUN/wave2/tasks.json" "$QC_RUN/results-replay-UNIQUE" 1
```

wrapper 在作业内加载 `xtbflow-qc`、私有 `runtime.env`、固定 GQC Python 和
OpenBLAS 动态库，使用 Slurm 分配的 GPU 可见性；在进入量化前验证 native 哈希和
数值资格。该示例最多 180 个操作、30 分钟，另受本轮预算约束。

软件验证在 n2 对第二批精确源码执行：
`PYTHONPATH=src:vendor/mechai_reusable pytest -q` 得到 441 passed / 68 subtests；
`python -m unittest discover -s tests/reusable -p 'test_*.py'` 得到 56 passed；
`python scripts/validate_bootstrap.py` 得到 551 个文件一致。
软件检查总计限制 300 秒，隐藏 GPU、设置两个 CPU 线程；结果不算量化证据。

本工作为 #207 的堆叠后续，保留人工审查，不自动合并。

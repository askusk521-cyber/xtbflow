# n2 GenerativeQC 首轮互变异构筛查（2026-11-14）

**已完成真实计算与技能安装。** 复用了已有编译的 GenerativeQC，在 Slurm 的一张
RTX PRO 6000 上完成有限构象的 B3LYP/def2-SVP 优化；随后由 CPU PySCF 独立核验
最低点，并计算 M06-2X/def2-TZVP 单点作为方法/基组敏感性对照。

三组已选结构均表现为羰基形态更稳定。该结果支持对反应活性形态进行进一步审查，
但并不证明 Diels–Alder 反应失活，也没有建立超出已知互变异构规律的新机制。
按候选计划，这一结果本身不足以支持扩大训练或把普通烯醇不稳定包装成新课题。

## 化学结果

所有差值均为 **烯醇减羰基**，正值表示本轮选中的羰基构型更稳定，单位 kcal/mol。

| 来源种子 / 本轮单体对 | GQC B3LYP 优化电子能差 | M06-2X/def2-TZVP 单点能差 | 单构象谐振自由能差，298.15 K / 1 atm 气相 |
| --- | ---: | ---: | ---: |
| R6：`C=CC=CO` / `C=CCC=O` | +3.0145 | +2.2225 | +4.4979 |
| R30：`CC=CC=CO` / `CC=CCC=O` | +3.4161 | +2.6661 | +4.8599 |
| R51：`Cc1ccc(O)o1` / `CC1=CCC(=O)O1` | +12.0281 | +10.6954 | +11.8984 |

输入独立生成，使用明确的气相、电荷 0、多重度 1 模型；没有读取原始参考 TS、产物
几何、势垒标签或训练模型。8 个连接图经有限 E/Z 枚举形成 13 个起始结构，由 26 个
ETKDG/MMFF 初始构象选出。13 个结构最终均达到最大力 3e-5 Hartree/Bohr 的阈值，
且优化后图身份和立体化学核对全部通过。

每个连接图选择本轮采样中能量最低的一个构型进行独立频率检查，合计 8 个代表，
包括丁二烯和呋喃两个对照。8 个代表都没有虚频；最低保留频率范围为
58.90–616.09 cm^-1。这不表示 13 个起始结构全部分别计算了 Hessian，也不证明
已经找到每个分子的全局最低构象。

R6 羰基代表的 Hessian-vector 有限差分抽查在 1e-3 和 3e-4 Bohr 两个尺度上的
相对误差均约 2.01e-4，小于预先设定的 1% 阈值。Hessian、频率与谐振热校正由
**CPU PySCF 2.14.0** 提供，不能记成 GenerativeQC Hessian 的资格验证。

M06-2X 数值是在选定 B3LYP 几何上的单点，未重新优化或计算其热校正。
自由能列使用相同 B3LYP 级别的独立 PySCF 能量和谐振热校正。未做构象系综、
其他互变异构连接方式、进一步双键迁移、溶剂、水活度、互变异构速率或 DA 竞争路径。
因此不能从表中直接推断实际反应物种群、反应速度、转化率或不可逆失活。

完整数值：[minima_4622.json](minima_4622.json)。

## 程序可复用性与真实运行边界

- 实际使用 upstream `2bf3fd98da9c6cfba46ca83345e63e1d934e4780` 的既有 wheel/venv，
  Python 3.11.17，FP64，CUDA，球谐 def2-SVP，未使用密度拟合。
- 主库 SHA-256：`67bc98a4549fe8854ed6ae65791c824024bd77a9e8e498c6891dea3b8de670c4`。
  其余插件库及 Python calculator 哈希保存在资格报告中。
- 新门槛包含水与 R6 两种形态的有限差分，以及同级别 CPU PySCF 比较。
  最大方向力误差 2.79e-8 Hartree/Bohr；独立能量差不超过 5.9e-12 Hartree；
  最大独立 Cartesian 力差 3.28e-5 Hartree/Bohr。
- 只对上述实际测过的 B3LYP 路径作本轮资格结论。旧构建的 CUDA PBE 等力失败
  不能因为库可加载或 capability 显示支持而被抹去；
  [原始历史记录](../generativeqc_force_root_cause_20261109.json)按原字节保留。
- n2 的私有 `~/.config/n2-qc/runtime.env` 已记录实际解释器、OpenBLAS 目录与库身份。
  用户级 `n2-quantum-chemistry` 技能已安装；仓库版本位于 `skills/n2-quantum-chemistry/`。

## 各阶段、失败和成本

| 阶段 | Slurm | 实际结果 | 新量化操作 |
| --- | --- | --- | ---: |
| 首次库加载 | 4618 | OpenBLAS 搜索路径缺失；未进入量化计算 | 0 |
| 修复环境后的资格检查 | 4619 | 原有库通过新数值检查 | 18 |
| 原始 Cartesian BFGS | 4620 | 时间预算耗尽；已收敛、迭代上限与未轮到项均保留 | 456 |
| 单独注册的 TRIC 续算 | 4621 | 全部 13 个起始结构达到相同力阈值 | 91 |
| 独立最低点/敏感性核验 | 4622，CPU | 8 个代表通过最低点检查，三对比较齐全 | 28 |

合计 **593 次账本量化操作**：562 次 GQC CUDA 能量/力调用，31 次独立 PySCF
操作，其中包含 8 次解析 Hessian。不同操作不能视为相同计算成本；原始调用时间和
各阶段 CPU/墙钟记录都保留。完整证据与日志哈希见
[execution_receipt.json](execution_receipt.json)。

原始优化计时 1502.25 秒，在最后一个 native 操作返回后触发 1500 秒软期限停止；
没有继续发起下一次调用，30 分钟 Slurm 硬上限未触发。所有 456 次调用均已结算。
续算用了 336.51 秒、91 次新调用，没有重跑已收敛起点。独立 CPU 阶段用了
1054.93 秒、4437.97 进程 CPU 秒，低于其 1100 秒软期限及 20 分钟硬上限。
原始失败、未收敛状态与开销没有被续算覆盖。

Slurm accounting storage 关闭，结束后的部分 scontrol 记录已过期，因此精确的
调度分配 GPU 秒数为 unknown；不把进程计时充当调度计费时长。srun 的退出码和
独立阶段报告均保留，最终队列为空。本轮保留的运行数据小于 1 MiB。

## 源码、核验与重放

- 准备、资格与首次优化源码：`323fd156987322ed7181dd48b4fe514a162f702d`。
- 续算、结构核对与独立核验源码：`aad1b5362883c5c2364acd2eec5c6c1e90ff4d13`。
- 输入包 SHA-256：`d6175d057bbd193b6c2e8b1e48066fb61cd8a92ac7f9fbace199e08b021bc3b2`。
- 主协议 SHA-256：`b4245b2ece9f1695bb0445045d0cf81ac54916c26c7072a7fabd39510e155f58`。
- n2 在 `aad1b53` 上通过 435 pytest 测试、68 子测试及 56 reusable 测试；
  bootstrap 527 文件通过。软件测试不能代替上述物理检查。
- 技能通过 skill-creator 的结构验证；其私有运行环境配置已在 4621 续算中实际使用。

入口、输入限制和前两阶段命令见
[注册执行方案](../../research/GQC_TAUTOMER_PILOT_20261114.md)。所有输出路径须是
新的空目录；不要覆盖本轮报告。复用相同输入时，原始包仍保留在 n2，仓库不含坐标原始字节。
以下续算/核验命令要求在 n2 预先设置 `QC_CODE` 为上述 `aad1b53` 干净源码，
`QC_FIRST` 为原阶段结果目录，`QC_RUN` 为新的阶段输出父目录。

```bash
ssh n2 '
  set -e
  cd "$QC_CODE"
  test "$(git rev-parse HEAD)" = aad1b5362883c5c2364acd2eec5c6c1e90ff4d13
  test -z "$(git status --porcelain)"
  source ~/miniconda3/etc/profile.d/conda.sh
  conda activate xtbflow-qc
  source ~/.config/n2-qc/runtime.env
  export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
  export LD_LIBRARY_PATH="$GQC_OPENBLAS_DIR:$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  srun --partition=main --gres=gpu:pro6000:1 --nodes=1 --ntasks=1 \
    --cpus-per-task=4 --mem=16G --time=00:12:00 \
    --output="$QC_RUN/continuation-%j.out" --error="$QC_RUN/continuation-%j.err" \
    "$GQC_PYTHON" scripts/continue_gqc_tautomer_minima.py \
    --config configs/research/gqc_tautomer_pilot_20261114.json \
    --continuation configs/research/gqc_tautomer_continuation_20261114.json \
    --inputs "$QC_FIRST/prepare/inputs.json" --optimized "$QC_FIRST/optimize/report.json" \
    --gate "$QC_FIRST/qualify-openblas/report.json" --out "$QC_RUN/continue"
  conda activate xtbflow
  export CUDA_VISIBLE_DEVICES=
  timeout 60s python scripts/gqc_tautomer_pilot.py verify \
    --config configs/research/gqc_tautomer_pilot_20261114.json \
    --inputs "$QC_FIRST/prepare/inputs.json" --optimized "$QC_RUN/continue/report.json" \
    --out "$QC_RUN/verify"
  conda activate xtbflow-qc
  srun --partition=main --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16G --time=00:20:00 \
    --output="$QC_RUN/minima-%j.out" --error="$QC_RUN/minima-%j.err" \
    python scripts/check_gqc_tautomer_minima.py \
    --config configs/research/gqc_tautomer_frequency_20261114.json \
    --verified "$QC_RUN/verify/report.json" --optimized "$QC_RUN/continue/report.json" \
    --gate "$QC_FIRST/qualify-openblas/report.json" --out "$QC_RUN/minima"
'
```

`QC_*` 变量必须在远端有实际值，不能依赖 SSH 自动继承本地变量。重放会重新消耗
计算预算；日常查看本轮结论直接读证据即可，不为更新日期重复真实计算。

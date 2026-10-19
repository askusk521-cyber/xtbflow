# 改用 CUDA 运行时记录 GPU 的决定（2026-10-17）

本文件记录执行决定，不是科学结果。

## 背景与更正

`INFRASTRUCTURE_ZH.md`（2026-10-10）认为 n2 上"所有 GPU 计算"都受驱动不匹配影响。这个判断**不对**：它只依据 `nvidia-smi` 的报错，没有实际做 CUDA 计算。

2026-10-17 经 Slurm 实测（`--gres=gpu:pro6000:1`）：

| 检查 | 结果 |
|---|---|
| `nvidia-smi` | 仍报 `Driver/library version mismatch`，退出码 18。NVML 用户态库 595.99，已加载内核模块 595.91.07 |
| PyTorch CUDA | 可用，矩阵乘法正常 |
| gpu4pyscf | 水分子 ωB97X/6-31G* SCF 收敛，解析 Hessian 正常 |
| 数值复现（Slurm 3371） | 用冻结的 `source-bb09c34` 中未改动的 `qc_protocol.single_point`，在 2752 锚点的最终几何上重算能量。存档 −245.96712539110587 Eh，重算 −245.96712539110592 Eh，差 5.7×10⁻¹⁴ Eh |

坏的只是 NVML 管理库，CUDA 计算正常。2825–2827 失败，是因为冻结执行脚本 `review_open_world_dft.py` 第 164 行在任何量化步骤之前，用 `nvidia-smi`（`check=True`）记录 GPU 型号。这三次仍属于量化前的基础设施失败，分类不变。

检查脚本与结果在 n2 的 `/home/lhshen/xtbflow-runs/gpu-check-20261017/`。这些检查约用 5 分钟 GPU，不计入任务 3a 的台账。

## 决定

验收方（Claude Code）提出两条路：

1. 等管理员重载驱动；
2. 做一个带版本号的最小改动。

负责人于 2026-10-17 明确选择第 2 条，原话："批准带版本号的最小改动：执行脚本和调度器改用 cupy 记录 GPU 信息并预检，qc_protocol、样本和设置都不变，然后继续剩下的 28 项。"

## 改了什么

- **`scripts/review_open_world_dft_v2.py`（执行脚本第 2 版）**：
  - 原样导入第 1 版并运行它的 `main()`；
  - 只把第 1 版里那一条 `nvidia-smi` 查询，改为从 CUDA 运行时（cupy）作答。每个可见设备记录一行"型号, 驱动版本, source=cuda-runtime (...)"，驱动版本读自 `/proc/driver/nvidia/version`，与 `nvidia-smi` 以前报告的是同一个值；
  - 作答前先做一次很小的 CUDA 计算。没有可见设备或计算结果不对时，照样在量化前失败。
- **`scripts/review_open_world_dispatch_v4.py`（调度器第 4 版）**：与第 3 版相比只改三处：
  - Slurm 内的预检改为运行 `review_open_world_dft_v2.py gpu-check`；
  - 每项改用第 2 版执行脚本；
  - 台账写入 `o2-jobs/ledger_v4.json`。
  - 第 1–3 版台账都计入 16 GPU·h 预算。其余规则全部从第 2、3 版导入，没有改动：结果分类、量化前失败的隔离、全用户串行、每作业预留 2 小时、记录在案的链异常继续、连续 3 次停下。
- **`tests/test_open_world_dft_v2.py`**：
  - 驱动版本解析；
  - 无设备或自检失败时报错；
  - 替身只截获那一条查询、其他调用（git 检查等）照常转发；
  - 截获的参数与冻结脚本中的调用逐字相同；
  - 调度器使用第 2 版执行脚本、预检不调用 `nvidia-smi`。

## 没有改的

- `qc_protocol.py`（协议 `v1b-h-gpu4pyscf-3`）。
- 冻结计划 `o2-plan.json` 及其输入哈希。
- 样本、SCF 与几何优化设置。
- 干净源码与 Slurm 守卫、每项只尝试一次。
- 已有脚本、台账和工作目录。
- 2753 仍是计算失败，不移动、不重跑。

## 执行方式

1. 从推送后的提交在 n2 新建干净克隆，跑完整测试。
2. 确认新克隆中的 `qc_protocol.py` 与计划记录的哈希相同。
3. 运行：

   ```bash
   python scripts/review_open_world_dispatch_v4.py --root /home/lhshen/xtbflow-runs/followup-20261009 \
     --source <新克隆> --quarantine-preqc
   ```

   `--quarantine-preqc` 把 2825–2827 留下的三个空工作目录移到 `o2-run-preqc-failed/`，让这三项第一次真正被尝试。

剩余 28 项预计约 11 GPU·h，在 16 GPU·h 上限内。

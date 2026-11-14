# n2 运行环境

核查日期：2026-09-30。以下内容来自 `ssh n2` 的只读观察，记录环境身份和边界，
不等同于训练或量化计算结果。

## 主机、仓库与调度

- 主机：`node2`；SSH 别名：`n2`。
- Git checkout：`~/xtbflow/xtbflow`。
- 观测分支：`codex/track-b-real-development`，checkout `fb6b234`，工作树干净。
- 远端主线：`origin/main@ec803556`；该 checkout 落后主线 3 个提交。
- 不直接重置已有工作树。真实作业从目标 commit 创建独立执行目录。
- Slurm：21.08.5；分区：`main`；GRES：`gpu:pro6000:4`；节点观测为 192 CPU、约
  500000 MiB 内存；本轮核查时没有排队作业。

## `xtbflow` Conda 环境

运行入口：

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate xtbflow
```

本轮观测版本：

| 软件 | 版本 |
| --- | --- |
| Python | 3.10.19 |
| PyTorch | 2.10.0+cu128 |
| NumPy | 1.26.4 |
| SciPy | 1.15.3 |
| RDKit | 2024.3.3 |
| ASE | 3.29.0 |
| tblite | 0.7.0 |
| pytest | 9.1.1 |
| PyYAML | 6.0.3 |

未通过 Slurm GPU 分配时，`torch.cuda.is_available()` 为 `False`，不能据此宣称 GPU
不可用或已完成 GPU 基准。正确的有限探针应申请明确 GPU 和时限，例如：

```bash
srun --partition=main --gres=gpu:pro6000:1 --nodes=1 --ntasks=1 \
  --time=00:05:00 bash -lc 'source ~/miniconda3/etc/profile.d/conda.sh; \
  conda activate xtbflow; python -c "import torch; \
  print(torch.cuda.get_device_name(0))"'
```

## 计算器边界

本轮 shell 的 PATH 中没有 `xtb`、`cp2k.psmp` 或 `cp2k.popt`。tblite 和 xTB Python
分发包的存在不等于独立 xTB 可执行文件、CP2K 环境或科学资格已验证。CP2K 任务必须
显式激活 `xtbflow-cp2k`，记录可执行文件、版本、协议哈希和持久化 artifact。

## 缓存与数据

`~/.cache/xtbflow` 约 3.6G，包含 Reaction-QM v2、RGD1 v6 和 Transition1x 缓存。
原始数据仅留在 n2 缓存；仓库只保存来源 URL、版本、文件大小、MD5/SHA-256、字段审计
和摘要。缓存中的记录没有自动获得 Track-B 准入，缺失 charge、multiplicity、坐标映射、
反应族或 overlap 证据的记录继续保持 `quarantine`。

## 可复现核查命令

```bash
ssh n2 'hostname; cd ~/xtbflow/xtbflow; git rev-parse HEAD; git rev-parse origin/main; \
  source ~/miniconda3/etc/profile.d/conda.sh; conda activate xtbflow; \
  python --version; python -c "import torch; print(torch.__version__)"; \
  command -v sbatch; command -v srun; squeue -u "$USER"'
```

每个真实阶段仍需另建带日期或运行 ID 的证据文件，写明执行主机、源码 SHA、配置／
协议哈希、来源资产哈希、环境版本、资源消耗、状态和 claim limits。


## 2026-11-11 integration observation

Issue #197 exact code c7e8b0bf49b7c2d09c170f5e81e232470933d948 ran in node2
`xtbflow`: Python 3.10.19, PyTorch 2.10.0+cu128, NumPy **2.2.6**, SciPy 1.15.3,
RDKit 2024.3.3, ASE 3.29.0, tblite 0.7.0, xTB distribution 22.1, pytest 9.1.1.
This observation supersedes the old NumPy 1.26.4 baseline for this run only.
Full software validation (419 pytest tests/68 subtests, 56 reusable tests)
and fake/cached integration pass. Receipt:
`docs/evidence/integration_197_20261111/software_c7e8b0b.json`.
No trained-checkpoint forwards or quantum chemistry were executed by this stage.

## 2026-11-14 GenerativeQC runtime and first tautomer pilot

The owner requested reuse of the compiled GenerativeQC program and Slurm GPU
execution. The reusable wheel/venv uses Python 3.11.17, NumPy 2.4.6, SciPy
1.17.1, PySCF 2.14.0 and CuPy CUDA12x 14.2.0. Its upstream source is
`2bf3fd98da9c6cfba46ca83345e63e1d934e4780`; main library SHA-256 is
`67bc98a4549fe8854ed6ae65791c824024bd77a9e8e498c6891dea3b8de670c4`.
The node's `latest` source bundle was not a complete runtime. Existing compiled
artifacts were preserved. The wheel required its existing scipy_openblas32
directory on LD_LIBRARY_PATH; the initial missing-library failure is archived.

One `gpu:pro6000:1` allocation at a time, partition `main`, ran a new bounded
B3LYP/def2-SVP force/energy gate and the tautomer optimization. CPU PySCF then
checked eight representative minima and an independent single-point method.
The `xtbflow` Python 3.10 environment was used for structure preparation and
identity checking; the GQC interpreter was explicitly selected for native QC.
These observations do not qualify other GQC methods, native Hessians, CP2K,
solution prediction or reaction connectivity.

Full outcomes, exact source commits, protocols, failed-work costs, environment
and commands: [pilot evidence](evidence/gqc_tautomer_20261114/RESULTS_ZH.md).
Slurm accounting storage was disabled; program time is not relabelled as exact
allocated GPU time. A private n2 runtime.env and the reusable
`skills/n2-quantum-chemistry/` instructions preserve the verified invocation.

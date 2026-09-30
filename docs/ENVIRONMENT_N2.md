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

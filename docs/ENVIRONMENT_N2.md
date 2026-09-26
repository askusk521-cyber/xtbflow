# n2 运行环境

这份记录描述 `xtbflow` 在 n2 上的实际接管环境。它记录已观察到的软件和硬件身份，不代表 CP2K、xTBloom 或训练流程已经安装，也不构成科学结果。

## 代码与主机

- 远程仓库：`git@github.com:askusk521-cyber/xtbflow.git`
- 分支：`main`
- 工作目录：`/home/lhshen/xtbflow/xtbflow`
- SSH 别名：`n2`；主机名：`node2`
- 操作系统内核：Ubuntu 22.04.1，Linux 6.8.0-138-generic
- 计算调度：Slurm 21.08.5，分区 `main`
- 节点资源：192 个逻辑 CPU，约 500000 MiB 内存；磁盘使用量和配额尚未完成独立配额核验

## GPU

Slurm 的实际 GRES 名称是 `gpu:pro6000:4`。节点上观察到 4 张：

- NVIDIA RTX PRO 6000 Blackwell Workstation Edition
- 驱动：595.91.07
- 显存：97887 MiB/卡（约 96 GiB）

申请一张卡的示例：

```bash
srun --partition=main --gres=gpu:pro6000:1 --nodes=1 --ntasks=1 \
  --time=00:05:00 bash -lc 'source ~/miniconda3/etc/profile.d/conda.sh; \
  conda activate xtbflow; python -c "import torch; \
  print(torch.cuda.get_device_name(0))"'
```

## Python 环境

专用 Conda 环境名称是 `xtbflow`，环境快照见 [`configs/environment-n2.yml`](../configs/environment-n2.yml)。关键版本如下：

- Python 3.10.19
- PyTorch 2.10.0+cu128，CUDA runtime 12.8
- NumPy 1.26.4，SciPy 1.15.3
- RDKit 2024.03.3
- ASE 3.29.0
- PyYAML 6.0.3，pytest 9.1.1，pydantic 2.13.5
- CMake 4.3.0，Ninja 1.13.2

当前节点没有系统 `nvcc`、CP2K 或已验证的 xTBloom 安装；PyTorch CUDA runtime 已在 Slurm 分配的 GPU 上完成运行时冒烟测试。

## 已完成验证

在仓库根目录执行：

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate xtbflow
PYTHONPATH=vendor/mechai_reusable pytest -q tests/reusable
python scripts/validate_bootstrap.py
```

结果：`56 passed, 68 subtests passed`；完整性检查通过。GPU 冒烟测试使用 `srun` 分配一张 `pro6000`，`torch.cuda.is_available()` 为真，设备名为 RTX PRO 6000，1024×1024 CUDA 矩阵乘法成功。

## 后续使用约定

- 普通 CPU/Python 检查使用 `xtbflow` 环境。
- 真实 GPU 测试、基准和分析通过 Slurm `main` 分区申请 `gpu:pro6000:1`，设置有限的 `--time`，保留 Slurm 设置的 `CUDA_VISIBLE_DEVICES`。
- 主机私有路径写在被忽略的 `configs/host.local.yaml`；凭据、私钥、原始数据和模型权重不上传到 Git。

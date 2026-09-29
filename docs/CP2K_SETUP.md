# xtbflow 的 CP2K 配置

当前执行环境使用独立的 `xtbflow-cp2k` Conda 环境，避免 CP2K 的 MPI/Fortran
运行库改动训练环境 `xtbflow`。安装规格见
[`configs/calculators/cp2k/environment.yml`](../configs/calculators/cp2k/environment.yml)，
物理协议候选见
[`configs/calculators/cp2k/protocol_v0.1.yaml`](../configs/calculators/cp2k/protocol_v0.1.yaml)。

## 安装或重建

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda env create -f configs/calculators/cp2k/environment.yml
conda activate xtbflow-cp2k
cp2k.psmp -v
```

CP2K 数据文件由同一 Conda 包提供，不需要另行下载：

- `BASIS_MOLOPT`
- `GTH_POTENTIALS`
- `dftd3.dat`

## xtbflow 运行约定

CP2K 输入渲染器对孤立分子固定使用 `PSOLVER MT`，并显式请求
`&PRINT / &FORCES`。CP2K 输出的力是 Hartree/Bohr，适配器转换为 xtbflow
合同中的 Hartree/Å。每个体系仍必须显式给出总电荷和多重度。

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate xtbflow
export XTBFlow_CP2K_EXECUTABLE=~/miniconda3/envs/xtbflow-cp2k/bin/cp2k.psmp
python scripts/cp2k_smoke.py
```

目标域的最小状态/元素校准使用三个独立协议身份：中性单重态 H₂O、带电双重态
H₂⁺ 和中性单重态 H₂S。它只检查 CP2K 安装、输入渲染、收敛输出和 E/F 单位，
不会把结果自动升级为正式参考标签：

```bash
python scripts/cp2k_calibration_smoke.py \
  --threads 4 \
  --output docs/evidence/cp2k_calibration_smoke.json
```

本机已完成该套件；结果保存在
[`cp2k_calibration_smoke.json`](evidence/cp2k_calibration_smoke.json)，三个体系均返回
`success`。这仍然不包括 cutoff/SCF 收敛曲线、过渡态路径或溶液自由能验证。

Slurm CPU 入口见
[`slurm_calibration_smoke.sbatch`](../configs/calculators/cp2k/slurm_calibration_smoke.sbatch)。

当前工作候选使用 `CUTOFF 600 Ry`、`REL_CUTOFF 50 Ry`。600→700 Ry 的水分子
对照显示最大力差约 `6.1×10⁻⁵ Ha/Å`，但绝对能量差约 `3.5×10⁻⁵ Ha`，因此
协议仍标记为 `provisional`，没有宣称严格能量收敛。变更这些数值前运行
[`cp2k_convergence_smoke.py`](../scripts/cp2k_convergence_smoke.py)，它会把每个
cutoff 的能量、力和运行时间写入证据文件：

```bash
python scripts/cp2k_convergence_smoke.py \
  --cutoffs 500 600 700 \
  --output docs/evidence/cp2k_convergence_smoke.json
```

本次 600→700 Ry 结果见
[`cp2k_convergence_600_700.json`](evidence/cp2k_convergence_600_700.json)。
`status: fail` 只表示严格的绝对能量阈值尚未满足；两次 CP2K 运行本身均成功。

`installation.status` 只表示运行时已安装；进入正式参考标签前仍需完成同构型
能量/力、收敛阈值和路径验证。

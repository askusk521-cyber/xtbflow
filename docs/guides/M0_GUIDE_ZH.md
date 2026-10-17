# xtbflow M0 阶段执行指导：联合与串联的去/留闸门

Oct 7, 2026 · @askusk

## 0. 执行代理须知

M0 只回答一个问题：在参数量、训练步数和数据完全相同的条件下，联合生成（B 臂）是否优于串联生成（A 臂）。本方案写给执行代理（Sonnet 5 级别或更低）和项目负责人，两周内必须给出“去”或“留”的结论。

**读法。** 按章节顺序执行任务 0 到任务 4。每个任务都有“输入、步骤、代码、验收命令、停止条件”。代码块是规格，不是参考：函数名、张量形状、默认超参都按块里写的实现。块里标 `# TODO(agent)` 的地方才由代理自行补全。

**硬性规则：**

1. 只做本方案列出的任务。M0 期间不新增治理类基础设施：不加新的合同 schema、审计脚本、预算账本字段，也不改 issue 模板。
2. 每个任务一个分支、一个 PR。PR 描述必须贴出验收命令的完整输出，以及 `git diff --stat`。
3. 任何数字结果都必须来自实际运行生成的日志或 JSON 文件。不得估计，不得手填，不得从旧 PR 复制。
4. A 臂和 B 臂共用一套超参和一个训练脚本。两臂之间只允许存在第 5 节“两臂差异表”列出的差异，其他差异一律视为实验缺陷。
5. 不允许按测试集结果调参。超参只能看验证集；测试集在第 7 节规定的时刻只评测一次。
6. 随机种子固定为 0、1、2。每个种子的模型、配置和日志各存一个目录，不覆盖。
7. 单元测试必须在 CPU 上 5 分钟内跑完；GPU 训练只在 n2 上通过 ssh 运行。

**必须停下来问负责人的情况（停止条件）：**

- 任一验收命令连续两次修复后仍失败。
- 需要修改本方案规定的接口、张量形状或评测定义。
- 任一臂在 3 个种子中有 2 个出现 NaN 或训练损失不下降（第 9 节有判据）。
- 发现数据泄漏：同一个反应物母体同时出现在训练集和测试集。
- 单次训练超过 12 GPU 小时，或总 GPU 用量超过 150 GPU 小时。

**术语：** R = 反应物，P = 产物，TS = 过渡态。“事件” = R 到 P 的成键/断键变化。“母体（parent）” = 同一个反应物分子；同一母体的所有反应必须在同一个数据切分里。

## 1. 问题、两个对照臂与去/留判据

判据只看一个主指标：在 Transition1x 官方测试集上，B 臂（联合）的“事件＋几何同时命中率”比 A 臂（串联）高，且按母体配对 bootstrap 的 95% 置信区间下限大于 0，就判“去”。其余指标只用于解释，不参与决策。

### 1.1 任务定义（单端）

- **输入（推理时允许）：** 元素序号 Z、反应物三维坐标 X\_R、由 X\_R 推断出的反应物键—电子矩阵 b\_R、总电荷 0、自旋多重度 1。
- **输出：** 每个反应物采样 N = 32 个候选。每个候选包含产物键—电子矩阵 b̂\_P（整数，经仓库现有 `decode_endpoint` 修复和校验）和过渡态坐标 X̂\_TS（与 X\_R 原子顺序相同）。
- **推理时禁止输入：** 产物几何、TS 几何、产物 b\_P、反应模板、该反应物已知反应的数目 K。代码里必须有一个断言，确保采样函数的参数里不出现这些字段。

### 1.2 两个对照臂

| 项目 | A 臂：串联 | B 臂：联合 |
| --- | --- | --- |
| 网络个数 | 2 个：事件网络 g、几何网络 h | 1 个：f |
| 骨干 | 与 B 同一个类、同样 6 层，通道数缩小 | 6 层，128 个标量通道，32 个向量通道 |
| 参数量 | g 与 h 合计等于 f 的参数量，误差 ±5% | 基准 |
| 训练 | g 学 b\_0→b\_P；h 在真实 b\_P 条件下学 X\_0→X\_TS；两者分别训练 | f 同时学 b\_0→b\_P 与 X\_0→X\_TS，两个模态共用同一个时间 t |
| 采样 | 先积分 g 得到 b̂\_P 并解码；再把 b̂\_P 作为条件积分 h | 事件和几何两个状态在每一步同步积分，最后解码事件 |
| 每个候选的网络调用 | g 50 次 + h 50 次（每次约半个网络） | f 50 次 |

可选第三臂 A2x：g 和 h 各用与 f 相同的宽度（总参数约为 B 的 2 倍），只跑种子 0。它只用来判断差距是否来自容量，不参与判据。

### 1.3 主指标

对测试集中的每个反应 j，从它自己的反应物坐标采样 32 个候选。若至少有一个候选同时满足下面两条，则 Hit\_j = 1，否则为 0：

1. 事件匹配：候选的成键/断键集合与参考反应相同，允许差一个反应物分子自同构（例如甲基上 3 个氢互换）。定义见第 7 节。
2. 几何匹配：在同一个自同构下，候选 TS 与参考 TS 的 Kabsch 对齐 RMSD（含氢，单位 Å）不超过 δ = 0.5 Å。

主指标 M 是“母体宏平均”：先在每个母体（同一个反应物分子）内对 Hit\_j 求平均，再对所有母体求平均。3 个种子先在母体层面平均，再进入 bootstrap。

### 1.4 判据

令 Δ = M\_B − M\_A。对测试母体做配对 bootstrap 10,000 次（两臂用同一组重采样母体），取 95% 百分位区间 \[L, U\]。

| 结论 | 条件 | 后续动作 |
| --- | --- | --- |
| 去（GO） | L > 0；3 个种子中至少 2 个的 Δ > 0；有效率差（B − A）的区间下限大于 −0.02 | 进入 M1：全量数据、守恒消融、外部基线 |
| 放弃联合卖点（NO-GO） | U < 0.02，或 Δ 的点估计 ≤ 0 | 转向方向 B（水相 TS 数据集加基准）；M0 结果作为负结果保留 |
| 不确定 | 其他情况 | 允许延长一次，最多 4 天：加种子 3、4，并把 N 提到 64。延长后仍未满足“去”，按 NO-GO 处理 |

### 1.5 次要指标（只报告，不决策）

- 事件召回：只看事件匹配的 Hit\_j。
- 有效率：32 个候选中，解码成功、RDKit 能消毒（sanitize）、且确实发生了键变化的比例。
- 命中反应上的 TS RMSD 中位数。
- 事件—几何一致性：候选自己预测的成键对在 TS 中变短、断键对变长的比例。
- 每个反应物产生的不同有效事件数。
- Hit 随 δ 变化的曲线（δ = 0.1 到 1.0 Å，步长 0.1）。另报 δ = 0.3 和 1.0 Å 时结论是否翻转。

与上一版评估报告的差异：报告里写的是“top-k 事件准确率或 TS 有效率”。这里改为单一主指标，避免多重比较；xTB 过渡态验证移到 M1，M0 用参考 DFT TS 的 RMSD 代替。

## 2. 仓库现状与文件地图

现有原型不能直接用于 M0，因为它是确定性的单层模型，采样时 64 个候选完全相同；M0 在新目录 `src/xtbflow/m0/` 里重建模型和训练，复用仓库的键—电子工具与解码器，不改动原型。下表是读代码（主分支 d6220b8，2026-10-06）得到的事实。

### 2.1 现有原型的问题

| 位置 | 现状 | 对 M0 的影响 |
| --- | --- | --- |
| `src/xtbflow/models/joint_flow.py` | 事件分支是一层成对 MLP，几何分支是一层成对消息，隐藏宽度 64；`configs/models/joint_flow_runtime_v0.1.json` 自己写明“尚未实现 6 层消息传递” | 容量太小，学不了 Transition1x 的 TS 几何 |
| `scripts/run_dft_da_joint_training.py` 的 `_loss` | 起点就是反应物本身（b\_0 = b\_R，X\_0 = X\_R），没有噪声；时间 τ 用 `linspace(0.15, 0.85, batch)`，与样本在批次中的位置绑定 | 学到的是一个确定性映射，一个反应物只能生成一个 TS；单端任务要求同一反应物生成多个通道 |
| `src/xtbflow/evaluation/blind_rollout.py` | `initial_event_noise_scale` 和 `initial_geometry_noise_scale` 默认 0，NFE 默认 4 | 64 个候选去重后只剩 1 个 |
| `forward_control("serial_independent")` | 与联合模式共用同一套参数，只是把事件先推进一个 Euler 步再给几何 | 这不是真正的串联 p(E |
| `ConservationProjector.project_masked` | 每个样本在 Python 循环里调用一次 `pinv` | 太慢；只有“总电子数”一个约束时有闭式解（第 6 节） |
| issue #44（坐标反传 NaN） | 已关闭；距离已改为 `sqrt(x + 1e-12)`，CI 已覆盖整个 `tests/` | 任务 0 只需验证，不需要再修 |
| Transition1x | 仓库审计的是 React-OT 预处理文件 `train_rpsb_all.pkl`，因许可证不明被标为“仅诊断” | M0 改用官方 HDF5（figshare，MIT 许可证），见第 4 节 |

### 2.2 复用、不改动的模块

- `xtbflow.models.conserved_be`：`pack_be`、`unpack_be`、`upper_triangle_indices`、`upper_triangle_weights`。
- `xtbflow.models.event_decoder.decode_endpoint`：把连续的 b 修复为整数并校验。
- `mechai.data.events.LewisState` 与 `VALENCE_ELECTRONS`：键—电子矩阵的约定是对角元为非键电子数（偶数），非对角元为键级，整矩阵求和等于价电子总数减电荷。
- 不要修改：`src/xtbflow/models/` 下的原型文件、`calculators/`、`runtime/`、`governance/`、`vendor/`、`archive/`。

### 2.3 新增文件清单

| 文件 | 内容 | 所在任务 |
| --- | --- | --- |
| `src/xtbflow/m0/__init__.py` | 空文件，标记包 | 1 |
| `src/xtbflow/m0/t1x_data.py` | 读 HDF5、推断键级、构造 b\_R 与 b\_P、母体编号、CSR 缓存读写 | 1 |
| `src/xtbflow/m0/batching.py` | 把不同原子数的反应补零成批，生成掩码 | 1 |
| `src/xtbflow/m0/flow.py` | 守恒投影闭式解、噪声、插值、速度目标、损失 | 2 |
| `src/xtbflow/m0/backbone.py` | 带边特征的等变标量—向量骨干（PaiNN 式），6 层 | 2 |
| `src/xtbflow/m0/model.py` | `ReactionFlowNet`、三种角色、参数量对齐 | 2 |
| `src/xtbflow/m0/sampler.py` | 两臂的 Euler 积分与事件解码 | 3 |
| `src/xtbflow/m0/metrics.py` | 自同构、事件匹配、RMSD、Hit、母体 bootstrap | 4 |
| `scripts/m0_prepare_t1x.py` | 生成缓存与数据清单 | 1 |
| `scripts/m0_toy_multimodal.py` | 两模态玩具问题，检查采样不塌缩 | 3 |
| `scripts/m0_train.py` | 训练任一角色 | 3 |
| `scripts/m0_sample.py` | 对验证或测试集采样，写候选文件 | 3 |
| `scripts/m0_evaluate.py` | 计算全部指标与 bootstrap | 4 |
| `scripts/m0_gate_report.py` | 生成最终闸门报告 | 4 |
| `configs/m0/base.json` | 全部超参（两臂共用） | 3 |
| `tests/test_m0_*.py` | 每个模块的单元测试 | 各任务 |

另外两处必改：`pyproject.toml` 的 `packages` 列表里加上 `"xtbflow.m0"`（仓库用显式包列表，不加就装不进去）；每次新增跟踪文件后运行 `python scripts/validate_bootstrap.py --refresh` 更新清单。

**数据与产物存放：** Transition1x 原始文件和缓存不进仓库，放在 n2 的 `$XTBFLOW_DATA/t1x/`（建议 `/home/lhshen/data/t1x`）。模型权重放在 `$XTBFLOW_RUNS/m0/`，同样不进仓库。仓库只提交小于 1 MB 的 JSON 结果，放在 `docs/evidence/m0/`。

## 3. 任务 0：环境体检与数值安全原语（第 1 天）

issue #44 已经修复并关闭，任务 0 不再修 bug，而是确认现有测试全绿，并写好 M0 所有代码都要用的“不会产生 NaN 的成对几何”函数。

**分支：** `m0/task0-env`。

### 3.1 步骤

1. 在开发机上：`git checkout main && git pull && git checkout -b m0/task0-env`。
2. 安装：`python -m pip install -e ".[test,data]"`。确认 `python -c "import torch; print(torch.__version__)"` 输出 2.10.x（`pyproject.toml` 要求 `torch>=2.10,<2.11`）。
3. 跑现有两套测试，记录通过数：
   1. `PYTHONPATH=src:vendor/mechai_reusable pytest -q`
   2. `PYTHONPATH=src:vendor/mechai_reusable python -m unittest discover -s tests/reusable -p 'test_*.py'`
4. 单独确认 #44 的回归测试：`pytest -q tests/test_joint_flow.py -k collision` 必须通过。
5. 在 n2 上：`nvidia-smi`，以及 `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"`。把输出写进 `docs/evidence/m0/env_n2.json`，字段为 `hostname`、`gpu`、`torch`、`cuda_available`、`git_sha`、`date`。
6. 新建 `src/xtbflow/m0/__init__.py`（空）和 `src/xtbflow/m0/numerics.py`（代码见 3.2），在 `pyproject.toml` 的 `packages` 里加 `"xtbflow.m0"`。
7. 新建 `tests/test_m0_numerics.py`（代码见 3.3），运行通过后执行 `python scripts/validate_bootstrap.py --refresh`。

### 3.2 `src/xtbflow/m0/numerics.py`

NaN 的根源是 sqrt 在 0 处导数无穷大，以及 `torch.where` 被屏蔽分支里的 inf 在反传时变成 0×inf。下面的写法先把屏蔽位置换成 1 再开方，从源头避免两种情况。M0 里所有距离、单位向量都必须调用这个函数，禁止在别处手写 `norm()` 或 `sqrt()`。

```python
"""NaN-safe pair geometry shared by every M0 module."""
from __future__ import annotations

import torch
from torch import Tensor

EPS_DIST2 = 1.0e-8  # Å^2，距离下限约 1e-4 Å


def pair_mask_from_atoms(atom_mask: Tensor) -> Tensor:
    """[B,N] bool -> [B,N,N] bool, True 表示 i != j 且两者都是真实原子。"""
    n = atom_mask.shape[1]
    eye = torch.eye(n, dtype=torch.bool, device=atom_mask.device)[None]
    return atom_mask[:, :, None] & atom_mask[:, None, :] & ~eye


def pair_geometry(x: Tensor, atom_mask: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """x: [B,N,3] float; atom_mask: [B,N] bool.

    Returns (rel, dist, unit, pair_mask):
      rel[b,i,j]  = x[b,j] - x[b,i]                 [B,N,N,3]
      dist[b,i,j] = |rel|，被屏蔽的位置为 0          [B,N,N]
      unit[b,i,j] = rel/|rel|，被屏蔽的位置为 0       [B,N,N,3]
      pair_mask                                      [B,N,N] bool
    """
    if x.ndim != 3 or x.shape[-1] != 3:
        raise ValueError("x must be [B,N,3]")
    pair_mask = pair_mask_from_atoms(atom_mask)
    rel = x[:, None, :, :] - x[:, :, None, :]
    d2 = rel.square().sum(dim=-1)
    d2 = torch.where(pair_mask, d2, torch.ones_like(d2))  # 先替换，再开方
    dist = torch.sqrt(d2 + EPS_DIST2)
    unit = rel / dist[..., None]
    zero = torch.zeros_like(dist)
    dist = torch.where(pair_mask, dist, zero)
    unit = torch.where(pair_mask[..., None], unit, torch.zeros_like(unit))
    return rel, dist, unit, pair_mask


def masked_mean(x: Tensor, mask: Tensor, dim: int) -> Tensor:
    """对 mask 为 True 的元素求平均；mask 会广播到 x 的形状。"""
    m = mask.to(x.dtype)
    while m.ndim < x.ndim:
        m = m[..., None]
    return (x * m).sum(dim=dim) / m.sum(dim=dim).clamp_min(1.0)


def remove_masked_com(x: Tensor, atom_mask: Tensor) -> Tensor:
    """减去真实原子的质心（几何中心），填充原子保持为 0。x: [B,N,3]。"""
    com = masked_mean(x, atom_mask, dim=1)  # [B,3]
    return (x - com[:, None, :]) * atom_mask[..., None].to(x.dtype)
```

### 3.3 `tests/test_m0_numerics.py`

```python
import math

import torch

from xtbflow.m0.numerics import pair_geometry, remove_masked_com


def _rotation(seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(3, 3, generator=g, dtype=torch.float64))
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def test_gradients_finite_with_coincident_and_padded_atoms():
    x = torch.randn(2, 6, 3, dtype=torch.float64)
    x[0, 1] = x[0, 0]            # 两个真实原子重合
    x[1, 4:] = 0.0               # 填充原子坐标为 0
    x.requires_grad_(True)
    mask = torch.tensor([[True] * 6, [True] * 4 + [False] * 2])
    _, dist, unit, pm = pair_geometry(x, mask)
    rbf = torch.exp(-((dist[..., None] - torch.linspace(0, 6, 16, dtype=x.dtype)) ** 2))
    loss = dist.sum() + unit.sum() + (rbf * pm[..., None]).sum()
    loss.backward()
    assert torch.isfinite(x.grad).all()
    assert torch.all(x.grad[1, 4:] == 0)


def test_gradcheck_generic_positions():
    x = torch.randn(1, 5, 3, dtype=torch.float64, requires_grad=True)
    mask = torch.ones(1, 5, dtype=torch.bool)
    f = lambda y: pair_geometry(y, mask)[1].sum() + pair_geometry(y, mask)[2].square().sum()
    assert torch.autograd.gradcheck(f, (x,))


def test_distances_invariant_and_units_equivariant_under_rotation():
    x = torch.randn(1, 7, 3, dtype=torch.float64)
    mask = torch.ones(1, 7, dtype=torch.bool)
    r = _rotation(0)
    _, d1, u1, _ = pair_geometry(x, mask)
    _, d2, u2, _ = pair_geometry(x @ r.T, mask)
    assert torch.allclose(d1, d2, atol=1e-10)
    assert torch.allclose(u1 @ r.T, u2, atol=1e-10)


def test_remove_com_keeps_padding_zero():
    x = torch.randn(1, 5, 3)
    mask = torch.tensor([[True, True, True, False, False]])
    y = remove_masked_com(x, mask)
    assert torch.allclose(y[0, :3].mean(0), torch.zeros(3), atol=1e-6)
    assert torch.all(y[0, 3:] == 0)
```

### 3.4 验收

- 现有两套测试全部通过，通过数写进 PR。
- `pytest -q tests/test_m0_numerics.py` 4 个测试通过。
- `docs/evidence/m0/env_n2.json` 存在，`cuda_available` 为 true。
- PR 描述按 `AGENTS.md` 要求附“中文说明”，不合并，等负责人批准。

## 4. 任务 1：Transition1x 数据管线（第 1–3 天）

用官方 HDF5 和官方切分（按分子式划分，验证集和测试集各约 5%），在开发机上一次性生成一个小于 50 MB 的缓存文件，再复制到 n2。训练端只需要 torch 和 numpy。

**分支：** `m0/task1-data`。**依赖：** 开发机装 `h5py` 与 `rdkit`（`pip install -e ".[data]"`）。

### 4.1 下载（开发机，n2 网络受限）

文件 6,623,929,000 字节，MIT 许可证，[figshare 记录](https://api.figshare.com/v2/articles/19614657)。

```bash
mkdir -p "$XTBFLOW_DATA/t1x" && cd "$XTBFLOW_DATA/t1x"
curl -L -o Transition1x.h5 https://ndownloader.figshare.com/files/36035789
stat -c %s Transition1x.h5          # 必须等于 6623929000
sha256sum Transition1x.h5 | tee Transition1x.h5.sha256
```

### 4.2 先看清文件结构，再写读取代码

下面的键名来自 Transition1x 论文和官方加载器的写法，**必须先打印确认**。若实际结构与预期不同，按实际结构改 `iter_split`，并把差异写进 PR；若找不到反应物/TS/产物三个子组，停下来问负责人。

```python
import h5py
path = "Transition1x.h5"
with h5py.File(path, "r") as f:
    print("root:", list(f.keys()))                  # 预期含 data、train、val、test
    split = f["train"]
    formula = sorted(split.keys())[0]
    rxn = sorted(split[formula].keys())[0]
    print(formula, rxn)
    split[formula][rxn].visititems(lambda name, obj: print(name, getattr(obj, "shape", "")))
```

预期输出：反应组下有 `atomic_numbers`（形状 (N,)）、`positions`（整条 NEB 路径）、能量与力数据集，以及 `reactant`、`transition_state`、`product` 三个子组；子组里 `positions` 形状为 (1, N, 3)，能量键名类似 `wB97x_6-31G(d).energy`（单位 eV）。把打印结果原样存为 `docs/evidence/m0/t1x_layout.txt`。

### 4.3 处理规则（每个反应）

1. 元素只能是 H、C、N、O，否则丢弃（原因码 `element_out_of_scope`）。
2. 读反应物、TS、产物坐标（Å）和三者能量。
3. 反应物和产物各自用 RDKit `DetermineBonds(charge=0)` 推断键级，然后 `Kekulize`。出现自由基、净电荷不为 0、原子顺序改变，均丢弃。
4. 构造键—电子矩阵 b：非对角元为键级（1、2、3），对角元为非键电子数 = 价电子数 − 形式电荷 − 键级之和。对角元必须是非负偶数，并通过仓库的 `LewisState.validate()`。
5. 事件：连接性变化集合 C = {(i, j), i < j：反应物中有键与产物中有键不一致}。C 为空则丢弃（`no_connectivity_change`）。只有键级变化、连接性不变的键记为 `order_only_changes`，只统计不丢弃。
6. 坐标：反应物减去几何中心；TS 与产物用 Kabsch（只允许真旋转，不允许镜像）对齐到反应物上。
7. 母体编号：反应物去氢后的规范 SMILES（不含立体）取 SHA1 前 12 位。
8. 切分：直接用 HDF5 的 `train`、`val`、`test` 组。断言三者的分子式两两不相交、母体编号两两不相交。

### 4.4 `src/xtbflow/m0/t1x_data.py`

```python
"""Transition1x -> M0 CSR cache.  Building needs h5py + rdkit; loading needs numpy only."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterator

import numpy as np

SPLITS = {"train": 0, "val": 1, "test": 2}
SYMBOL = {1: "H", 6: "C", 7: "N", 8: "O"}
VALENCE_E = {1: 1, 6: 4, 7: 5, 8: 6}
POS_KEY = "positions"
ENERGY_KEY = "wB97x_6-31G(d).energy"  # 以 4.2 打印出的实际键名为准


class DropReaction(Exception):
    """Raised to exclude one reaction; str(exc) is the reason code."""


@dataclass(frozen=True)
class RawReaction:
    split: str
    formula: str
    rxn: str
    z: np.ndarray      # [N] int64
    x_r: np.ndarray    # [N,3] float64, Å
    x_ts: np.ndarray
    x_p: np.ndarray
    e_r: float         # eV
    e_ts: float
    e_p: float


def iter_split(h5_path: Path, split: str, limit: int | None = None) -> Iterator[RawReaction]:
    import h5py

    count = 0
    with h5py.File(h5_path, "r") as f:
        root = f[split]
        for formula in sorted(root.keys()):
            for rxn in sorted(root[formula].keys()):
                g = root[formula][rxn]
                z = np.asarray(g["atomic_numbers"], dtype=np.int64).reshape(-1)

                def frame(name: str) -> tuple[np.ndarray, float]:
                    sub = g[name]
                    x = np.asarray(sub[POS_KEY], dtype=np.float64)
                    x = x[0] if x.ndim == 3 else x
                    e = float(np.asarray(sub[ENERGY_KEY], dtype=np.float64).reshape(-1)[0])
                    return x, e

                x_r, e_r = frame("reactant")
                x_ts, e_ts = frame("transition_state")
                x_p, e_p = frame("product")
                yield RawReaction(split, formula, rxn, z, x_r, x_ts, x_p, e_r, e_ts, e_p)
                count += 1
                if limit is not None and count >= limit:
                    return


def kabsch_align(mobile: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Rotate (proper rotation only) and translate `mobile` onto `ref`.
    Returns coordinates centred at ref's geometric centre."""
    ref_c = ref - ref.mean(axis=0)
    mob_c = mobile - mobile.mean(axis=0)
    h = mob_c.T @ ref_c
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rot = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return mob_c @ rot.T + ref.mean(axis=0)


def perceive_mol(z: np.ndarray, x: np.ndarray):
    from rdkit import Chem
    from rdkit.Chem import rdDetermineBonds

    lines = [str(len(z)), ""] + [
        f"{SYMBOL[int(a)]} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}" for a, p in zip(z, x)
    ]
    mol = Chem.MolFromXYZBlock("\n".join(lines) + "\n")
    if mol is None:
        raise DropReaction("xyz_parse_failed")
    try:
        rdDetermineBonds.DetermineBonds(mol, charge=0)
        Chem.Kekulize(mol, clearAromaticFlags=True)
    except Exception:  # RDKit 不同版本抛不同异常类型
        raise DropReaction("determine_bonds_failed") from None
    if any(a.GetNumRadicalElectrons() for a in mol.GetAtoms()):
        raise DropReaction("radical")
    if sum(a.GetFormalCharge() for a in mol.GetAtoms()) != 0:
        raise DropReaction("net_charge")
    if [a.GetAtomicNum() for a in mol.GetAtoms()] != [int(v) for v in z]:
        raise DropReaction("atom_order_changed")
    return mol


def be_matrix(mol) -> np.ndarray:
    n = mol.GetNumAtoms()
    b = np.zeros((n, n), dtype=np.int64)
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        order = bond.GetBondTypeAsDouble()
        if order not in (1.0, 2.0, 3.0):
            raise DropReaction("non_integer_bond_order")
        b[i, j] = b[j, i] = int(order)
    bond_sum = b.sum(axis=1)  # 此时对角线为 0
    for atom in mol.GetAtoms():
        i = atom.GetIdx()
        lone = VALENCE_E[atom.GetAtomicNum()] - atom.GetFormalCharge() - int(bond_sum[i])
        if lone < 0 or lone % 2:
            raise DropReaction("bad_lone_electrons")
        b[i, i] = lone
    return b


def _validate_lewis(z: np.ndarray, b: np.ndarray) -> None:
    from mechai.data.events import LewisState

    symbols = tuple(SYMBOL[int(v)] for v in z)
    rows = tuple(tuple(int(v) for v in row) for row in b)
    try:
        LewisState(symbols, rows, 0, 1).validate()
    except ValueError:
        raise DropReaction("lewis_invalid") from None


def connectivity_change(b_r: np.ndarray, b_p: np.ndarray) -> np.ndarray:
    """Upper-triangle pairs (i<j) whose bonded/non-bonded status changes. Returns [K,2] int."""
    changed = (b_r > 0) != (b_p > 0)
    changed = np.triu(changed, k=1)
    return np.argwhere(changed)


def process_reaction(raw: RawReaction) -> dict:
    from rdkit import Chem

    if not {int(v) for v in raw.z} <= set(SYMBOL):
        raise DropReaction("element_out_of_scope")
    n = len(raw.z)
    for x in (raw.x_r, raw.x_ts, raw.x_p):
        if x.shape != (n, 3) or not np.isfinite(x).all():
            raise DropReaction("bad_coordinates")
    mol_r = perceive_mol(raw.z, raw.x_r)
    mol_p = perceive_mol(raw.z, raw.x_p)
    b_r, b_p = be_matrix(mol_r), be_matrix(mol_p)
    _validate_lewis(raw.z, b_r)
    _validate_lewis(raw.z, b_p)
    pairs = connectivity_change(b_r, b_p)
    if len(pairs) == 0:
        raise DropReaction("no_connectivity_change")
    off = ~np.eye(n, dtype=bool)
    order_only = int(np.triu((b_r != b_p) & off & ((b_r > 0) == (b_p > 0)), k=1).sum())
    x_r = raw.x_r - raw.x_r.mean(axis=0)
    try:
        smiles_r = Chem.MolToSmiles(Chem.RemoveHs(mol_r), isomericSmiles=False)
        smiles_p = Chem.MolToSmiles(Chem.RemoveHs(mol_p), isomericSmiles=False)
    except Exception:
        raise DropReaction("smiles_failed") from None
    return {
        "split": SPLITS[raw.split],
        "formula": raw.formula,
        "rxn_key": f"{raw.formula}/{raw.rxn}",
        "z": raw.z.astype(np.int8),
        "x_r": x_r.astype(np.float32),
        "x_ts": kabsch_align(raw.x_ts, x_r).astype(np.float32),
        "x_p": kabsch_align(raw.x_p, x_r).astype(np.float32),
        "b_r": b_r.astype(np.int8),
        "b_p": b_p.astype(np.int8),
        "energies": np.array([raw.e_r, raw.e_ts, raw.e_p], dtype=np.float64),
        "smiles_r": smiles_r,
        "smiles_p": smiles_p,
        "parent_id": hashlib.sha1(smiles_r.encode()).hexdigest()[:12],
        "n_changed": len(pairs),
        "order_only": order_only,
        "any_formal_charge": any(a.GetFormalCharge() != 0 for m in (mol_r, mol_p) for a in m.GetAtoms()),
    }


def write_cache(records: list[dict], path: Path) -> None:
    n_atoms = np.array([len(r["z"]) for r in records], dtype=np.int64)
    atom_off = np.concatenate([[0], np.cumsum(n_atoms)])
    pair_off = np.concatenate([[0], np.cumsum(n_atoms ** 2)])
    cat = lambda key: np.concatenate([r[key].reshape(-1) for r in records])
    np.savez_compressed(
        path,
        n_atoms=n_atoms, atom_off=atom_off, pair_off=pair_off,
        z=cat("z"),
        x_r=cat("x_r").reshape(-1, 3), x_ts=cat("x_ts").reshape(-1, 3), x_p=cat("x_p").reshape(-1, 3),
        b_r=cat("b_r"), b_p=cat("b_p"),
        energies=np.stack([r["energies"] for r in records]),
        split=np.array([r["split"] for r in records], dtype=np.int8),
        parent_id=np.array([r["parent_id"] for r in records]),
        rxn_key=np.array([r["rxn_key"] for r in records]),
        formula=np.array([r["formula"] for r in records]),
        smiles_r=np.array([r["smiles_r"] for r in records]),
        smiles_p=np.array([r["smiles_p"] for r in records]),
    )


class T1xCache:
    """Read-only view of the CSR cache.  No pickle is used."""

    def __init__(self, path: Path):
        data = np.load(path, allow_pickle=False)
        self.arrays = {key: data[key] for key in data.files}
        self.split = self.arrays["split"]
        self.parent_id = self.arrays["parent_id"]

    def __len__(self) -> int:
        return len(self.split)

    def reaction(self, i: int) -> dict:
        a = self.arrays
        s, e = int(a["atom_off"][i]), int(a["atom_off"][i + 1])
        ps, pe = int(a["pair_off"][i]), int(a["pair_off"][i + 1])
        n = e - s
        return {
            "z": a["z"][s:e].astype(np.int64),
            "x_r": a["x_r"][s:e], "x_ts": a["x_ts"][s:e], "x_p": a["x_p"][s:e],
            "b_r": a["b_r"][ps:pe].reshape(n, n).astype(np.float32),
            "b_p": a["b_p"][ps:pe].reshape(n, n).astype(np.float32),
            "parent_id": str(a["parent_id"][i]),
            "index": i,
        }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:  # 分块读取，避免把 6.6 GB 读进内存
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_cache(h5_path: Path, out_path: Path, limit: int | None = None) -> dict:
    records: list[dict] = []
    dropped: Counter[str] = Counter()
    seen = Counter()
    for split in SPLITS:
        for raw in iter_split(h5_path, split, limit):
            seen[split] += 1
            try:
                records.append(process_reaction(raw))
            except DropReaction as exc:
                dropped[f"{split}:{exc}"] += 1
    by_split = {s: [r for r in records if r["split"] == v] for s, v in SPLITS.items()}
    for key in ("formula", "parent_id"):
        sets = {s: {r[key] for r in rs} for s, rs in by_split.items()}
        for a in SPLITS:
            for b in SPLITS:
                if a < b and sets[a] & sets[b]:
                    raise RuntimeError(f"leakage: {key} shared by {a} and {b}")
    write_cache(records, out_path)
    parents = {s: Counter(r["parent_id"] for r in rs) for s, rs in by_split.items()}
    manifest = {
        "schema": "xtbflow-m0-t1x-cache/v1",
        "h5_sha256": _sha256_file(Path(h5_path)) if limit is None else "skipped_for_limit",
        "cache_sha256": hashlib.sha256(Path(out_path).read_bytes()).hexdigest(),
        "seen": dict(seen),
        "kept": {s: len(rs) for s, rs in by_split.items()},
        "dropped": dict(sorted(dropped.items())),
        "parents": {s: len(c) for s, c in parents.items()},
        "reactions_per_parent_max": {s: max(c.values()) if c else 0 for s, c in parents.items()},
        "max_atoms": int(max(len(r["z"]) for r in records)),
        "order_only_reactions": int(sum(r["order_only"] > 0 for r in records)),
        "formal_charge_reactions": int(sum(r["any_formal_charge"] for r in records)),
    }
    return manifest
```

注意：`build_cache` 对 6.6 GB 文件分块求 SHA-256，约需几分钟，属正常。

### 4.5 `scripts/m0_prepare_t1x.py`

```python
import argparse, json
from pathlib import Path
from xtbflow.m0.t1x_data import build_cache

p = argparse.ArgumentParser()
p.add_argument("--h5", type=Path, required=True)
p.add_argument("--out", type=Path, required=True)          # 例如 $XTBFLOW_DATA/t1x/t1x_m0_v1.npz
p.add_argument("--limit", type=int, default=None)          # 调试时用 --limit 50
a = p.parse_args()
manifest = build_cache(a.h5, a.out, a.limit)
a.out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
print(json.dumps(manifest, indent=2, sort_keys=True))
```

先跑 `--limit 50` 检查无异常，再跑全量。全量结果把 manifest 复制一份到 `docs/evidence/m0/t1x_cache_manifest.json` 提交。然后 `rsync` 缓存与 manifest 到 n2 的 `$XTBFLOW_DATA/t1x/`。

### 4.6 `src/xtbflow/m0/batching.py`

```python
"""Pad variable-size reactions into dense batches."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from .t1x_data import T1xCache

ELEMENT_INDEX = {1: 1, 6: 2, 7: 3, 8: 4}  # 0 = padding
N_ELEMENT_TYPES = 5


class T1xDataset(Dataset):
    def __init__(self, cache: T1xCache, split: int, limit: int | None = None):
        self.cache = cache
        idx = np.flatnonzero(cache.split == split)
        self.idx = idx if limit is None else idx[:limit]

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, k: int) -> dict:
        return self.cache.reaction(int(self.idx[k]))


def collate(items: list[dict]) -> dict[str, torch.Tensor]:
    bsz = len(items)
    n = max(len(it["z"]) for it in items)
    out = {
        "z": torch.zeros(bsz, n, dtype=torch.long),
        "atom_mask": torch.zeros(bsz, n, dtype=torch.bool),
        "x_r": torch.zeros(bsz, n, 3),
        "x_ts": torch.zeros(bsz, n, 3),
        "x_p": torch.zeros(bsz, n, 3),
        "b_r": torch.zeros(bsz, n, n),
        "b_p": torch.zeros(bsz, n, n),
        "index": torch.tensor([it["index"] for it in items], dtype=torch.long),
    }
    for k, it in enumerate(items):
        m = len(it["z"])
        out["z"][k, :m] = torch.tensor([ELEMENT_INDEX[int(v)] for v in it["z"]])
        out["atom_mask"][k, :m] = True
        for key in ("x_r", "x_ts", "x_p"):
            out[key][k, :m] = torch.from_numpy(np.asarray(it[key], dtype=np.float32))
        for key in ("b_r", "b_p"):
            out[key][k, :m, :m] = torch.from_numpy(np.asarray(it[key], dtype=np.float32))
    return out
```

### 4.7 测试 `tests/test_m0_data.py`

不依赖 HDF5，只用 RDKit 生成的小分子。需要 RDKit 的测试用 `pytest.importorskip("rdkit")` 开头，这样 CI 没装 RDKit 时会跳过而不是失败。必须包含：

1. `kabsch_align`：对随机坐标施加随机真旋转和平移后，对齐回去的 RMSD < 1e-6；对镜像坐标，对齐后的 RMSD 必须明显大于 0（证明没有允许镜像）。
2. `be_matrix`：用 `Chem.AddHs(Chem.MolFromSmiles("CC=O"))` 嵌入三维坐标后跑 `perceive_mol`，结果矩阵对称、对角元为偶数、整矩阵之和等于价电子总数 18（C、C、O 加 4 个 H：4+4+6+4）。
3. `connectivity_change`：手工构造两个 4×4 矩阵，检查返回的对子。
4. CSR 往返：手工构造 2 条原子数不同的记录，`write_cache` 后用 `T1xCache` 读回，逐字段相等。
5. `collate`：两个原子数不同的样本，检查填充位置为 0、掩码正确。

### 4.8 验收与停止条件

| 检查项 | 通过标准 | 不通过时 |
| --- | --- | --- |
| 保留比例 | 三个切分合计保留 ≥ 85% | 停下来，把 `dropped` 原因分布发给负责人 |
| 测试集母体数 | ≥ 40 | 停下来问，不要自行改切分 |
| 泄漏 | 分子式、母体在切分间无交集 | 代码会直接报错；停下来问 |
| 最大原子数 | 预期约 23 | 若 > 30，检查是否读错了数据集 |
| 单元测试 | `pytest -q tests/test_m0_data.py` 全部通过 | 修复后再提 PR |

PR 中贴出完整 manifest，并用一句话报告 `order_only_reactions` 和 `formal_charge_reactions` 的数量。

## 5. 任务 2：模型架构与核心代码（第 3–5 天）

两臂共用同一个网络类 `ReactionFlowNet`，只靠“角色”区分：B 臂用一个 `joint` 角色，A 臂用一个 `event` 角色加一个 `geometry` 角色。这样两臂除了第 5.6 节的差异表之外完全相同，结果差异才能归因于“联合还是串联”。

```text
A 臂（串联）：反应物输入 → 事件网络 g（50 步积分 b）→ 解码（取整并修复）→ 几何网络 h（以整数 b_P 为条件，50 步积分 x）→ 候选 (b_P, X_TS)
B 臂（联合）：反应物输入 → 联合网络 f（50 步，每步同时更新 b_t 与 x_t，共用时间 t）→ 最后解码 → 候选 (b_P, X_TS)
```

A 臂的几何网络只看到解码后的整数事件；B 臂在每一步都让事件和几何互相参考，这正是 M0 要检验的差别。

**分支：** `m0/task2-model`。

### 5.1 决定效果的五个设计点（必须照做）

1. **起点必须带噪声。** 单端任务里同一个反应物有多条反应通道。起点 b\_0 = b\_R + σ\_b·ε\_b，X\_0 = X\_R + σ\_x·ε\_x。没有噪声，模型只能输出各通道的平均，这正是现有原型的问题。
2. **时间 t 每个样本独立均匀抽取**，范围 \[0, 1\]。联合角色里事件与几何共用同一个 t。
3. **事件用完整 N×N 对称矩阵**，不用打包上三角。总电子数等于矩阵全部元素之和（对角元算一次，非对角元算两次），所以守恒约束就是“速度矩阵所有有效元素之和为 0”，投影有闭式解（5.2 的 `project_conserved`），不需要 `pinv`。
4. **骨干要深**：6 层标量—向量消息传递，与 `configs/models/xtbflow_v0.1.yaml` 声明的规格一致（128 个标量通道、32 个向量通道）。全连接图，不做邻居截断（分子最多约 23 个原子）。
5. **几何速度必须等变且去质心。** 任何数值不安全的距离计算都改用 `numerics.pair_geometry`。

### 5.2 `src/xtbflow/m0/flow.py`

```python
"""Flow-matching states, noise, interpolation and losses for M0."""
from __future__ import annotations

import math

import torch
from torch import Tensor

from .numerics import remove_masked_com

ROLES = ("joint", "event", "geometry")


def active_pairs(atom_mask: Tensor) -> Tensor:
    """[B,N] -> [B,N,N] bool, diagonal included."""
    return atom_mask[:, :, None] & atom_mask[:, None, :]


def project_conserved(v: Tensor, atom_mask: Tensor) -> Tensor:
    """Orthogonal projection onto {sum_ij v_ij = 0} over the active block.

    Electron count of a BE matrix b is sum_ij b_ij, so b + dt*v keeps it
    exactly.  Symmetry is preserved; padded entries stay 0.
    """
    m = active_pairs(atom_mask).to(v.dtype)
    v = v * m
    count = m.sum(dim=(1, 2), keepdim=True).clamp_min(1.0)
    return (v - v.sum(dim=(1, 2), keepdim=True) / count) * m


def symmetric_noise(atom_mask: Tensor, generator: torch.Generator | None = None) -> Tensor:
    b, n = atom_mask.shape
    g = torch.randn(b, n, n, generator=generator, device=atom_mask.device)
    eps = (g + g.transpose(1, 2)) / math.sqrt(2.0)
    eye = torch.eye(n, dtype=torch.bool, device=atom_mask.device)[None]
    eps = torch.where(eye, g, eps)  # 对角元方差也为 1
    return project_conserved(eps, atom_mask)


def geometry_noise(atom_mask: Tensor, generator: torch.Generator | None = None) -> Tensor:
    b, n = atom_mask.shape
    eps = torch.randn(b, n, 3, generator=generator, device=atom_mask.device)
    return remove_masked_com(eps, atom_mask)


def interpolate(a0: Tensor, a1: Tensor, t: Tensor) -> Tensor:
    tt = t.view((-1,) + (1,) * (a0.ndim - 1))
    return (1.0 - tt) * a0 + tt * a1


def training_inputs(role: str, batch: dict, sigma_b: float, sigma_x: float,
                    generator: torch.Generator | None = None) -> tuple[dict, dict]:
    """Model inputs and velocity targets for one training step.

    joint:    b_cur = b_t, x_cur = x_t, shared t          -> b1-b0 and x1-x0
    event:    b_cur = b_t, x_cur = x_r (no geometry state) -> b1-b0
    geometry: b_cur = b_p (clean label), x_cur = x_t       -> x1-x0
    """
    mask = batch["atom_mask"]
    t = torch.rand(mask.shape[0], generator=generator, device=mask.device)
    b0 = batch["b_r"] + sigma_b * symmetric_noise(mask, generator)
    x0 = batch["x_r"] + sigma_x * geometry_noise(mask, generator)
    b1, x1 = batch["b_p"], batch["x_ts"]
    inputs = {"z": batch["z"], "atom_mask": mask, "x_r": batch["x_r"], "b_r": batch["b_r"], "t": t}
    if role == "joint":
        inputs.update(b_cur=interpolate(b0, b1, t), x_cur=interpolate(x0, x1, t))
        targets = {"b_vel": b1 - b0, "x_vel": x1 - x0}
    elif role == "event":
        inputs.update(b_cur=interpolate(b0, b1, t), x_cur=batch["x_r"])
        targets = {"b_vel": b1 - b0}
    elif role == "geometry":
        inputs.update(b_cur=b1, x_cur=interpolate(x0, x1, t))
        targets = {"x_vel": x1 - x0}
    else:
        raise ValueError(f"unknown role {role!r}")
    return inputs, targets


def event_loss(pred: Tensor, target: Tensor, atom_mask: Tensor) -> Tensor:
    m = active_pairs(atom_mask).to(pred.dtype)
    return ((pred - target).square() * m).sum() / m.sum().clamp_min(1.0)


def geometry_loss(pred: Tensor, target: Tensor, atom_mask: Tensor) -> Tensor:
    m = atom_mask.to(pred.dtype)[..., None]
    return ((pred - target).square() * m).sum() / (3.0 * m.sum()).clamp_min(1.0)


def flow_loss(out: dict, targets: dict, atom_mask: Tensor, geometry_weight: float = 1.0) -> dict:
    losses: dict[str, Tensor] = {}
    total = 0.0
    if "b_vel" in targets:
        losses["event"] = event_loss(out["b_vel"], targets["b_vel"], atom_mask)
        total = total + losses["event"]
    if "x_vel" in targets:
        losses["geometry"] = geometry_loss(out["x_vel"], targets["x_vel"], atom_mask)
        total = total + geometry_weight * losses["geometry"]
    losses["total"] = total
    return losses
```

### 5.3 `src/xtbflow/m0/backbone.py`

输入的边特征都对 (i, j) 对称：当前距离的径向基、反应物距离的径向基、当前 b\_ij、反应物 b\_ij、两者之差、反应物中是否成键。节点初始向量是“当前坐标减反应物坐标”的位移，乘以可学习权重。每层先做消息，再做更新，标量和向量通道宽度可以不同。

```python
"""Equivariant scalar-vector trunk (PaiNN-style) with symmetric pair features."""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .batching import N_ELEMENT_TYPES
from .numerics import pair_geometry


def gaussian_rbf(d: Tensor, n: int, r_max: float) -> Tensor:
    centers = torch.linspace(0.0, r_max, n, device=d.device, dtype=d.dtype)
    width = r_max / (n - 1)
    return torch.exp(-(((d[..., None] - centers) / width) ** 2))


def cosine_envelope(d: Tensor, r_cut: float) -> Tensor:
    env = 0.5 * (torch.cos(math.pi * d.clamp(max=r_cut) / r_cut) + 1.0)
    return env * (d < r_cut).to(d.dtype)


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int, n_freq: int = 16):
        super().__init__()
        self.register_buffer("freq", torch.arange(1, n_freq + 1, dtype=torch.float32) * math.pi)
        self.mlp = nn.Sequential(nn.Linear(2 * n_freq, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t: Tensor) -> Tensor:  # t: [B] -> [B,dim]
        arg = t[:, None] * self.freq[None, :]
        return self.mlp(torch.cat([torch.sin(arg), torch.cos(arg)], dim=-1))


class PaiNNLayer(nn.Module):
    def __init__(self, fs: int, fv: int, fe: int):
        super().__init__()
        self.fs, self.fv = fs, fv
        self.norm = nn.LayerNorm(fs)
        self.time = nn.Linear(fs, fs)
        self.phi = nn.Sequential(nn.Linear(fs, fs), nn.SiLU(), nn.Linear(fs, fs + 2 * fv))
        self.filter = nn.Linear(fe, fs + 2 * fv)
        self.U = nn.Linear(fv, fv, bias=False)
        self.V = nn.Linear(fv, fv, bias=False)
        self.update = nn.Sequential(nn.Linear(fs + fv, fs), nn.SiLU(), nn.Linear(fs, fv + 2 * fs))
        self.inner_to_s = nn.Linear(fv, fs)

    def forward(self, s, v, edge, unit, env, pair_mask, atom_mask, t_emb):
        # s [B,N,fs]  v [B,N,3,fv]  edge [B,N,N,fe]  unit [B,N,N,3]  env [B,N,N]
        fs, fv = self.fs, self.fv
        h = self.norm(s) + self.time(t_emb)[:, None, :]
        w = self.filter(edge) * (env * pair_mask.to(env.dtype))[..., None]
        x = self.phi(h)[:, None, :, :] * w                    # 下标 [b, i, j]，j 是发送方
        scale = pair_mask.sum(dim=2).clamp_min(1).to(s.dtype).rsqrt()[..., None]  # [B,N,1]
        ds, dvv, dvr = torch.split(x, [fs, fv, fv], dim=-1)
        s = s + ds.sum(dim=2) * scale
        dv = (v[:, None, :, :, :] * dvv[:, :, :, None, :]).sum(dim=2) \
            + (unit[..., None] * dvr[:, :, :, None, :]).sum(dim=2)
        v = v + dv * scale[..., None]
        uv, vv = self.U(v), self.V(v)
        vv_norm = torch.sqrt(vv.square().sum(dim=2) + 1e-8)  # [B,N,fv]，在 0 处梯度安全
        a = self.update(torch.cat([s, vv_norm], dim=-1))
        a_vv, a_sv, a_ss = torch.split(a, [fv, fs, fs], dim=-1)
        s = s + a_ss + a_sv * self.inner_to_s((uv * vv).sum(dim=2))
        v = v + a_vv[:, :, None, :] * uv
        m = atom_mask.to(s.dtype)
        return s * m[..., None], v * m[..., None, None]


class ReactionTrunk(nn.Module):
    def __init__(self, scalar_dim: int = 128, vector_dim: int = 32, edge_dim: int = 64,
                 n_layers: int = 6, n_rbf: int = 32, n_rbf_reactant: int = 16, r_cut: float = 10.0):
        super().__init__()
        self.n_rbf, self.n_rbf_r, self.r_cut = n_rbf, n_rbf_reactant, r_cut
        self.embed = nn.Embedding(N_ELEMENT_TYPES, scalar_dim)
        self.node_b = nn.Linear(3, scalar_dim)
        self.time = TimeEmbedding(scalar_dim)
        self.disp = nn.Linear(1, vector_dim, bias=False)
        self.edge_mlp = nn.Sequential(
            nn.Linear(n_rbf + n_rbf_reactant + 4, edge_dim), nn.SiLU(), nn.Linear(edge_dim, edge_dim)
        )
        self.layers = nn.ModuleList(PaiNNLayer(scalar_dim, vector_dim, edge_dim) for _ in range(n_layers))

    def forward(self, z, atom_mask, x_cur, x_r, b_cur, b_r, t) -> dict:
        n = z.shape[1]
        _, d_cur, unit, pair_mask = pair_geometry(x_cur, atom_mask)
        _, d_r, _, _ = pair_geometry(x_r, atom_mask)
        eye = torch.eye(n, dtype=torch.bool, device=z.device)[None]
        off = (~eye).to(x_cur.dtype)
        db = b_cur - b_r
        edge_in = torch.cat([
            gaussian_rbf(d_cur, self.n_rbf, self.r_cut),
            gaussian_rbf(d_r, self.n_rbf_r, self.r_cut),
            (b_cur * off)[..., None] / 3.0,
            (b_r * off)[..., None] / 3.0,
            (db * off)[..., None] / 3.0,
            ((b_r > 0.5) & ~eye).to(x_cur.dtype)[..., None],
        ], dim=-1)
        edge = self.edge_mlp(edge_in) * pair_mask[..., None].to(x_cur.dtype)
        diag = lambda m: torch.diagonal(m, dim1=1, dim2=2)
        node_b = torch.stack([diag(b_cur), diag(b_r), diag(db)], dim=-1) / 4.0
        t_emb = self.time(t)
        mask = atom_mask.to(x_cur.dtype)
        s = (self.embed(z) + self.node_b(node_b) + t_emb[:, None, :]) * mask[..., None]
        disp = (x_cur - x_r) * mask[..., None]
        v = disp[..., None] * self.disp.weight.view(1, 1, 1, -1)  # [B,N,3,fv]
        env = cosine_envelope(d_cur, self.r_cut)
        for layer in self.layers:
            s, v = layer(s, v, edge, unit, env, pair_mask, atom_mask, t_emb)
        return {"s": s, "v": v, "edge": edge, "pair_mask": pair_mask}
```

### 5.4 `src/xtbflow/m0/model.py`

事件头对每个原子对输出一个标量，输入只用对称组合（s\_i + s\_j、s\_i ⊙ s\_j、边特征、向量内积），所以输出天然对称；再显式对称化一次，消除浮点误差。几何头用标量门控向量通道，再去质心。

```python
"""ReactionFlowNet: one class, three roles (joint / event / geometry)."""
from __future__ import annotations

import torch
from torch import Tensor, nn

from .backbone import ReactionTrunk
from .flow import ROLES, project_conserved
from .numerics import remove_masked_com


class EventHead(nn.Module):
    def __init__(self, fs: int, fv: int, fe: int):
        super().__init__()
        self.pair = nn.Sequential(
            nn.Linear(2 * fs + fe + fv, fs), nn.SiLU(), nn.Linear(fs, fs), nn.SiLU(), nn.Linear(fs, 1)
        )
        self.diag = nn.Sequential(nn.Linear(fs, fs), nn.SiLU(), nn.Linear(fs, 1))

    def forward(self, s, v, edge, pair_mask, atom_mask) -> Tensor:
        si, sj = s[:, :, None, :], s[:, None, :, :]
        vv = (v[:, :, None, :, :] * v[:, None, :, :, :]).sum(dim=3)  # <v_i, v_j> [B,N,N,fv]
        feat = torch.cat([si + sj, si * sj, edge, vv], dim=-1)
        off = self.pair(feat).squeeze(-1)
        off = 0.5 * (off + off.transpose(1, 2)) * pair_mask.to(off.dtype)
        dg = self.diag(s).squeeze(-1) * atom_mask.to(s.dtype)
        return off + torch.diag_embed(dg)


class GeometryHead(nn.Module):
    def __init__(self, fs: int, fv: int):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(fs, fs), nn.SiLU(), nn.Linear(fs, fv))
        self.out = nn.Linear(fv, 1, bias=False)

    def forward(self, s, v, atom_mask) -> Tensor:
        vel = self.out(v * self.gate(s)[:, :, None, :]).squeeze(-1)  # [B,N,3]
        return remove_masked_com(vel, atom_mask)


class ReactionFlowNet(nn.Module):
    def __init__(self, role: str, scalar_dim: int = 128, vector_dim: int = 32, edge_dim: int = 64,
                 n_layers: int = 6, n_rbf: int = 32, n_rbf_reactant: int = 16, r_cut: float = 10.0):
        super().__init__()
        if role not in ROLES:
            raise ValueError(f"role must be one of {ROLES}")
        self.role = role
        self.trunk = ReactionTrunk(scalar_dim, vector_dim, edge_dim, n_layers, n_rbf, n_rbf_reactant, r_cut)
        self.event_head = EventHead(scalar_dim, vector_dim, edge_dim) if role in ("joint", "event") else None
        self.geometry_head = GeometryHead(scalar_dim, vector_dim) if role in ("joint", "geometry") else None

    def forward(self, z, atom_mask, x_cur, x_r, b_cur, b_r, t) -> dict[str, Tensor]:
        if self.role == "event" and not torch.equal(x_cur, x_r):
            raise ValueError("event role must be called with x_cur = x_r")
        if self.role == "geometry" and not torch.equal(b_cur, torch.round(b_cur)):
            raise ValueError("geometry role needs an integer b_cur (label or decoded event)")
        h = self.trunk(z, atom_mask, x_cur, x_r, b_cur, b_r, t)
        out: dict[str, Tensor] = {}
        if self.event_head is not None:
            raw = self.event_head(h["s"], h["v"], h["edge"], h["pair_mask"], atom_mask)
            out["b_vel"] = project_conserved(raw, atom_mask)
        if self.geometry_head is not None:
            out["x_vel"] = self.geometry_head(h["s"], h["v"], atom_mask)
        return out


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def match_cascade_width(model_cfg: dict, tol: float = 0.05) -> dict:
    """Pick the A-arm width so event+geometry params match the joint net within tol."""
    target = count_parameters(ReactionFlowNet("joint", **model_cfg))
    best = None
    for fs in range(32, model_cfg["scalar_dim"] + 1, 4):
        fv = max(8, fs // 4)
        cfg = dict(model_cfg, scalar_dim=fs, vector_dim=fv)
        total = count_parameters(ReactionFlowNet("event", **cfg)) + count_parameters(ReactionFlowNet("geometry", **cfg))
        rel = abs(total - target) / target
        if best is None or rel < best["rel_diff"]:
            best = {"scalar_dim": fs, "vector_dim": fv, "cascade_params": total,
                    "joint_params": target, "rel_diff": rel}
    if best["rel_diff"] > tol:
        raise RuntimeError(f"no width within {tol:.0%}: {best}")
    return best
```

粗算：B 臂约 0.94M 参数；A 臂两个网络在标量宽度约 92 时合计与之相差约 2%。以 `match_cascade_width` 的实际输出为准，并写进 `configs/m0/base.json` 的 `cascade_model` 字段。

### 5.5 测试 `tests/test_m0_model.py`

全部在 CPU 上跑，用 `scalar_dim=32, vector_dim=8, n_layers=2` 的小模型，随机输入 B = 2、N = 6，第二个样本只有 4 个真实原子。容差 float32 下 1e-4。

1. `project_conserved`：输出在有效块上求和为 0、对称、填充位为 0；投影两次与一次相同。
2. `symmetric_noise`：对称、有效块求和为 0、填充位为 0。
3. 旋转加平移等变：对 `x_cur` 与 `x_r` 同时施加同一个随机真旋转 R 和平移，`b_vel` 不变，`x_vel` 变为原来的乘 Rᵀ。
4. 原子置换等变：对真实原子做随机置换，同时置换 z、坐标、b 的行和列，输出相应置换。
5. 填充不变：在批里多加一个填充原子，真实原子上的输出不变。
6. `b_vel` 严格对称；`x_vel` 在真实原子上的平均为 0。
7. 角色保护：`event` 角色传入 `x_cur != x_r` 抛错；`geometry` 角色传入非整数 `b_cur` 抛错。
8. 两个真实原子坐标重合时，对参数和坐标的梯度都有限。
9. `match_cascade_width` 对默认配置返回 `rel_diff ≤ 0.05`。

### 5.6 两臂差异表（实验中唯一允许的差异）

| 项目 | A 臂 | B 臂 |
| --- | --- | --- |
| 网络 | `event` + `geometry` 两个实例，宽度来自 `match_cascade_width` | 一个 `joint` 实例，128/32 |
| 训练时几何输入 | 事件网络：`x_cur = x_r`；几何网络：带噪插值 | 带噪插值 |
| 训练时事件输入 | 事件网络：带噪插值；几何网络：真实 b\_P | 带噪插值 |
| 采样顺序 | 先事件后几何（第 6.4 节） | 同步（第 6.4 节） |
| 噪声尺度 σ\_b、σ\_x | 按第 6.3 节各自在验证集上选 | 同左 |

其余全部相同：数据、切分、批大小、步数、优化器、学习率表、EMA、种子、采样步数 NFE、候选数 N、解码器、评测代码。

## 6. 任务 3：训练与采样（第 5–9 天）

先用一个两通道玩具问题证明两臂都能采到两个通道，再上 Transition1x：噪声尺度在验证集上小规模挑选，然后每臂 3 个种子各训练 60k 步。所有 GPU 训练合计预计 10–20 GPU 小时，远低于第 0 节的 150 GPU 小时上限。

**分支：** `m0/task3-train`。

### 6.1 `configs/m0/base.json`

`cascade_model` 一栏填 `match_cascade_width(model)` 的实际输出，填好后不再改动。

```json
{
  "schema": "xtbflow-m0-config/v1",
  "data": {"cache": "${XTBFLOW_DATA}/t1x/t1x_m0_v1.npz"},
  "model": {"scalar_dim": 128, "vector_dim": 32, "edge_dim": 64, "n_layers": 6,
            "n_rbf": 32, "n_rbf_reactant": 16, "r_cut": 10.0},
  "cascade_model": {"scalar_dim": "FILL_FROM_match_cascade_width", "vector_dim": "FILL", "edge_dim": 64,
                    "n_layers": 6, "n_rbf": 32, "n_rbf_reactant": 16, "r_cut": 10.0},
  "flow": {"sigma_b": 0.5, "sigma_x": 0.5, "geometry_loss_weight": 1.0},
  "train": {"steps": 60000, "batch_size": 64, "lr": 5.0e-4, "min_lr": 1.0e-5, "weight_decay": 0.0,
            "warmup_steps": 1000, "grad_clip": 1.0, "ema_decay": 0.999,
            "first_eval": 20000, "eval_every": 10000, "num_workers": 4},
  "sample": {"nfe": 50, "n_samples_val": 8, "n_samples_test": 32, "queries_per_batch": 16},
  "eval": {"delta": 0.5, "bootstrap": 10000, "bootstrap_seed": 20261007},
  "seeds": [0, 1, 2]
}
```

### 6.2 `scripts/m0_train.py`（训练任一角色）

```python
"""Train one M0 role.
Example: python scripts/m0_train.py --config configs/m0/base.json --role joint --seed 0 \
         --out $XTBFLOW_RUNS/m0/full/joint_s0
"""
import argparse, copy, hashlib, json, math, os, random, subprocess, time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from xtbflow.m0.batching import T1xDataset, collate
from xtbflow.m0.flow import flow_loss, training_inputs
from xtbflow.m0.model import ReactionFlowNet, count_parameters
from xtbflow.m0.t1x_data import T1xCache


def lr_at(step: int, tc: dict) -> float:
    if step < tc["warmup_steps"]:
        return tc["lr"] * (step + 1) / tc["warmup_steps"]
    p = (step - tc["warmup_steps"]) / max(1, tc["steps"] - tc["warmup_steps"])
    return tc["min_lr"] + 0.5 * (tc["lr"] - tc["min_lr"]) * (1.0 + math.cos(math.pi * p))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--role", choices=("joint", "event", "geometry"), required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=None)       # 调参时用 20000
    ap.add_argument("--sigma-b", type=float, default=None)
    ap.add_argument("--sigma-x", type=float, default=None)
    ap.add_argument("--wide-cascade", action="store_true")   # 只用于可选的 A2x 对照
    ap.add_argument("--limit", type=int, default=None)       # 调试用
    a = ap.parse_args()

    cfg = json.loads(a.config.read_text())
    tc = dict(cfg["train"], steps=a.steps or cfg["train"]["steps"])
    sigma_b = cfg["flow"]["sigma_b"] if a.sigma_b is None else a.sigma_b
    sigma_x = cfg["flow"]["sigma_x"] if a.sigma_x is None else a.sigma_x
    model_cfg = cfg["model"] if (a.role == "joint" or a.wide_cascade) else cfg["cascade_model"]

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.manual_seed_all(a.seed)

    model = ReactionFlowNet(a.role, **model_cfg).to(device)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])

    cache_path = Path(os.path.expandvars(cfg["data"]["cache"]))
    ds = T1xDataset(T1xCache(cache_path), split=0, limit=a.limit)
    loader_gen = torch.Generator().manual_seed(a.seed)     # 三个角色同种子 => 同样的批次顺序
    loader = DataLoader(ds, batch_size=tc["batch_size"], shuffle=True, drop_last=True,
                        collate_fn=collate, num_workers=tc["num_workers"], generator=loader_gen,
                        persistent_workers=tc["num_workers"] > 0)
    noise_gen = torch.Generator(device=device).manual_seed(a.seed + 1)

    a.out.mkdir(parents=True, exist_ok=True)
    eval_steps = set(range(tc["first_eval"], tc["steps"] + 1, tc["eval_every"])) | {tc["steps"]}
    git_sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    meta = {"role": a.role, "seed": a.seed, "model_cfg": model_cfg, "sigma_b": sigma_b, "sigma_x": sigma_x,
            "train": tc, "params": count_parameters(model), "git_sha": git_sha,
            "config_sha256": hashlib.sha256(a.config.read_bytes()).hexdigest(),
            "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"}
    (a.out / "run_meta.json").write_text(json.dumps(meta, indent=2))

    log = (a.out / "train_log.jsonl").open("w")
    step, t0 = 0, time.time()
    while step < tc["steps"]:
        for batch in loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            for group in opt.param_groups:
                group["lr"] = lr_at(step, tc)
            inputs, targets = training_inputs(a.role, batch, sigma_b, sigma_x, noise_gen)
            losses = flow_loss(model(**inputs), targets, batch["atom_mask"], cfg["flow"]["geometry_loss_weight"])
            if not torch.isfinite(losses["total"]):
                (a.out / "NONFINITE.json").write_text(json.dumps(
                    {"step": step, "index": batch["index"].tolist()}))
                raise SystemExit(f"non-finite loss at step {step}")
            opt.zero_grad(set_to_none=True)
            losses["total"].backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
            opt.step()
            with torch.no_grad():
                for pe, p in zip(ema.parameters(), model.parameters()):
                    pe.lerp_(p, 1.0 - tc["ema_decay"])
            step += 1
            if step % 100 == 0:
                row = {"step": step, "lr": lr_at(step, tc), "grad_norm": float(gnorm),
                       "elapsed_s": time.time() - t0}
                row.update({k: float(v) for k, v in losses.items()})
                log.write(json.dumps(row) + "\n"); log.flush()
            if step in eval_steps:
                torch.save({"ema": ema.state_dict(), "model": model.state_dict(), "step": step, **meta},
                           a.out / f"ckpt_{step}.pt")
            if step >= tc["steps"]:
                break
    log.close()


if __name__ == "__main__":
    main()
```

要点：评测一律用 EMA 权重；三个角色用同一个 `--seed` 时，数据批次顺序相同，样本暴露量相同。

### 6.3 噪声尺度挑选与正式运行矩阵

噪声尺度会显著影响多通道覆盖，因此两臂各自在验证集上挑选，预算相同。挑选用种子 0、20k 步、验证集每个反应采 8 个候选，以第 1.3 节的主指标为准。

| 阶段 | 运行 | 个数 |
| --- | --- | --- |
| 挑选 B | `joint`，σ\_b ∈ {0.5, 1.0} × σ\_x ∈ {0.5, 1.0} | 4 |
| 挑选 A | `event`，σ\_b ∈ {0.5, 1.0}；`geometry`，σ\_x ∈ {0.5, 1.0}；两两组合成 4 条管线评测 | 4 次训练 |
| 正式 B | `joint`，种子 0、1、2，60k 步，选定的 σ | 3 |
| 正式 A | `event` 与 `geometry`，种子 0、1、2，60k 步，选定的 σ | 6 |
| 可选 A2x | `event` 与 `geometry` 加 `--wide-cascade`，种子 0 | 2 |

**检查点选择：** 在步数 20k、30k、40k、50k、60k 中，选验证集主指标最高的一个（N = 8）。A 臂把同一步数的事件检查点与几何检查点配成一对来评测，不允许混搭不同步数。

**先测速：** 第一个正式运行跑 500 步后看 `train_log.jsonl` 里的 `elapsed_s`，推算 60k 步用时。若超过 12 GPU 小时，停下来问。

### 6.4 `src/xtbflow/m0/sampler.py`

两臂用同一个随机数发生器、同样的抽取顺序（先事件噪声，后几何噪声），因此同一种子下两臂的起点噪声完全一样，比较是配对的。

```python
"""Generation for both arms.  Product/TS fields are forbidden at sampling."""
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from xtbflow.models.conserved_be import pack_be
from xtbflow.models.event_decoder import DecodeError, decode_endpoint

from .flow import geometry_noise, symmetric_noise

REACTANT_KEYS = ("z", "atom_mask", "x_r", "b_r", "index")
FORBIDDEN = {"x_ts", "x_p", "b_p"}
INDEX_TO_SYMBOL = {1: "H", 2: "C", 3: "N", 4: "O"}


def reactant_view(batch: dict) -> dict:
    return {k: batch[k] for k in REACTANT_KEYS}


def _repeat(rv: dict, n: int) -> dict:
    if FORBIDDEN & set(rv):
        raise ValueError(f"forbidden inputs at sampling: {sorted(FORBIDDEN & set(rv))}")
    return {k: v.repeat_interleave(n, dim=0) for k, v in rv.items()}


def decode_be(b: Tensor, z_index: Tensor, n: int) -> tuple[np.ndarray | None, np.ndarray]:
    """One padded sample -> (repaired integer BE [n,n] or None, fallback integer BE [n,n])."""
    bb = b[:n, :n].detach().double().cpu()
    bb = 0.5 * (bb + bb.T)
    fallback = np.clip(np.rint(bb.numpy()), 0, 3)
    diag = np.clip(2 * np.rint(np.diag(bb.numpy()) / 2), 0, 8)
    np.fill_diagonal(fallback, diag)
    symbols = [INDEX_TO_SYMBOL[int(v)] for v in z_index[:n]]
    try:
        decoded = decode_endpoint(pack_be(bb), symbols, charge=0, multiplicity=1)
    except (DecodeError, ValueError):
        return None, fallback.astype(np.int64)
    mat = np.array(decoded.state.be, dtype=np.int64)
    return mat, mat


def decode_batch(b: Tensor, rv: dict) -> tuple[Tensor, Tensor]:
    """[S,N,N] -> integer BE tensor (fallback where decoding failed) and ok flags [S]."""
    out = torch.zeros_like(b)
    ok = torch.zeros(b.shape[0], dtype=torch.bool, device=b.device)
    n_atoms = rv["atom_mask"].sum(dim=1).tolist()
    for k, n in enumerate(n_atoms):
        mat, fallback = decode_be(b[k], rv["z"][k], int(n))
        use = mat if mat is not None else fallback
        out[k, :n, :n] = torch.from_numpy(use).to(b)
        ok[k] = mat is not None
    return out, ok


@torch.no_grad()
def sample_joint(net, rv: dict, n_samples: int, nfe: int, sigma_b: float, sigma_x: float,
                 generator: torch.Generator) -> dict:
    rv = _repeat(rv, n_samples)
    mask = rv["atom_mask"]
    b = rv["b_r"] + sigma_b * symmetric_noise(mask, generator)
    x = rv["x_r"] + sigma_x * geometry_noise(mask, generator)
    dt = 1.0 / nfe
    for k in range(nfe):
        t = torch.full((mask.shape[0],), k * dt, device=mask.device)
        out = net(rv["z"], mask, x, rv["x_r"], b, rv["b_r"], t)
        b = b + dt * out["b_vel"]
        x = x + dt * out["x_vel"]
    return {"b_raw": b, "x": x, "rv": rv}


@torch.no_grad()
def sample_cascade(event_net, geometry_net, rv: dict, n_samples: int, nfe: int,
                   sigma_b: float, sigma_x: float, generator: torch.Generator) -> dict:
    rv = _repeat(rv, n_samples)
    mask = rv["atom_mask"]
    b = rv["b_r"] + sigma_b * symmetric_noise(mask, generator)
    dt = 1.0 / nfe
    for k in range(nfe):
        t = torch.full((mask.shape[0],), k * dt, device=mask.device)
        b = b + dt * event_net(rv["z"], mask, rv["x_r"], rv["x_r"], b, rv["b_r"], t)["b_vel"]
    b_int, _ = decode_batch(b, rv)                  # 第二步只看解码后的整数事件
    x = rv["x_r"] + sigma_x * geometry_noise(mask, generator)
    for k in range(nfe):
        t = torch.full((mask.shape[0],), k * dt, device=mask.device)
        x = x + dt * geometry_net(rv["z"], mask, x, rv["x_r"], b_int, rv["b_r"], t)["x_vel"]
    return {"b_raw": b, "x": x, "rv": rv}
```

`scripts/m0_sample.py` 的职责：读取选定的 EMA 检查点；按 `queries_per_batch` 把验证集或测试集的反应分批；每批调用 `reactant_view` 后再调用采样函数；随机数发生器为 `torch.Generator(device).manual_seed(10_000 + seed)`。输出 `candidates.npz`，字段为 `query_index`、`sample_id`、`n_atoms`、`atom_off`、`x`（所有候选坐标按 CSR 拼接）、`b_raw`（float32，按 n² 拼接）。解码与有效性判断放在评测里做，采样脚本不做。

### 6.5 两通道玩具检查 `scripts/m0_toy_multimodal.py`（上 GPU 之前必须通过）

体系是两个 H₂ 分子（4 个氢），有两种等概率的交换反应，各有自己的 TS。它在 CPU 上几分钟跑完，用来抓住“模式塌缩”和“事件与几何错配”两类致命 bug。

```python
"""Two-channel toy: both arms must cover both channels with matching geometry."""
import numpy as np
import torch

from xtbflow.m0.flow import flow_loss, training_inputs
from xtbflow.m0.model import ReactionFlowNet
from xtbflow.m0.sampler import decode_be, sample_cascade, sample_joint
from xtbflow.m0.t1x_data import kabsch_align

X_R = np.array([[0, 0, 0], [0.74, 0, 0], [0, 2.0, 0], [0.74, 2.0, 0]], dtype=float)
TS = {1: np.array([[0, 0, 0], [1.0, 0, 0], [0, 1.0, 0], [1.0, 1.0, 0]], dtype=float),   # 形成 0-2、1-3
      2: np.array([[0, 0, 0], [1.0, 0, 0], [1.0, 1.0, 0], [0, 1.0, 0]], dtype=float)}   # 形成 0-3、1-2


def be(pairs):
    b = np.zeros((4, 4))
    for i, j in pairs:
        b[i, j] = b[j, i] = 1
    return b


B_R = be([(0, 1), (2, 3)])
B_P = {1: be([(0, 2), (1, 3)]), 2: be([(0, 3), (1, 2)])}
XR_C = X_R - X_R.mean(0)
TS_AL = {c: kabsch_align(TS[c], XR_C) for c in TS}
SMALL = dict(scalar_dim=32, vector_dim=8, edge_dim=16, n_layers=2)


def batch_of(channels):
    k = len(channels)
    t = lambda a: torch.tensor(np.stack(a), dtype=torch.float32)
    return {"z": torch.ones(k, 4, dtype=torch.long), "atom_mask": torch.ones(k, 4, dtype=torch.bool),
            "x_r": t([XR_C] * k), "b_r": t([B_R] * k),
            "x_ts": t([TS_AL[c] for c in channels]), "b_p": t([B_P[c] for c in channels]),
            "index": torch.arange(k)}


def train(role, steps=4000, sigma=0.5, seed=0):
    torch.manual_seed(seed)
    net = ReactionFlowNet(role, **SMALL)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    gen = torch.Generator().manual_seed(seed + 1)
    for _ in range(steps):
        channels = torch.randint(1, 3, (64,), generator=gen).tolist()
        batch = batch_of(channels)
        inputs, targets = training_inputs(role, batch, sigma, sigma, gen)
        loss = flow_loss(net(**inputs), targets, batch["atom_mask"])["total"]
        opt.zero_grad(); loss.backward(); opt.step()
    return net.eval()


def rmsd(a, b):
    return float(np.sqrt(((kabsch_align(a, b) - b) ** 2).sum(1).mean()))


def judge(result, name):
    x = result["x"].numpy()
    counts, own_ok, own_rmsd = {1: 0, 2: 0, "other": 0}, 0, []
    for k in range(x.shape[0]):
        mat, _ = decode_be(result["b_raw"][k], result["rv"]["z"][k], 4)
        offd = None if mat is None else (mat - np.diag(np.diag(mat)))
        ch = next((c for c in (1, 2) if offd is not None and np.array_equal(offd, B_P[c])), "other")
        counts[ch] += 1
        if ch != "other":
            r_own, r_other = rmsd(x[k], TS_AL[ch]), rmsd(x[k], TS_AL[3 - ch])
            own_rmsd.append(r_own)
            own_ok += int(r_own < r_other)
    s = x.shape[0]
    matched = counts[1] + counts[2]
    report = {"arm": name, "share_1": counts[1] / s, "share_2": counts[2] / s, "valid": matched / s,
              "median_rmsd_own": float(np.median(own_rmsd)) if own_rmsd else None,
              "closer_to_own": own_ok / max(1, matched)}
    print(report)
    assert report["valid"] >= 0.8, report
    assert 0.25 <= report["share_1"] <= 0.75 and 0.25 <= report["share_2"] <= 0.75, report
    assert report["median_rmsd_own"] <= 0.3 and report["closer_to_own"] >= 0.8, report


if __name__ == "__main__":
    rv = {k: v for k, v in batch_of([1]).items() if k not in ("x_ts", "b_p")}
    joint = train("joint")
    judge(sample_joint(joint, rv, 128, 50, 0.5, 0.5, torch.Generator().manual_seed(7)), "B_joint")
    event, geom = train("event"), train("geometry")
    judge(sample_cascade(event, geom, rv, 128, 50, 0.5, 0.5, torch.Generator().manual_seed(7)), "A_cascade")
    print("TOY PASS")
```

若某一臂不通过：先把 `steps` 提到 10000；仍不通过，按第 9 节排查，不要上 GPU。玩具通过后把打印结果贴进 PR。

### 6.6 验收

- `python scripts/m0_toy_multimodal.py` 输出 `TOY PASS`。
- `--limit 256 --steps 300` 在 CPU 上把三个角色各跑一遍，`train_log.jsonl` 中损失下降、无 `NONFINITE.json`。
- 挑选阶段与正式阶段全部运行完成，每个运行目录都有 `run_meta.json`、`train_log.jsonl` 和全部检查点。
- 把每个运行的最终训练损失与所选检查点步数汇总为 `docs/evidence/m0/runs_summary.json`。

## 7. 任务 4：评测、统计与闸门报告（第 9–10 天）

评测脚本对两臂完全相同；测试集只在“选择冻结”之后采样一次。判据由代码 `decide()` 给出，不由人或代理主观判断。

**分支：** `m0/task4-eval`。

### 7.1 测试集纪律

1. 挑选和检查点选择全部完成后，先提交 `docs/evidence/m0/frozen_selection.json`：两臂选定的 σ\_b、σ\_x、检查点步数、检查点文件 SHA-256、当前 git 提交号。
2. 这个文件合入（或至少推送到分支）之后，才允许对测试集运行 `m0_sample.py`。
3. 测试集结果出来后，不得再改任何超参、检查点或评测代码；发现 bug 必须停下来问负责人，并在报告中记录。

### 7.2 `src/xtbflow/m0/metrics.py`

事件匹配允许相差一个反应物自同构。做法：对每个参考反应，先枚举反应物的全部自同构 π（RDKit 子结构自匹配，上限 10,000 个），把参考的变化集合按 π 重新编号，建一个“编号后的集合 → π 列表”的字典。候选的变化集合在字典里查到，就算事件匹配；再用对应的 π 把候选坐标换回参考编号，取最小 RMSD。

```python
"""Event matching up to reactant automorphism, RMSD, per-query metrics, bootstrap, decision."""
from __future__ import annotations

from collections import defaultdict

import numpy as np

VALENCE_E = {1: 1, 6: 4, 7: 5, 8: 6}
DELTAS = tuple(round(0.1 * k, 1) for k in range(1, 11))  # 0.1 ... 1.0 Å


def change_set(b_r: np.ndarray, b_p: np.ndarray) -> frozenset:
    changed = np.triu((b_r > 0) != (b_p > 0), k=1)
    return frozenset((int(i), int(j)) for i, j in np.argwhere(changed))


def be_to_mol(z: np.ndarray, b: np.ndarray, sanitize: bool = True):
    from rdkit import Chem

    order = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE}
    mol = Chem.RWMol()
    n = len(z)
    for i in range(n):
        atom = Chem.Atom(int(z[i]))
        atom.SetFormalCharge(int(VALENCE_E[int(z[i])] - b[i].sum()))  # 行和含对角元
        atom.SetNoImplicit(True)
        mol.AddAtom(atom)
    for i in range(n):
        for j in range(i + 1, n):
            if b[i, j] > 0:
                mol.AddBond(i, j, order[int(b[i, j])])
    mol = mol.GetMol()
    if sanitize:
        Chem.SanitizeMol(mol)
    else:
        mol.UpdatePropertyCache(strict=False)
    return mol


def is_valid_product(z: np.ndarray, b_r: np.ndarray, b_dec: np.ndarray | None) -> bool:
    if b_dec is None or len(change_set(b_r, b_dec)) == 0:
        return False
    try:
        be_to_mol(z, b_dec, sanitize=True)
    except Exception:
        return False
    return True


def automorphisms(z: np.ndarray, b_r: np.ndarray, max_matches: int = 10000) -> tuple[np.ndarray, bool]:
    """perms[k][q] = image of atom q under automorphism k; second value = cap reached."""
    try:
        mol = be_to_mol(z, b_r, sanitize=True)
        matches = mol.GetSubstructMatches(mol, uniquify=False, useChirality=False, maxMatches=max_matches)
    except Exception:
        matches = ()
    if not matches:
        return np.arange(len(z))[None, :], False
    return np.asarray(matches, dtype=np.int64), len(matches) >= max_matches


def reference_index(ref_pairs: frozenset, perms: np.ndarray) -> dict:
    """{relabelled reference change set: [k, ...]} with relabelled = {(pi(i), pi(j))}."""
    index: dict = defaultdict(list)
    for k, pi in enumerate(perms):
        relabelled = frozenset(tuple(sorted((int(pi[i]), int(pi[j])))) for i, j in ref_pairs)
        index[relabelled].append(k)
    return index


def batched_kabsch_rmsd(mobile: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """mobile [K,N,3], ref [N,3] -> [K] RMSD after the best proper rotation."""
    m = mobile - mobile.mean(axis=1, keepdims=True)
    r = ref - ref.mean(axis=0)
    h = np.einsum("kni,nj->kij", m, r)
    u, s, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(u) * np.linalg.det(vt))
    trace = s[:, 0] + s[:, 1] + d * s[:, 2]
    msd = ((m ** 2).sum(axis=(1, 2)) + (r ** 2).sum() - 2.0 * trace) / ref.shape[0]
    return np.sqrt(np.clip(msd, 0.0, None))


def evaluate_query(z, b_r, x_r, b_p_ref, x_ts_ref, cand_b_dec: list, cand_x: np.ndarray,
                   perms: np.ndarray) -> dict:
    """One test reaction and its S candidates (cand_b_dec[k] is an int matrix or None)."""
    index = reference_index(change_set(b_r, b_p_ref), perms)
    d_r = np.linalg.norm(x_r[:, None] - x_r[None], axis=-1)
    valid, best_rmsd, consistency, events = [], [], [], set()
    for b_dec, x in zip(cand_b_dec, cand_x):
        ok = is_valid_product(z, b_r, b_dec)
        valid.append(ok)
        if not ok:
            continue
        pairs = change_set(b_r, b_dec)
        events.add(pairs)
        d_c = np.linalg.norm(x[:, None] - x[None], axis=-1)
        agree = [(d_c[i, j] < d_r[i, j]) if b_dec[i, j] > 0 else (d_c[i, j] > d_r[i, j]) for i, j in pairs]
        consistency.append(float(np.mean(agree)))
        ks = index.get(pairs)
        if ks:
            best_rmsd.append(float(batched_kabsch_rmsd(x[perms[ks]], x_ts_ref).min()))
    best = min(best_rmsd) if best_rmsd else float("inf")
    return {
        "valid_frac": float(np.mean(valid)),
        "hit_event": bool(best_rmsd),
        "hit": {str(d): bool(best <= d) for d in DELTAS},
        "best_rmsd": best if best_rmsd else None,
        "consistency": float(np.mean(consistency)) if consistency else None,
        "unique_valid_events": len(events),
    }


def parent_means(per_query: dict[int, float], parent_of: dict[int, str]) -> dict[str, float]:
    groups: dict[str, list[float]] = defaultdict(list)
    for q, value in per_query.items():
        groups[parent_of[q]].append(value)
    return {p: float(np.mean(v)) for p, v in groups.items()}


def paired_bootstrap(a: dict[str, float], b: dict[str, float], n_boot: int, seed: int) -> dict:
    if set(a) != set(b):
        raise ValueError("arms must cover the same parents")
    parents = sorted(a)
    va, vb = np.array([a[p] for p in parents]), np.array([b[p] for p in parents])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(parents), size=(n_boot, len(parents)))
    diffs = (vb[idx] - va[idx]).mean(axis=1)
    return {"M_A": float(va.mean()), "M_B": float(vb.mean()), "delta": float((vb - va).mean()),
            "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
            "n_parents": len(parents)}


def decide(primary: dict, seed_deltas: list[float], valid: dict) -> str:
    low, high = primary["ci95"]
    if low > 0 and sum(d > 0 for d in seed_deltas) >= 2 and valid["ci95"][0] > -0.02:
        return "GO"
    if high < 0.02 or primary["delta"] <= 0:
        return "NO-GO"
    return "INCONCLUSIVE"
```

### 7.3 `scripts/m0_evaluate.py` 与 `scripts/m0_gate_report.py` 的职责

`m0_evaluate.py --cache ... --candidates .../candidates.npz --split test --out .../eval.jsonl`：

1. 对每个查询反应，用 `decode_be`（第 6.4 节）解码每个候选的 `b_raw`，得到整数矩阵或 None。
2. 对每个查询反应调用一次 `automorphisms`，再调用 `evaluate_query`。
3. 每个反应输出一行 JSON：`index`、`parent_id`、`evaluate_query` 的全部字段、`automorphism_cap_hit`。

`m0_gate_report.py --a <A 臂 3 个 eval.jsonl> --b <B 臂 3 个 eval.jsonl> --config ...`：

1. 每臂把 3 个种子的 `hit["0.5"]` 在反应层面平均，再用 `parent_means` 聚合到母体。
2. 调用 `paired_bootstrap` 得到主指标；对有效率 `valid_frac` 做同样的配对 bootstrap。
3. 每个种子单独算一次 Δ（不做 bootstrap），得到 `seed_deltas`。
4. 调用 `decide()`。另外对 δ = 0.3 与 1.0 Å 重算一遍结论，只报告、不改结论。
5. 写 `docs/evidence/m0/gate_result.json` 和 `docs/m0/M0_GATE_REPORT.md`。脚本先检查 `frozen_selection.json` 的提交时间早于所有测试集 `candidates.npz` 的生成时间，否则拒绝出报告。

### 7.4 `gate_result.json` 必须包含的字段

| 字段 | 含义 |
| --- | --- |
| `decision` | GO、NO-GO 或 INCONCLUSIVE |
| `primary` | `paired_bootstrap` 的输出（M\_A、M\_B、delta、ci95、n\_parents） |
| `seed_deltas` | 3 个种子各自的 Δ |
| `validity` | 有效率的配对 bootstrap |
| `secondary` | 两臂的事件召回、命中反应的 RMSD 中位数、一致性、不同有效事件数，按种子平均 |
| `delta_curve` | δ 从 0.1 到 1.0 Å 时两臂的主指标 |
| `sensitivity` | δ = 0.3 与 1.0 Å 时 `decide()` 的结论 |
| `a2x` | 若跑了 A2x，给出种子 0 的主指标；否则为 null |
| `params` | 两臂的实际参数量与相对差 |
| `automorphism_cap_hits` | 自同构达到上限的反应数 |
| `provenance` | git 提交号、缓存 SHA-256、配置 SHA-256、全部检查点 SHA-256 |

### 7.5 测试 `tests/test_m0_metrics.py`

1. `batched_kabsch_rmsd`：随机坐标的旋转平移副本 RMSD < 1e-6；与 `kabsch_align` 算出的 RMSD 一致。
2. 自同构匹配：水分子（原子顺序 O、H、H），参考事件为 O–H1 异裂（产物 H⁺ 加 OH⁻，O 的对角元为 6，这样 RDKit 能消毒）；候选为 O–H2 异裂、坐标是参考 TS 交换两个 H 后的结果，必须判为事件匹配且 RMSD < 1e-6。
3. 有效性：`b_dec=None` 无效；没有任何键变化无效；一个正常的乙烯产物有效。
4. `paired_bootstrap`：两臂相同时 delta 为 0 且区间包含 0；B 臂每个母体都高 0.1 时 delta 为 0.1、区间宽度 < 1e-9。
5. `decide()`：分别构造满足 GO、NO-GO、INCONCLUSIVE 的输入，检查返回值。

### 7.6 验收

- 上述测试通过；`gate_result.json` 与 `M0_GATE_REPORT.md` 生成。
- 报告第一句写结论（GO、NO-GO 或 INCONCLUSIVE）加主指标数值与区间；正文不超过两页；附 δ 曲线图与每个种子的 Δ。

## 8. 两周时间表与每日验收

从 10 月 8 日开工，共 10 个工作日，Oct 21, 2026 给出闸门结论。只有结论为“不确定”时，才启用 10 月 22–27 日的延长。

| 阶段 | 日期（2026） |
| --- | --- |
| 任务 0 环境体检 | 10/8 |
| 任务 1 数据管线 | 10/8–10/12 |
| 任务 2 模型与测试 | 10/12–10/14 |
| 任务 3 玩具与调参 | 10/14–10/16 |
| 任务 3 正式训练 | 10/16–10/20 |
| 任务 4 评测与报告 | 10/20–10/21 |
| 去/留闸门结论 | 10/21 |
| 不确定时延长（可选） | 10/22–10/27 |

正式训练占时最长，因为每臂 3 个种子共 9 个运行；只要玩具检查和测速都通过，这一段可以并行跑完。

### 8.1 每日验收清单

- [ ] 10/8：任务 0 的 PR 已提交，`env_n2.json` 显示 GPU 可用
- [ ] 10/9：`t1x_layout.txt` 已提交，`--limit 50` 的缓存构建无异常
- [ ] 10/12：全量缓存与 manifest 已提交，保留比例 ≥ 85%，测试母体 ≥ 40
- [ ] 10/13：`test_m0_model.py` 的 9 条测试全部通过
- [ ] 10/14：`cascade_model` 宽度写入配置，参数差 ≤ 5%；玩具检查输出 `TOY PASS`
- [ ] 10/15：8 个挑选运行完成，两臂的 σ 已选定
- [ ] 10/16：9 个正式运行全部启动，测速结果写进 PR
- [ ] 10/20：检查点选定，`frozen_selection.json` 提交后才对测试集采样
- [ ] 10/21：`gate_result.json` 与 `M0_GATE_REPORT.md` 已提交，负责人确认结论

### 8.2 闸门报告的结构

`docs/m0/M0_GATE_REPORT.md` 按以下顺序写，正文不超过两页，字段定义见第 7.4 节：

1. 第一句：结论（GO、NO-GO 或 INCONCLUSIVE），加 M\_A、M\_B、Δ 和 95% 区间。
2. 表 1：主指标、有效率及其区间，3 个种子各自的 Δ。
3. 图 1：δ 从 0.1 到 1.0 Å 的两臂曲线。
4. 表 2：次要指标与参数量；若跑了 A2x，单列一行。
5. 与本方案不一致的地方（若有），逐条说明原因。
6. 下一步：GO 进入 M1；NO-GO 转向方向 B；INCONCLUSIVE 启动延长。

## 9. 故障排查与停止条件

按症状查表；表中“两臂同改”的处理必须同时作用于 A、B 两臂，并在 PR 里写明。表里没有的情况，或连续两次处理无效，按第 0 节停下来问负责人。

| 症状 | 最可能的原因 | 处理 |
| --- | --- | --- |
| 训练一开始损失就是 NaN | 某处没用 `pair_geometry`，自己写了 `norm()`；或学习率过高 | 全文搜索 `norm(` 与 `sqrt(`，改为 `pair_geometry`；仍有问题时两臂同改为 lr = 2e-4 |
| 玩具问题里所有候选都是同一个通道 | 采样或训练没加噪声；t 不是逐样本随机；σ 太小 | 检查 `training_inputs` 与 `sample_*` 是否都用了噪声；把玩具的 σ 改为 1.0 再试 |
| 玩具问题有效率很低 | 对角元约定错了（应为非键电子数，不是价电子数）；事件头不对称 | 打印 `DecodeError` 的原因；跑 `test_m0_model.py` 第 6 条 |
| 玩具问题事件对、几何却贴近另一个通道 | 联合臂：`b_cur` 与 `x_cur` 来自不同时间点；串联臂：几何网络拿到的是未解码的连续 b | 联合臂检查共用同一个 t；串联臂检查 `sample_cascade` 传的是 `b_int` |
| 旋转等变测试失败 | 把绝对坐标当成标量特征；向量通道上的 Linear 带了偏置；几何速度没去质心 | 对照第 5.3、5.4 节逐行检查 |
| 置换等变测试失败 | 事件头用了上三角打包或位置编码；批内跨原子的操作写错维度 | 事件头只能用对称组合，见第 5.4 节 |
| 两臂验证集主指标在 20k 步都接近 0 | TS 没对齐到反应物坐标系；单位不是 Å；几何损失没有下降 | 抽 5 个样本画出 R 与 TS 的 RMSD；检查 `train_log.jsonl`，几何损失从 1k 步到 20k 步应至少下降 3 倍 |
| GPU 显存不够 | 批大小 64 的成对张量过大 | 两臂同改：批大小 32、步数 120k，保持样本暴露量不变 |
| 训练太慢（推算单次超过 12 GPU 小时） | 数据加载成瓶颈或 GPU 太弱 | 先把 `num_workers` 调到 8；仍超时，停下来问 |
| 超过 2% 的测试反应触发自同构上限 | 分子里甲基很多 | 只在报告中记录数量，不改上限 |
| 数据保留比例低于 85% | 键级推断失败集中在某类结构 | 停下来，附 `dropped` 原因分布 |

**INCONCLUSIVE 时的延长程序（最多 4 天）：** 两臂都追加种子 3、4 的正式训练；测试集每个反应采样数从 32 提到 64；其余设置不变。重新运行 `m0_gate_report.py`，此时 `seed_deltas` 有 5 个值，规则改为“至少 3 个为正”。延长后仍不是 GO，按 NO-GO 处理。

**NO-GO 之后：** 不再为“联合”调参。把 M0 的代码、配置和结果完整保留，负结果写进报告；下一步转向评估报告中的方向 B（水相质子/酰基转移 TS 数据集加生成模型基准）。M0 的单端评测代码可以直接复用为那个基准的评测部分。

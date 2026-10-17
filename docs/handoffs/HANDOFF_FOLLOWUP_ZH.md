# xtbflow 交接：三项后续任务（强化枚举基线、并联机制线的最后检验、开放世界 DFT 认证）

2026-10-09 · 交给一个从未接触过本项目的 AI 执行 · 计算在 n2 上 · 完成后由 Claude Code（本文作者）验收

本文自成一体：读完就能开工，不需要读之前的对话。凡是写着"停下来"的地方，必须停下、写明原因并等待负责人，不要自行绕过。负责人已批准本文列出的全部算力（含 DFT），你可以连续推进耗时任务，但只限本文范围之内。

---

## 0. 一页摘要

| 任务 | 要回答的问题 | 主要算力 | 分支（基于） |
|---|---|---|---|
| **任务 1** 强化枚举基线（T1b） | 换成综述里的标准做法（YARP 式：断 ≤2、成 ≤2 的穷举 + 产物热力学筛选），生成模型还能胜过穷举吗？ | CPU ≤ 60 核时，无 GPU | `explore/enum-thermo`（基于 `explore/followup-base`） |
| **任务 2a** 反应坐标诊断 | 前两轮"鞍点感知"用的反应坐标 u，在真实 TS 上是不是负曲率的反应模？ | CPU < 2 核时 | `explore/smc-oracle`（基于 `explore/followup-base`） |
| **任务 2b** 用更好的物理打分重做中途筛选 | 把 GFN2-xTB 换成 AIMNet2 / g-xTB 后，"生成中途按物理筛选"能否胜过"生成完再排序"？ | GPU < 0.5 小时，CPU ≤ 10 核时 | 同 2a |
| **任务 3a** 开放世界 DFT 认证 | xTB 层面确认的 10 条目录外新通道，在 DFT（ωB97X/6-31G(d)）层面还成立吗？势垒比已知最佳通道高还是低？ | GPU ≤ 16 小时 | `review/open-world-o2`（基于 `review/open-world-o1`） |
| **任务 3b** AIMNet2 训练数据核查 | AIMNet2 / AIMNet2-rxn 的训练集是否包含 Transition1x？ | 无 | 同 3a |

**不做什么。**
- 不碰 V1b 的正式计算（任务 3a 只借用 V1b 的 DFT 协议代码，不改它）；
- 不改任何已有的证据、配置或脚本；
- 不合并任何 PR；
- 不接受任何第三方许可、不登录任何网站。

**建议顺序。**
1. 先把任务 3a 的 DFT 作业排起来，它最耗时：约 30 个串行 Slurm 作业，墙钟约 12–16 小时。
2. 等 GPU 的同时，在 CPU 上做 2a → 2b 的打分 → 任务 1。
3. 任务 3b 随时可做。

---

## 1. 项目背景（最少必要）

### 1.1 xtbflow 在做什么

目标：只给反应物，用一个流匹配生成模型同时提出"发生什么反应"（事件）和"过渡态（TS）长什么样"（3D 几何），再用便宜的量子化学方法和少量 DFT 去验证。

- **母体（parent）**：一个反应物分子，中性闭壳层，元素 C/H/N/O，≤23 个原子。
- **事件 / 通道**：键—电子矩阵（BE 矩阵）的变化 b_p − b_r。对角元是孤对电子数，非对角元是键级。规范事件 ID `channel_id` 由 `xtbflow.v1.data.canonical_event(b_r, b_p, perms)` 给出。
- **目录**：数据来自 Transition1x（T1x，ωB97X/6-31G(d)）。每个母体有若干已知通道和目录势垒（kcal/mol）。最佳通道 = 目录势垒最低者（`best_channel_ids`）。
- **生成臂**：B0 = 联合模型无引导（最强），A0/A1/A2 = 串联变体，B1 = 联合加学习型引导。
- **划分**：训练 386 个母体；开发集（dev）62 个母体、21 个分子式组；筛查集（screen）342 个母体、70 组。M0 的官方 val/test **永远不要碰**。

### 1.2 与本交接直接相关的已有结论（全部探索性，PR 均未合并）

| PR | 结论 |
|---|---|
| #113 T1 枚举基线 | 断 ≤3、成 ≤3（b3f3）穷举，每个母体约 3 万个事件，用只在真实反应上训练过的势垒头 S_N 排序。AUC(S_N) − AUC(B0) = −0.177 [−0.230, −0.123]，生成胜出。**局限**：S_N 没见过大量不合理的穷举产物，所以这是个偏弱的基线 → 任务 1 |
| #114 T2 物理打分方法 | 同母体排序 ρ：真实 TS 上 AIMNet2 0.924 / g-xTB 0.909 / GFN2 0.749；生成几何上 0.619 / 0.602 / 0.508 |
| #115 T3 开放世界第一阶段 | 40 个目录外候选中，17 个在 GFN2-xTB 层面严格认证为新通道；阳性对照 17/20。12 个能与基准比较，只有 1 个 xTB 势垒更低 → 任务 3a |
| #112 鞍点感知推动 pg-1 | 沿 (I − 2uuᵀ)F 推几何能改写事件，但几乎不偏向低势垒 |
| #116 中途筛选 smc-1 | 在 s=0.7 用 GFN2 打分剪掉一半、克隆另一半，相对"生成完再排序"：ΔH = −0.016 [−0.032, +0.000]，判为 `NO_GAIN_OVER_POSTHOC`。弛豫打分器有缺陷：沿 u 反射在 u 为正曲率方向时一路上爬，真实 TS 上弛豫后只有 17% 是负曲率 → 任务 2a、2b |
| #117 smc-1 补充 | 终点改用 GFN2 原始能量排序后，原始能量剪枝仍无增益；剪枝丢掉最佳通道的损失是主因 |

**负责人已确认：** CPU 计算可以直接在 n2 登录 shell 上跑（不走 Slurm），GPU 必须走 Slurm。

---

## 2. 环境与"不影响其他任务"（硬性规则）

### 2.1 机器与仓库

- **仓库**：`git@github.com:askusk521-cyber/xtbflow.git`，默认分支 `main`。
  - 你只开 PR，不合并、不开自动合并、不直推 main。
  - 开工前读仓库根目录的 `AGENTS.md`。
- **n2**：用户 `lhshen`，192 核，4×RTX PRO 6000（96 GiB），Slurm 21.08，Ubuntu 22.04。单节点：登录 shell 就在计算节点上，能看到 GPU。
- **整合基线分支 `explore/followup-base`**（代码基线 `8571273`，本文档在其后的一个 docs 提交里；以分支最新提交为准）：
  - 由 `explore/smc-selection`（#116）合并 `review/enum-baseline`（#113）和 `review/oracle-benchmark`（#114）得到；
  - n2 上已测：pytest 330 通过、reusable unittest 56 通过、bootstrap 511 个文件通过；
  - 本交接文档也在这个分支上（`docs/handoffs/HANDOFF_FOLLOWUP_ZH.md`）。

### 2.2 GPU 与 CPU

- **GPU：必须经 Slurm**（`sbatch`/`srun`）。
  - 分区 `main`，单卡 `--gres=gpu:pro6000:1`；
  - 不要手动设 `CUDA_VISIBLE_DEVICES`，Slurm 会分配；
  - 单个作业墙钟 ≤ 120 分钟，作业名以 `fu-` 开头。
- **同一时刻全项目只允许 1 个 Slurm 作业。**
  - 提交前先 `squeue -u lhshen`（`$USER` 在 Slurm 里不可用）；
  - 有作业在跑或排队（包括别人的），就用 `--dependency=afterany:<jobid>` 排在后面；
  - **绝不 `scancel` 不是你提交的作业。**
- **CPU：直接在登录 shell 上跑**，用 `nohup ... &` 挂后台，日志写进运行目录。
  - 每个 Python 命令都设 `CUDA_VISIBLE_DEVICES=`，防止误用 GPU；
  - 加 `nice -n 10`，进程池 worker ≤ 48；
  - 启动重任务前看 `uptime`：1 分钟负载 > 120 就等；
  - 同一时刻你自己最多一个重 CPU 任务。
- **`sbatch --wrap` 用的是 `/bin/sh`**，没有 `source`/`conda activate`，直接用解释器绝对路径：

```bash
sbatch --parsable --partition=main --gres=gpu:pro6000:1 --cpus-per-task=8 --mem=64G --time=02:00:00 \
  --job-name=fu-dft --output=$OUT/dft-%j.log \
  --wrap "cd $S; export PYTHONPATH=src:vendor/mechai_reusable OMP_NUM_THREADS=8; /home/lhshen/miniconda3/envs/xtbflow-qc/bin/python scripts/<脚本>.py ..."
```

- Slurm 记账没开，`sacct` 用不了。完成情况靠日志和 `scontrol show job <id>`（作业结束后很快消失，结束时立即存下来）。

### 2.3 不能动的东西（只读）

- **其他任务的目录**：
  - `/home/lhshen/xtbflow-runs/review-align-20261009/`、`explore-smc-20261009/`、`explore-geoinfo-20261008/`、`acceptance-*`、`v1a-*`、`v1b-*`；
  - 可以读文件、调用其中的环境和二进制，但不得写入、删除、改名。
- `/home/lhshen/xtbflow/` 下的主 checkout 和所有 worktree：不切分支、不重置、不 stash、不改文件。
- **conda 环境**：`xtbflow`、`xtbflow-qc`、`xtbflow-cp2k` 以及 `review-align-20261009/envs/*`，不往里装任何包。本任务不需要新包；真需要时新建环境并停下来说明。
- **仓库里已有的文件**：只新增，不修改，`bootstrap_inventory.json` 除外。尤其不改：
  - `src/xtbflow/v1/qc_protocol.py`、`scripts/v1b_d1.py`、`scripts/explore_smc.py`、`scripts/review_*.py`、`src/xtbflow/v1/enumeration.py`；
  - 需要其中的函数时 import（脚本可 `sys.path.insert(0, 'scripts')` 后 import）。

### 2.4 你的目录与代码同步

- 运行根目录：`R=/home/lhshen/xtbflow-runs/followup-20261009`，只在这里写东西。
- **开发**：在 `$R/dev-<任务>` 新建克隆。

```bash
mkdir -p $R && cd $R
git clone -q --no-checkout /home/lhshen/xtbflow-runs/explore-smc-20261009/source-2116d43 dev-enum
cd dev-enum && git remote set-url origin git@github.com:askusk521-cyber/xtbflow.git
git fetch -q origin explore/followup-base && git checkout -q -b explore/enum-thermo FETCH_HEAD
```

- 任务 3 用 `git fetch origin review/open-world-o1` 后 `checkout -b review/open-world-o2 FETCH_HEAD`。
- **正式运行**一律从已推送提交的干净克隆跑，目录 `$R/source-<短SHA>`；`git status --porcelain` 必须为空。
- n2 访问 GitHub 走 SSH，HTTPS 不稳定。n2 的 git 身份已配置。
- **提交规范**：
  - subject 用 `feat:`/`fix:`/`test:`/`docs:`/`chore:` 前缀，祈使句；
  - 新增文件后运行 `python scripts/validate_bootstrap.py --refresh`，`bootstrap_inventory.json` 放进同一提交；
  - commit 末尾四行 trailer：`Coding-Agent`、`<agent>-Version`、`Model`、`Reasoning-Effort`，填真实值，查不到写 `unspecified`；
  - PR 描述 = 英文说明 + `中文说明` 段（至少三句完整中文，本科生能看懂），写清行为变化、验证命令和输出、证据边界、Slurm 作业号和算力，末尾四行署名；
  - 不提交 > 1 MB 的数据文件。大文件留在 n2，把 SHA-256 写进 `SHA256SUMS`。

### 2.5 Python 环境与外部资产（只读使用）

| 名称 | 路径 | 用途 |
|---|---|---|
| `xtbflow` 环境 | `/home/lhshen/miniconda3/envs/xtbflow/bin/python`（PyTorch cu128、tblite 0.7.0、RDKit 2024.03.3） | 生成、解码、目录匹配、GFN2-xTB |
| `aimnet` 环境 | `/home/lhshen/xtbflow-runs/review-align-20261009/envs/aimnet/bin/python`（Python 3.11、aimnet 0.2.0、torch 2.10.0+cpu、RDKit 2026.03.6、geomeTRIC 1.1.1；**没有 tblite**） | AIMNet2，以及任务 1 的产物几何 |
| `xtbflow-qc` 环境 | `/home/lhshen/miniconda3/envs/xtbflow-qc/bin/python`（PySCF、gpu4pyscf、geomeTRIC；只能在 Slurm GPU 作业里导入 gpu4pyscf） | 任务 3a 的 DFT |
| AIMNet2 模型 | `review-align-20261009/tools/aimnet2_wb97m_d3_0.pt`，sha256 `f0f7c054539ad3261bd36f9b11c56d12f87cb723e25bea7521755bbd3ec24e28` | `REVIEW_AIMNET_MODELS=/home/lhshen/xtbflow-runs/review-align-20261009/tools` |
| g-xTB | `review-align-20261009/tools/gxtb/xtb-6.7.1/bin/xtb`，sha256 `1b4e30b68ed4e88b4075f60d97f4756ee440fe92d3294cc20826d53f8121cd26` | `REVIEW_GXTB_BINARY=<该路径>`；**不要加 `--uhf 0`**（见 #114 的 `LICENSES.md`） |

**物理打分统一用 #114 的 `scripts/review_oracle_benchmark.py` 里的 `energy(method, z, x, cfg)`**，method ∈ {`GFN2-xTB`, `g-xTB`, `AIMNet2`}，返回 (kcal/mol, status)，`cfg` 用 `configs/review/oracle_benchmark.json` 的 `xtb` 块。

**环境分工：**
- 解码、`valid_endpoint`、`canonical_event`、`CatalogueMatcher` 一律在 `xtbflow` 环境里做，保持与前几轮一致（RDKit 版本不同）；
- AIMNet2 在 `aimnet` 环境里做，中间结果用 JSON / npz 交接。

---

## 3. 科学纪律（硬性规则）

1. **全部探索性。** 每份报告的第一句写"本报告全部结果为探索性"，并给出主结果的数值和区间。
2. **先冻结、后看数据。**
   - 每个任务先提交并推送配置 JSON，把本文规定的定义、比较、读法、门禁写进去，再跑任何正式计算；
   - 冒烟测试只检查流程能否跑通，不看、不输出科学指标；
   - 结果出来后若要改，开新版本，两版都报告。
3. **不调参。** 本文给定的阈值、网格、样本规则都不按结果调整。
4. **区间方法。**
   - 指标先在（母体, 训练 seed）层面算，对 seed 求平均得到母体值；
   - 配对差按母体算；
   - 再用 `xtbflow.v1.metrics.cluster_summary(values, formula_of)`（分组键 = 母体的 `split_group`）给双侧 95% 区间；
   - 比例用 Wilson 95% 区间。
5. **不声称化学结论。** 代理命中 ≠ 认证的 TS；xTB 认证 ≠ DFT 认证；DFT 认证的 10 条择优样本不能外推总体发现率。
6. **确定性。** 所有排序有确定的平局规则（sha256），随机数用 `xtbflow.v1.rng.addressed_seed`。验收时我会从你的原始输出重跑分析，结果必须逐字节相同。

---

## 4. 输入数据（n2，只读，用前核对哈希）

设 `V=/home/lhshen/xtbflow-runs/v1a-20261007`，`RA=/home/lhshen/xtbflow-runs/review-align-20261009`，`SM=/home/lhshen/xtbflow-runs/explore-smc-20261009/smc-2116d43`。

| 输入 | 内容 |
|---|---|
| `$V/data/{parent_catalog.json, reference_catalog.jsonl, split_manifest.json}` | 母体、参考记录（`x_ts`、`b_p` 已映射到锚定原子顺序）、划分。读法见 `scripts/explore_geometry_information.py` 的 `load_inputs` |
| `$RA/enum-screen-9403d6c/<parent_id>.{npz,scores.npz,json}` | #113 的 screen b3f3 枚举：`channels`、`b`（int8 BE 矩阵）；`S_E`/`S_N` 分数；元数据 |
| `$RA/enum-dev-7c28436/` | 同上，dev |
| `/home/lhshen/xtbflow-runs/v1a-r-20261008/extract/streams.jsonl.gz` | V1a screen 五臂候选流（#113 的生成侧输入） |
| `$SM/`（#116 原始输出） | `g1-*.pt`（检查点 k = 30, 35, 40, 45, 50 的 b_trace/x_trace/b_hat/x_hat）、`g2-*.pt`、`plans.json`、`endpoint_rows.json`、`none_decoded.json`、`checkpoint_scores.json`；哈希见分支 `explore/smc-selection` 的 `docs/evidence/explore_smc/SHA256SUMS` |
| `$RA/open-621b900/<组>/<候选或参考 ID>/endpoint/verdict.json` | #115 的 xTB 认证链；`ts.x` 是 xTB 优化后的 TS 几何（Å），`ts.energy_hartree` |
| `$V/screen/efficiency_s0.candidates.jsonl` | screen、训练 seed 0 的逐候选结果，含 `candidate_id`、`b_dec`（解码事件） |
| 检查 A 的 558 条 dev 参考记录 | 本分支 `docs/evidence/explore_geoinfo/check_a_rows.jsonl` |

**要复用的函数**（import，不复制改写）：
- `xtbflow.v1.enumeration.enumerate_events`；
- `xtbflow.v1.review_enum_metrics` 的 `random_recall`、`generator_curve`、`ranked_curve`、`paired_auc`；
- `scripts/review_enum_analyze.py` 的 `analyze`；
- `xtbflow.v1.smc` 的 `selection_plan`、`restart_state`、`population_metrics`、`score_key`、`tie_hash`；
- `scripts/explore_smc.py` 的各阶段函数和 `decode_endpoint`；
- `xtbflow.v1.explore_rollout.event_direction`；
- `xtbflow.m0.metrics.be_to_mol`；
- `xtbflow.v1.metrics.cluster_summary`；
- 任务 3：`xtbflow.v1.qc_protocol` 的 `Method`、`Meter`、`certification_chain`、`minimum_stage`、`finish`。

---

## 5. 任务 1：YARP 式枚举基线（T1b）

### 5.1 问题

#113 用 b3f3 穷举，再用只见过真实反应的势垒头 S_N 排序。综述里成熟的做法（YARP）是：断 ≤2、成 ≤2（b2f2）穷举 → 产物几何 → 便宜的热力学筛选 → 再排序送验。问：换成这种更合理的枚举流水线后，要找到最低势垒的已知通道，它需要送验的不同事件数，是否仍多于 B0？

### 5.2 步骤

**T1b-0 复现门禁（先做）。**
- 用本分支代码重跑 #113 的分析：

```bash
python scripts/review_enum_analyze.py --enumdir $RA/enum-screen-9403d6c --devdir $RA/enum-dev-7c28436 --out <你的目录>/t1_repro.json
```

- 结果必须与分支上的 `docs/evidence/review_enum/results.json` **逐字节相同**；
- B0 在前 n 个完成候选里命中最佳事件的比例必须为 0.107212 / 0.191033 / 0.309942 / 0.468811 / 0.625731（n = 1, 2, 4, 8, 16；与 PR #107 全精度值 |Δ| < 1e-9）。

**T1b-1 b2f2 子集。**
- 从已有的 b3f3 枚举里取 b2f2 子集，不重新穷举：
  - Δb = b − b_r，只看 i<j 的非对角元；
  - 断键数 = Σ max(0, −Δb_ij)，成键数 = Σ max(0, Δb_ij)；
  - 两者都 ≤ 2 的事件即 b2f2。b3f3 的其余约束（对角元只在键变化原子上改变、Δ ∈ {−2, 0, +2}、电子守恒、|形式电荷| ≤ 1）与 b2f2 相同，所以子集是精确的。
- **一致性测试（必须）**：取 screen 上 N_enum 最小的 3 个母体，用 `enumerate_events(z, b_r, perms, size=2)` 重新穷举，`channel_id` 集合必须与过滤结果完全相同。
- 统计每个母体的 b2f2 事件数、目录通道覆盖率、最佳通道覆盖率。

**T1b-2 产物热力学流水线**（`aimnet` 环境）。对每个事件的 BE 矩阵 b，以及反应物 b_r：
1. `mol = be_to_mol(z, b)`；`Chem.GetMolFrags(mol, asMols=True)` 拆成碎片。
2. 每个碎片：ETKDGv3 嵌入一个构象。
   - `randomSeed = int(sha256(f"{parent_id}|{channel_id}|{fragment_index}").hexdigest()[:8], 16) % 2**31`；反应物的 channel_id 记作 `reactant`；
   - 失败就用 `useRandomCoords=True` 重试一次；
   - 然后 MMFF94 优化（maxIters = 1000）；没有 MMFF 参数就用 UFF；都没有就保留嵌入几何，状态记 `no_forcefield`。
3. 碎片质心沿 x 轴依次相距 10 Å 摆放，合成一个体系。
4. AIMNet2（`aimnet2_wb97m_d3_0.pt`）单点能，总电荷 0，换算成 kcal/mol。
5. ΔE_rxn = E(产物) − E(反应物)，两者都走同一条流水线。
6. 任一步失败：状态记失败原因，ΔE = 无。

**T1b-3 阈值 τ（只用训练集，在碰 screen 之前完成）。**
- 对训练集 386 个母体的全部目录通道（`reference_catalog` 中每个母体的不同 `channel_id`，取该通道任一记录的 `b_p`），用同一流水线算 ΔE_rxn；
- τ = 成功值的第 95 百分位（`numpy.percentile`，默认线性插值）；
- 把 τ、成功率和分布写进 `configs/explore/enum_thermo_tau.json`，**提交并推送后**才能开始 screen 的计算。

**T1b-4 screen 上的打分器**（全部只在 b2f2 子集上）：

| 打分器 | 排序规则（升序，平局按 sha256(channel_id)） |
|---|---|
| `S_rand2` | 随机，用 `random_recall` 的解析期望 |
| `S_N2` | 已有的 S_N 分数，不加物理 |
| `S_thermo` | ΔE_rxn；失败的排在全部成功者之后，按 S_N 排 |
| `S_N_thermo`（主） | 先放"成功且 ΔE_rxn ≤ τ"的事件，按 S_N 排；再放其余事件，按 S_N 排 |

**T1b-5 召回与 AUC。**
- 与 #113 完全相同：k 网格 {1, 2, 4, 8, 16, 32, 64}，AUC 取 k ∈ {1, 2, 4, 8, 16}，log₂k 梯形权重 [1, 2, 2, 2, 1]；删失规则和读法见 `configs/review/enum_baseline.json`；
- 直接调用 `ranked_curve` / `paired_auc`；
- 生成侧沿用 #113 的输入，不重算。

### 5.3 比较与读法（写进配置 `configs/explore/enum_thermo.json`）

- **主比较**：AUC(`S_N_thermo`) − AUC(B0)。
  - 双侧下界 > 0 → `ENUMERATION_BEATS_GENERATION`；
  - 上界 < 0 → `GENERATION_BEATS_ENUMERATION`；
  - 否则 `INCONCLUSIVE`。
- **次要比较**（报区间，不设闸门）：
  - `S_thermo` − B0；`S_N2` − B0；
  - `S_N_thermo` − `S_N2`：物理筛选本身有没有用。下界 > 0 记 `THERMO_FILTER_HELPS`；
  - `S_N_thermo` 对 A0、A1、A2、B1。
- **另报**：
  - b2f2 的最佳通道覆盖率（枚举的天花板）；
  - 只在"最佳通道属于 b2f2"的母体上重算主比较（条件分析）；
  - 最佳通道不在 b2f2 内、但生成臂命中的母体数；
  - 每个母体的 b2f2 事件数中位数和 95% 分位；
  - 流水线成功率与失败原因；
  - 每个事件的平均耗时；
  - 被 τ 筛掉的比例。

### 5.4 预算与冒烟

- 先在 5 个训练母体（按 parent_id 排序取前 5）上测每个事件的平均耗时，乘以 screen 的 b2f2 事件总数得到预估。
- 预估 > 60 核时就改用预写的子集：按 sha256(parent_id) 升序取前 171 个 screen 母体，报告里写明。
- 冒烟用 2 个 dev 母体，只检查流程。

### 5.5 交付

- 证据目录 `docs/evidence/explore_enum_thermo/`：`REPORT_ZH.md`、`results.json`、manifest、`SHA256SUMS`；
- 一张 k 对召回率的图。

---

## 6. 任务 2a：反应坐标 u 是不是反应模？

### 6.1 问题

pg-1 的推动和 smc-1 的弛豫打分器都用 u = `event_direction(b̂, b_r, x)`：由预测键级变化给出的方向，成键原子对互相靠近、断键原子对互相远离。只有当 u 接近 TS 的负曲率反应模时，(I − 2uuᵀ)F 才是"鞍点感知"的。smc-1 里，真实 TS 经过弛豫后只有 17% 沿 u 为负曲率，但那是在弛豫跑偏之后测的。这里直接在**未弛豫的真实 TS** 上测。

### 6.2 计算（558 条 dev 参考记录）

对每条记录：
- x = 参考记录的 `x_ts`（锚定顺序），u = `event_direction(b_p, b_r, x)`。
- **GFN2-xTB Hessian**：解析梯度的中心差分，步长 0.005 Å，对称化，单位 kcal/mol/Å²；梯度用 `explore_geometry_information._force_job`，换算方式同 `xtbflow.v1.xtb_score.gradient_to_force`。投影掉平动和转动。
- 输出：
  - **虚频数**：质量加权、投影后的 Hessian 对角化后换算成 cm⁻¹，ν < −30 cm⁻¹ 的个数。质量用最丰同位素：H 1.00782503、C 12.0、N 14.0030740、O 15.9949146。
  - **c_u = uᵀHu**。
  - **overlap**：|⟨v_min, u⟩|，v_min 是投影后的笛卡尔 Hessian 最低本征值对应的单位本征向量。
  - **best_neg_overlap**：所有负本征值的本征向量中 |⟨v, u⟩| 的最大值。
- **次要**：用 AIMNet2 的力（`AIMNet2Calculator(..., forces=True)`）重复 c_u 和 overlap。
- **单元测试**：
  - 对一个已优化的小分子极小点（例如 GFN2 优化后的水），6 个零模之外全为正频；
  - Hessian 对称。

### 6.3 读法（写进配置 `configs/explore/u_diagnostic.json`）

用 GFN2 主结果：

| 读法 | 条件 |
|---|---|
| `U_IS_REACTION_MODE` | overlap 中位数 ≥ 0.7，且 c_u < 0 的比例 ≥ 0.7 |
| `U_NOT_REACTION_MODE` | overlap 中位数 < 0.3，或 c_u < 0 的比例 < 0.5 |
| `U_PARTIAL` | 其他 |

另报：
- 恰有一个虚频的记录比例（真实 DFT TS 在 GFN2 势能面上是不是一阶鞍点）；
- 按事件大小（断键数 + 成键数）分层的 overlap 中位数；
- overlap 和 c_u 的母体平均及聚类区间。

---

## 7. 任务 2b：用 AIMNet2 / g-xTB 重做中途筛选（smc-2）

### 7.1 问题

smc-1 用 GFN2 打分，而 GFN2 在生成几何上的排序能力只有 ρ ≈ 0.51。换成更好的 AIMNet2（0.62）或 g-xTB（0.60）后，"在生成中途按物理剪枝、克隆"能否胜过"生成完再用同一打分排序"？这是并联机制线的最后一次检验。

### 7.2 设计（尽量复用 smc-1，只换打分器）

- **复用**：smc-1 的 G1 输出（不推臂完整轨迹和检查点状态）、全部 13 个臂的终点、`plans.json`。用前按 #116 的 `SHA256SUMS` 核对哈希。不重新生成 none、random_*、raw_*、relaxed_* 这 13 个臂。
- **新检查点打分器**（对检查点 k ∈ {30, 35, 40, 45} 的预测终点 x̂₁）：
  - `aimnet`：E_AIMNet2(x̂₁) − E_AIMNet2(母体锚定 x_r)；
  - `gxtb`：同样用 g-xTB；
  - 都用 #114 的 `energy()`；最小原子间距 < 0.5 Å 或计算失败记为失败。
- **新臂**（8 个）：`aimnet_{0.6,0.7,0.8,0.9}`、`gxtb_{0.6,0.7,0.8,0.9}`。
  - 剪枝—克隆规则与 smc-1 的 `raw` 完全相同：成功者在前、按势垒升序、平局 `tie_hash(parent_id, seed, proposal)`、保留 16 个、每个幸存者一个克隆；
  - 克隆用 `restart_state`，噪声键与 smc-1 相同（不含臂名）；
  - 续跑与 smc-1 的 `cmd_g2` 相同：`sampler.rollout(start=k_c)`，批次布局不变。一个 Slurm 作业跑完 8 个臂（预计 < 2 分钟）。
- **终点排序**（全部 21 个臂，含复用的 13 个）：
  - **主排序器**：终点几何上的 AIMNet2 原始能量；
  - **次排序器**：g-xTB 原始能量、GFN2 原始能量（GFN2 的值可直接取 smc-1 `endpoint_rows.json` 的 `score.raw_barrier`）；
  - 种群指标用 `population_metrics(rows, best, curvature_layers=False)`，其中 `row['score']` 换成对应排序器的 {status, barrier}。
- **主指标**：H = mean(hit@1, hit@2, hit@4)，定义与 smc-1 相同。次要指标同 smc-1：hit@∞、种群平均事件效用、不同合法事件数、合法率、best-reference 命中率。
- **诊断**：
  - 各检查点上 aimnet/gxtb 打分与终点事件目录势垒的同母体 Spearman，算法同 smc-1 的 `checkpoint_correlations`；
  - 误剪率、克隆事件改变率、仅克隆命中最佳；
  - 按 #117 的方法把 ΔH 分解为"可用性变化"（hit@∞）和"排序损失变化"。

### 7.3 主比较与读法（写进配置 `configs/explore/smc_oracle.json`，版本 `smc-2`）

- **主臂** `aimnet_0.7`，主排序器 AIMNet2。H 对 `none` 和对 `random_0.7` 的配对差。
- 读法与 smc-1 相同：

| 读法 | 条件 |
|---|---|
| `IN_GENERATION_SELECTION_HELPS` | 对 none 下界 > 0，且对 random 下界 > 0 |
| `NOT_PHYSICS_SPECIFIC` | 对 none 下界 > 0，对 random 区间触及 0 |
| `NO_GAIN_OVER_POSTHOC` | 对 none 区间含 0 |
| `SELECTION_HURTS` | 对 none 上界 < 0 |

  其余情形记 `UNCLASSIFIED_BY_FROZEN_TABLE`。
- **每个新臂都给读法**，三个排序器都报。
- **对这条研究线的预写建议**（只是建议，由负责人决定）：
  - 主臂为 `NO_GAIN_OVER_POSTHOC` 或 `SELECTION_HURTS`，且 8 个新臂在主排序器下都不是 `IN_GENERATION_SELECTION_HELPS` → `LINE_CLOSURE_RECOMMENDED`；
  - 有任一新臂为 `IN_GENERATION_SELECTION_HELPS` → `NEEDS_HELDOUT_CONFIRMATION`（本任务不做确认）。

### 7.4 门禁（任一不过就停）

1. AIMNet2 和 g-xTB 在 558 条真实 TS 上复现 #114：ρ = 0.9239845589178955 和 0.9089169115946805（|Δ| < 1e-9），汇总方法同检查 A 的 `min` 汇总。
2. smc-1 输入文件哈希全部与 #116 的 `SHA256SUMS` 一致。
3. 用 smc-1 原来的 relaxed 终点分数重算 13 个臂的母体指标，与 smc-1 `results.json` 的 `parent_metrics` 完全相同。
4. 新臂幸存者的终点事件与 none 的一致率 ≥ 0.999，报告最大坐标差。

---

## 8. 任务 3：开放世界第二阶段（DFT）与 AIMNet2 数据核查

### 8.1 任务 3a：至多 10 条 xTB 新通道的 DFT 认证

负责人已批准 #115 的第二阶段提案（`docs/evidence/review_open_world/STAGE2_PROPOSAL_ZH.md`）和算力。

**样本**（先冻结后计算）：
- 从 #115 的 `results.json` / `verdicts.csv` 取 `OUTSIDE_CATALOGUE_EVENT` 组中状态为 `STRICT_JOINT_GRAPH_VALID`、且同母体 `BASELINE` 链也是 `STRICT_JOINT_GRAPH_VALID` 的候选（应为 12 个母体）。
- 按 ΔEa_xTB 升序排，平局 sha256(candidate_id)，每母体最多一个，取前 10。
- 把 10 个 candidate_id、对应 baseline 的 reference_id 和母体写进 `configs/review/open_world_o2.json`，在**任何 DFT 作业之前**提交并推送。
- 名单必须能从 #115 的文件独立重现，我会重抽核对。

**每个入选母体 3 个计算项：**

| 项 | 起点几何 | 事件 |
|---|---|---|
| `candidate` | #115 该候选 xTB 链的 `ts.x`（xTB 优化后的 TS） | 候选的 `b_dec`（`$V/screen/efficiency_s0.candidates.jsonl`） |
| `reference` | 该母体 #115 BASELINE 链所用 reference_id 的 T1x `x_ts` | 该记录的 `b_p` |
| `anchor` | 母体锚定反应物 `x_r` | — |

- `reference` 的原子顺序处理照抄 `scripts/v1b_d1.py` 的 `plan()`：`perm = permutations[0]`，`x_ts[perm]`，`b_p[np.ix_(perm, perm)]`。

**代码：**
- 新脚本 `scripts/review_open_world_dft.py`，子命令 `plan` 和 `run --item <id>`。
- `run` 照抄 `scripts/v1b_d1.py` 的 `run()`：
  - 调用 `qc.certification_chain(z, x, b_r, perms, b_predicted, qc.Method(device='gpu'), work, meter)` 和 `qc.minimum_stage(...)` + `qc.finish(...)`；
  - 写 `verdict.json` 和 `environment.json`。
- 协议版本必须是 `v1b-h-gpu4pyscf-3`（ωB97X/6-31G(d)，gpu4pyscf，严格鞍点、双向 IRC、端点频率和图身份认证）。
- **不修改 `qc_protocol.py`**；端点低频软模按协议原样判定，不新增规则。
- 协议内置的重试照常执行（`certification_chain` 里的 TS 收紧重试、`minimum_stage` 的一次统一端点重试）。除此之外不做任何重试，`IRC_LIMIT` 等状态如实记录；想额外重试就停下来问。

**作业：**
- 每个计算项一个 Slurm 作业：`--gres=gpu:pro6000:1 --cpus-per-task=8 --mem=64G --time=02:00:00`，环境 `xtbflow-qc`。
- 先跑一个最小母体的 `anchor` 确认流程，再按"anchor → reference → candidate"的顺序逐母体推进。
- 严格串行：提交下一个作业前先 `squeue -u lhshen`，有任何作业在跑或排队就等它结束。用一个登录 shell 上的小驱动脚本循环提交，每次提交前检查一次。这样别的任务（例如 2b 的 GPU 作业）可以插在两个 DFT 作业之间。
- 超时记 `JOB_TIMEOUT`，不重跑。

**指标：**
- 候选与基准各自的严格认证通过数，附 Wilson 区间。
- 对候选和基准都通过的母体：
  - ΔEa_DFT = E_DFT(候选 TS) − E_DFT(基准 TS)，kcal/mol；
  - 各自相对 anchor 的势垒；
  - 基准势垒与目录势垒之差（健全性检查，应接近 0；若 |差| > 2 kcal/mol 要在报告里讨论）；
  - ΔEa_DFT 与 #115 的 ΔEa_xTB 的符号一致率和散点。
- 报告 ΔEa_DFT < 0 和 ≤ +5 kcal/mol 的个数。
- 每条链的阶段计时、GPU 墙钟、梯度和 Hessian 调用数。
- **不要**把 SCF 或优化失败说成"通道不存在"；不要用 10 条择优样本推出总体发现率；这些母体属于 screen 集，报告里写明。

**预算**：GPU 合计 ≤ 16 小时（D1 实测每条候选链约 0.68 GPU 小时、参考链 0.25–0.7、anchor 约 0.04）。累计超过 16 小时就停。

### 8.2 任务 3b：AIMNet2 训练数据是否含 T1x

- 只读公开资料：AIMNet2 和 AIMNet2-rxn 的论文 / 预印本（含 SI）、`github.com/isayevlab/aimnetcentral` 的文档与模型注册信息、数据集说明。
- 确定 `aimnet2_wb97m_d3_0.pt` 和 `aimnet2_rxn_0.pt` 的训练数据来源，判断是否包含 Transition1x 的结构（或由 T1x 反应派生的结构）。
- **交付**：`docs/evidence/aimnet_provenance/REPORT_ZH.md`（放在 3a 的分支上）。
  - 每条判断附原文出处：URL + 章节或表格号，引用不超过 15 个词；
  - 结论三选一：`INCLUDES_T1X`、`EXCLUDES_T1X`、`UNRESOLVED`。
- 不接受任何许可、不登录、不下载 > 100 MB 的东西。

---

## 9. 什么时候必须停下来

- 任何复现门禁不通过（5.2 的 T1b-0、7.4、b2f2 一致性测试），或输入哈希与本文 / #116 记录不符；
- 预估算力超出本文预算：任务 1 CPU 60 核时（且子集方案也超）、任务 2 GPU 0.5 小时或 CPU 10 核时、任务 3 GPU 16 小时；
- 需要修改已有文件、本文的定义 / 读法 / 样本规则，或需要新装包；
- 任务 3a 出现协议外的失败类型，或想做协议内置之外的重试；
- 与其他任务冲突：负载持续 > 120，你的作业排队超过 2 小时，发现别人的作业或目录被你影响；
- 想用 M0 val/test、做本文以外的 DFT、碰 V1b 正式计算。

停下时写清：卡在哪里、有哪些选项、各自的代价。写在 PR 描述或 `$R/STATUS.md` 里，负责人回来会看。不要等回复时空转：可以先推进与阻塞无关的其他任务。

---

## 10. 交付与汇报

**三个 PR，都不合并：**
- `explore/enum-thermo` → base `explore/followup-base`；
- `explore/smc-oracle` → base `explore/followup-base`；
- `review/open-world-o2` → base `review/open-world-o1`。

全部完成后（或在任何停止点），给负责人写不超过 12 行的汇总：
- 任务 1：主比较数值、区间、读法；`THERMO_FILTER_HELPS` 与否；b2f2 最佳通道覆盖率；
- 任务 2a：读法、overlap 中位数、c_u < 0 比例；
- 任务 2b：`aimnet_0.7` 对 none / random 的 ΔH、区间、读法；研究线建议；
- 任务 3a：候选和基准的 DFT 通过数；ΔEa_DFT 列表；与 xTB 的符号一致率；
- 任务 3b：结论和关键出处；
- PR 链接、Slurm 作业号、实际 GPU / CPU 用量。

---

## 11. 验收清单（Claude Code 会逐项核对）

**通用**

1. PR 的 base 正确、未合并；`git diff --name-status <base>...<head>` 只有新增文件，外加 `bootstrap_inventory.json`。
2. 每个任务的配置首次推送早于所有正式结果；冒烟没有输出科学指标；任务 1 的 τ 文件推送早于 screen 计算；任务 3a 的样本配置推送早于第一个 DFT 作业。
3. 每个结果 manifest 记录 `source_commit`（已推送）、`dirty=false`、配置 sha256、输入文件 sha256、Slurm 作业号（CPU 阶段为 null）；n2 上存在对应的 `source-<sha>` 干净克隆。
4. 我从你的 source 克隆和原始输出重跑分析，`results.json` 逐字节相同。
5. 在 n2 上通过 `python scripts/validate_bootstrap.py`、`python -m unittest discover -s tests/reusable -p "test_*.py"`、`python -m pytest -q tests --ignore=tests/reusable`；新增纯函数都有单元测试。
6. `REPORT_ZH.md` 第一句符合第 3 节第 1 条；每个数字都能在 `results.json` 里找到；`SHA256SUMS` 与文件一致；偏差和限制写全。
7. GPU 只经 Slurm，同一时刻最多 1 个作业，作业号齐全；CPU 用了 nice、worker ≤ 48；没有写入其他任务的目录或环境；没有取消别人的作业。

**任务 1**

8. T1b-0 逐字节复现与 B0 数字复现通过；b2f2 一致性测试通过。
9. τ 只来自训练集；流水线的种子、碎片处理、失败处理与 5.2 一致（我会抽 50 个事件在 `aimnet` 环境里独立重算 ΔE_rxn，要求 |差| ≤ 1e-4 kcal/mol）。
10. 若用了子集方案，子集按 5.4 的规则选取，报告写明。

**任务 2**

11. 2a 的 Hessian 单元测试通过；我会抽 10 条记录独立重算 c_u 和 overlap。
12. 7.4 的四项门禁全部通过；新臂的选择方案按规则可重现（我会重算全部方案）；没有重新生成复用的 13 个臂。
13. 读法和研究线建议严格按 7.3 给出。

**任务 3**

14. 样本名单可从 #115 的文件独立重现；DFT 协议版本为 `v1b-h-gpu4pyscf-3`，`qc_protocol.py` 未改；每个计算项有 `verdict.json`、`environment.json`、Slurm 记录；除协议内置重试外没有额外重试。
15. ΔEa_DFT 只对候选和基准都通过的母体计算；GPU 合计 ≤ 16 小时。
16. 3b 的每条判断都有可核对的出处；没有接受许可或登录。

验收不通过的项，我会在 PR 上用行内评论指出，修好后再验。

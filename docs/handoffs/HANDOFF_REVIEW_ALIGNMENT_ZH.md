# xtbflow 交接：综述对齐的三项探索性任务（枚举基线、物理预言机、开放世界终点）

2026-10-08 · 交给一个从未接触过本项目的 AI 执行 · 计算在 n2 上 · 完成后由 Claude Code（本文作者）验收

本文自成一体：读完它就能开工，不需要读之前的对话。凡是写着"停下来问负责人"的地方，必须停下，不要自行决定。

---

## 0. 一页摘要

**要做什么。** 三项相互独立、全部标为"探索性"的任务，各开一个分支、一个 PR，不合并：

| 任务 | 回答的问题 | 主要算力 | 分支（基线） |
|---|---|---|---|
| T1 枚举基线 | "穷举所有小幅键变化 + 便宜打分"能否比 xtbflow 的生成模型更快找到最低势垒的已知反应通道？ | CPU 为主，GPU ≤2 小时 | `review/enum-baseline`（基于 `explore/geometry-lead-clock`） |
| T2 物理预言机 | 哪种便宜的能量方法最会给同一反应物的各通道按势垒排序？在模型生成的不完美几何上还剩多少？ | CPU 为主，GPU ≤1 小时 | `review/oracle-benchmark`（基于 `explore/geometry-lead-clock`） |
| T3 开放世界终点（第一阶段） | 模型生成的"目录外"候选里，有多少在 xTB 层面是真实的新反应通道？ | 只用 CPU | `review/open-world-o1`（基于 `codex/v1b-20261007`） |

**不做什么。** 不做任何 DFT 计算；不碰 V1b（负责人已暂缓）；不改任何已有的证据、配置或脚本；不合并任何 PR。

**建议顺序。** T1 → T2 → T3。T1 最便宜、对项目方向最有决定性。

---

## 1. 项目背景（给完全不了解的人）

### 1.1 xtbflow 在做什么

xtbflow 是一个科研项目，目标是：**只给反应物**，用生成模型提出"会发生什么反应"和"过渡态（TS）大致长什么样"，再用便宜的量子化学方法（GFN2-xTB）和少量 DFT 去验证。核心主张是：在同等物理计算成本下，比规则枚举或"先定事件、再生成几何"的串联模型发现更多有效的竞争反应通道。

关键术语：

- **母体（parent）**：一个反应物分子（闭壳层、中性、元素 C/H/N/O，≤23 个原子）。
- **事件 / 通道（event / channel）**：反应物到产物的键变化。用"键—电子矩阵"（BE 矩阵）表示：对角元是该原子的孤对电子数，非对角元是键级。产物 BE 矩阵减去反应物 BE 矩阵就是事件；对反应物的分子自同构取商后得到规范的事件 ID `channel_id`（函数 `xtbflow.v1.data.canonical_event`）。
- **目录（catalogue）**：数据来自 Transition1x（T1x，ωB97X/6-31G(d) 的 NEB/CINEB 反应路径数据集）。每个母体有若干条已知反应（参考记录），每条有 TS 几何和"目录势垒" `catalog_barrier_kcal`（TS 能量减去该母体锚定反应物能量，kcal/mol）。**最佳通道**是目录势垒最低的那个（或几个并列）通道。
- **联合 / 并联模型（joint, B 臂）**：一个流匹配模型同时生成事件 b 和 TS 几何 x。**串联模型（cascade, A 臂）**：先生成事件，解码后再生成几何。
- **V1a 的五个臂**：A0（串联无引导）、A1（串联 + 用只看事件的势垒头 h_E 引导事件）、A2（强自适应串联搜索）、B0（联合无引导）、B1（联合 + 用看几何的势垒头 h_X 引导几何）。
- **代理状态（proxy_status）**：生成候选与目录比对的结果。`PROXY_MATCH`（事件已知且 TS 几何 RMSD ≤0.5 Å）、`KNOWN_EVENT_GEOMETRY_MISS`（事件已知但几何不匹配）、`OUTSIDE_CATALOGUE_EVENT`（合法事件但目录里没有）、`INVALID_OUTPUT`（解码不出合法结构）。
- **划分**：V1a 把 T1x 官方训练集按分子式分组切成训练 386 个母体、开发集（dev）62 个母体/21 个分子式组、正式筛查集（screen）342 个母体/70 组。官方 val/test 已在更早的 M0 实验用过，**永远不要碰**。

### 1.2 已经确立的事实（全部在 GitHub 上，PR 均未合并）

| 实验 | PR | 结论（一句话） |
|---|---|---|
| M0 联合 vs 串联 | #101–#105 | 无引导时打平（Δ = −0.015，95% 区间 [−0.067, 0.033]），联合的合法率更低 |
| V1a 代理筛查 | #106 | 正式决策 GO_V1b：B1−A2 的成本–召回 AUC +0.093；但无引导的 B0 最高（0.494），B1 只有 0.265 |
| R 零算力重分析 | #107 | 按候选数对齐时 B0≈A0；事件引导（A1）提高单候选命中率，几何引导（B1）几乎不提高 |
| N 步数对齐对照 | #109 | B0 的预算优势可由积分步数解释 |
| 几何信息检查 | #110 | 给势垒头真实 TS 几何，同母体排序没有可检出增益（配对差 +0.027 [−0.111, +0.165]）；GFN2-xTB 在真实 TS 上排序最好（ρ = 0.749） |
| 几何先行时钟 | #111 | 推迟事件定型能让物理信号的作用窗口从 15% 升到 28–35%，生成质量几乎不变 |
| 物理引导 pg-1 | 无 PR（进行中） | **另一个会话正在跑**，分支 `explore/physics-guidance`，见 2.6 节 |

### 1.3 为什么要做这三项任务

依据：Joung, Casetti, Raghavan, Coley, *An overview of reaction outcome prediction with physics-based and data-driven methods*, Chem. Soc. Rev. 2026, 55, 6768–6813（doi:10.1039/D6CS00079G，CC-BY 4.0）。对照这篇综述，xtbflow 有三处与领域通病同源的薄弱点：

1. **缺外部物理基线（综述 4.1.1、6.2 节）。** 小分子上"最多断 2 个键、成 2 个键"（b2f2）的穷举枚举是成熟做法（YARP 就是 b2f2 穷举 + GFN2-xTB + DFT）。V1a 只比较了内部变体，没比过它 → **T1**。
2. **物理预言机精度（综述 2.2.3 节）。** 综述认为半经验能量通常不够用，点名 g-xTB 和基于 OMol25 / AIMNet2 的机器学习势更有希望 → **T2**。
3. **封闭世界评测（综述 3.1.3、7 节）。** 主指标只认 T1x 目录里的已知通道；V1a 的 B0 候选有约 46% 是目录外事件，它们一律算未命中。如果模型真发现了新通道，现有指标会惩罚它 → **T3**。

---

## 2. 环境与工作方式（硬性规则）

### 2.1 机器与仓库

- **远程仓库**：`https://github.com/askusk521-cyber/xtbflow`，SSH 地址 `git@github.com:askusk521-cyber/xtbflow.git`。默认分支 `main`。**任何合并都需要负责人明确批准；你只开 PR，不合并、不开自动合并、不直推 main。**
- **n2**：主机名 node2，用户 `lhshen`，4×RTX PRO 6000（96 GiB），192 核，Slurm 21.08，Ubuntu 22.04。n2 是单节点：登录 shell 就在计算节点上，**能看到 GPU，但任何 GPU 计算都必须经 Slurm（`srun`/`sbatch`）**。在登录 shell 上做 CPU 冒烟测试时设 `CUDA_VISIBLE_DEVICES=` 防止误用 GPU。
- 仓库根目录的 `AGENTS.md` 是项目规则，开工前读一遍。

### 2.2 n2 上不能动的东西

- `/home/lhshen/xtbflow/` 不是仓库，是一个装着几十个 worktree 的父目录。主 checkout `/home/lhshen/xtbflow/xtbflow` 和所有 `xtbflow-*`、`/tmp/xtbflow-*` 目录都属于别的任务：**不要切分支、重置、stash、改文件**。
- `/home/lhshen/xtbflow-runs/` 下已有的运行目录全部只读。你只在 `/home/lhshen/xtbflow-runs/review-align-20261009/` 下新建自己的目录。
- 现有 conda 环境 `xtbflow`、`xtbflow-qc`、`xtbflow-cp2k` 不得修改（不往里 pip/conda 安装）。需要新包时**新建**环境（见 2.5 节）。

### 2.3 代码同步：只走 Git

1. 在你自己的工作副本里开发：本地机器，或你在 n2 上**新建**的克隆（例如 `/home/lhshen/xtbflow-runs/review-align-20261009/dev-clone`）。
2. commit → push 到你的分支。
3. 正式运行一律从**已推送提交的干净克隆**跑，目录名带 SHA：

```bash
R=/home/lhshen/xtbflow-runs/review-align-20261009; SHA=<短SHA>; BR=<你的分支>
mkdir -p $R && cd $R
git clone -q --no-checkout /home/lhshen/xtbflow/xtbflow-dev source-$SHA   # 从本机已有仓库克隆，避免慢速 HTTPS
cd source-$SHA && git fetch -q git@github.com:askusk521-cyber/xtbflow.git $BR && git checkout -q --detach $SHA
git status --porcelain   # 必须为空
```

n2 访问 GitHub 的 HTTPS 不稳定，SSH 可用；n2 的 git 身份已配置好。

### 2.4 Slurm

- 分区 `main`；单卡 `--gres=gpu:pro6000:1`；不要用旧值 `gpu:5090:1`；**不要手动设置 `CUDA_VISIBLE_DEVICES`**（Slurm 会分配）。
- `squeue -u lhshen`（`$USER` 在这里不可用）。Slurm 记账未开启，`sacct` 用不了，靠日志和状态文件确认完成。
- `sbatch --wrap` 用的是 `/bin/sh`，里面没有 `source`/`conda activate`。直接用解释器的绝对路径：

```bash
S=$R/source-$SHA; OUT=$R/<task>-$SHA
sbatch --parsable --partition=main --gres=gpu:pro6000:1 --cpus-per-task=16 --mem=48G --time=01:30:00 \
  --job-name=ra-enum --output=$OUT/enum-%j.log \
  --wrap "cd $S; export PYTHONPATH=src:vendor/mechai_reusable OMP_NUM_THREADS=1 PATH=/home/lhshen/miniconda3/envs/xtbflow/bin:\$PATH; python scripts/<你的脚本>.py ..."
```

只用 CPU 的作业去掉 `--gres`。
- **并发上限：项目登记为同一时刻 1 个作业**（`configs/resources.yaml`）。提交前先 `squeue -u lhshen`；如果已有作业在跑（包括别的会话的），用 `--dependency=afterany:<jobid>` 排在后面，不要并行。**绝不 `scancel` 不是你提交的作业。**
- 单个作业墙钟 ≤120 分钟。长任务拆成多个串行作业。

### 2.5 Python 环境

| 环境 | 内容 | 用途 |
|---|---|---|
| `xtbflow`（`/home/lhshen/miniconda3/envs/xtbflow/bin/python`） | Python 3.10.19、PyTorch 2.10.0+cu128、NumPy 2.2.6、SciPy 1.15.3、RDKit 2024.03.3、tblite 0.7.0、h5py | T1、T2 |
| `xtbflow-qc` | Python 3.11、PySCF 2.14、gpu4pyscf 1.8.1、geomeTRIC 1.1.1、RDKit；**没有 tblite** | 不用 |
| 你新建的环境 | 例如 `conda create -n xtbflow-xtbchain --clone xtbflow` 后装 `geometric==1.1.1 pyscf==2.14.*` | T3，以及 T2 的新预言机 |

n2 直连 PyPI 很慢（约 100 KB/s）。每次安装临时加 `--index-url https://pypi.tuna.tsinghua.edu.cn/simple`，不要写入持久配置。每个新环境都把 `pip freeze` 存进对应任务的证据目录。

### 2.6 与另一个正在运行的会话共存

另一个 AI 会话正在做"物理引导 pg-1"实验：分支 `explore/physics-guidance`，n2 目录 `/home/lhshen/xtbflow-runs/explore-geoinfo-20261008/pg-*`，作业名以 `pg-` 开头（如 `pg-push`、`pg-analyze`）。

- 不要动它的分支、目录和作业。
- **不要修改 `scripts/explore_geometry_information.py`**，那个分支正在改它，改了会冲突。你只**新增**文件，复用 `src/xtbflow/v1/geometry_information.py` 等模块里的函数。
- 你的 Slurm 作业要排在它的作业后面（见 2.4 节的并发规则）。

### 2.7 提交与 PR 规范

- commit subject 用 `feat:` / `fix:` / `test:` / `docs:` / `chore:` 前缀，祈使句。
- 新增或修改被清单覆盖的文件后，运行 `python scripts/validate_bootstrap.py --refresh`，并把 `bootstrap_inventory.json` 放进同一个提交。`vendor/mechai_reusable/` 不能原地修改；`archive/` 只作历史。
- 在 Windows 上开发时：先 `git config core.autocrlf false`；所有写文件都用 LF（Python 用 `newline='\n'`），否则 bootstrap 哈希会失败。从 PowerShell 管道传给 ssh 的脚本会带 BOM，用 `ssh n2 "echo <base64> | base64 -d | bash"` 传。
- PR 描述按 `AGENTS.md`：英文说明 + 标明 `中文说明` 的段落（至少三句完整中文，本科生能看懂）。要写清行为变化、验证命令和输出、证据边界、Slurm 作业号和算力用量。
- 署名：commit 末尾加四行 trailer，PR/评论末尾加四行（`Coding agent / ... version / Model / Reasoning effort`），填你自己的真实值，查不到的写 `unspecified`。
- 不提交凭据、模型权重、大于 1 MB 的数据文件。大文件留在 n2，只把 SHA-256 写进 `SHA256SUMS`。

---

## 3. 科研纪律（硬性规则）

1. **全部探索性。** 每份报告的第一句写明"本报告全部结果为探索性"，并给出主要发现的数字和区间。不改变 V1a 的任何结论。
2. **先冻结，后看数据。** 每个任务先提交一个分析配置 JSON，把比较对象、指标、网格、区间方法、读法写死，推送后再读评估数据。配置提交必须早于任何结果文件。结果出来后若要改配置，开新版本，两版都报告。冒烟测试只能检查流程是否跑通，不能看指标。
3. **数据范围。** 只用本文列出的输入。不碰 M0 官方 val/test。dev 和 screen 都已被用过（screen 只分析过一次正式结论），本任务里**不能在它们上面调任何超参数**：枚举域、打分器、采样规则全部事先固定。
4. **不改已有东西。** `docs/evidence/` 下已有目录、已有配置、已有脚本、冻结的模型权重和目录文件一律只读。
5. **不做 DFT，不碰 V1b。** T3 的 DFT 阶段只写提案，不运行。
6. **区间方法统一**：先在母体内平均（跨训练 seed），再用分子式聚类三明治 t 区间（`xtbflow.v1.metrics.cluster_summary(values, formula_of)`，分组键用母体的 `split_group`），报告双侧 95%。
7. **不要声称化学结论。** 代理命中不等于认证过的 TS；xTB 层面的认证不等于 DFT 认证。

---

## 4. 输入数据与资产（全部在 n2，只读）

V1a 运行根目录 `V=/home/lhshen/xtbflow-runs/v1a-20261007`。

| 文件 | 内容 | SHA-256（前 16 位） |
|---|---|---|
| `$V/data/parent_catalog.json` | 全部母体：`parent_id`、`query_id`、`atomic_numbers`、`x_r`（锚定反应物，Å）、`b_r`、`permutations`（自同构）、`split_group`、`best_channel_ids`、`best_barrier_kcal`、`reference_ids` | `ebbbee13124444ae` |
| `$V/data/reference_catalog.jsonl` | 每条参考记录：`parent_id`、`reference_id`、`cache_index`、`channel_id`、`b_p`/`x_ts`/`x_r`（已映射到锚定原子顺序）、`catalog_barrier_kcal` | `864b032343046861` |
| `$V/data/split_manifest.json` | `train` / `development` / `screen_reserve` 的 `parent_ids`；screen 342 个母体见 `$V/formal/freeze.json` 的 `parent_ids` | `4ec80a3e0246be76` |
| `/home/lhshen/data/t1x/t1x_m0_v1.npz` | T1x 缓存，用 `xtbflow.m0.t1x_data.T1xCache(path).reaction(i)` 读，返回原始原子顺序的 `z`（原子序数）、`x_r`、`x_ts`、`x_p`、`b_r`、`b_p` | `7eb681a252a93fc5` |
| `$V/scores/E_{101,102,103}/` | V1a 冻结的只看事件的势垒头 h_E，用 `xtbflow.v1.assets.load_scores(V, 'E', device)` 加载 | — |
| `/home/lhshen/xtbflow-runs/explore-geoinfo-20261008/full-68f5733/heads/clean_heads/reactant_{101,102,103}.pt` | PR #110 配对重训的事件势垒头（下称 **N_clean**），比 h_E 强（同母体排序 ρ 0.693 对 0.608）。加载方法见 5.4 节 | `8c02aa36cddc3119` / `30f5455995b90f48` / `141ff11f4e9c20da` |
| `/home/lhshen/xtbflow-runs/v1a-r-20261008/extract/streams.jsonl.gz` | R 任务提取的 V1a screen 全部候选流：5,130 行 = 342 母体 × 5 臂 × 3 训练 seed。每行有 `parent_id`、`split_group`、`arm`、`training_seed`，以及按完成顺序排列的 `channel`（每个完成候选的 `channel_id`，非法为 null）、`event`（是否命中最佳事件）、`proxy_status`、`completion_units`，还有 `opens['D1']`（该候选是否开启一个新的不同事件） | `ade829063229f616` |
| `$V/screen/efficiency_s0.candidates.jsonl` | screen、训练 seed 0、五臂的逐候选匹配结果（47,565 行，含 `candidate_id`、`arm`、`parent_id`、`b_dec`、`proxy_status`、`predicted_channel_id`、`hits_best_event`） | `88b04e9fed80e190` |
| `$V/screen/efficiency_s0/streams.jsonl` | 同一批候选的原始流，含每个候选的生成几何 `x`（Å，锚定原子顺序），按 `candidate_id` 连接 | `8693180064f4f15c` |
| `/home/lhshen/xtbflow-runs/explore-geoinfo-20261008/full-68f5733/heads/check_a_rows.jsonl` | dev 62 个母体的 558 条参考记录及各打分器预测，含已算好的 GFN2-xTB 势垒 `xtb_ts`（仓库里同名副本在分支 `explore/geometry-information` 的 `docs/evidence/explore_geoinfo/`） | `34c49edbaba62458` |

要复用的代码（不要复制改写，直接 import）：

- `xtbflow.v1.data.canonical_event(b_r, b_p, perms)`：规范事件 ID，与目录的 `channel_id` 一致。
- `xtbflow.v1.proxy.valid_endpoint(z, b_r, b_p)`：与 V1a 解码完全相同的合法性判断（Lewis 结构校验 + RDKit 可构建，且不等于反应物）。
- `xtbflow.v1.interfaces.query_from_parents(rows, device)` 和 `xtbflow.v1.sampler.subset_query(q, indices)`：构造打分器输入。
- `xtbflow.v1.scores.BarrierHead`、`BarrierEnsemble`，`.components(query, b, x, t_b, t_x)['mean_kcal']`：给出 kcal/mol 势垒。
- `xtbflow.v1.geometry_information`：`event_table`、`ranking_metrics`、`within_parent_rho`、`xtb_energy_kcal(z, x, cfg)`（返回 `(kcal, status)`）。
- `xtbflow.v1.metrics.cluster_summary`。
- T3 另外复用分支 `codex/v1b-20261007` 上的 `xtbflow.v1.qc_protocol`：`harmonic`、`classify_saddle`、`classify_minimum`、`perceive_be`、`endpoint_identity`、常量 `CLEAR_NEGATIVE_CM1 = -30`。

---

## 5. 任务 T1：b2f2 枚举基线

### 5.1 问题

对每个母体，穷举所有小幅键变化得到全部合法产物（事件），用一个便宜的打分器排序，按顺序送去"验证"。问：要找到最低势垒的已知通道，这条流水线需要送验的不同事件数，比 V1a 生成模型更少还是更多？

成本轴取"送去物理验证的不同事件数 k"，因为真实成本由验证（TS 搜索、DFT）主导。生成模型的网络调用成本不计入，这对生成模型有利，所以"枚举更好"的结论是保守的。

### 5.2 步骤

**T1-0 确定枚举域（只用训练集，不看评估母体）。** 对训练集母体的全部目录事件，统计事件大小：断键数 = Σ max(0, −Δb_ij)，成键数 = Σ max(0, Δb_ij)（只算 i<j 的非对角元），对角元改变的原子数，产物中带形式电荷的原子数。从下面的阶梯里选**最小**的、能覆盖训练集 ≥95% 目录事件的那一档，写进配置：

- `b2f2`：断键 ≤2、成键 ≤2；对角元只允许在键变化涉及的原子上改变，每个原子 Δ ∈ {−2, 0, +2}；总电子守恒（Σ 对角元变化 + 2 × Σ_{i<j} Δb_ij = 0）；产物每个原子形式电荷 |q| ≤ 1。
- `b3f3`：同上，断键、成键各 ≤3。先在 5 个训练母体上估计每个母体的合法事件数；均值 > 50,000 时记为不可行，退回 `b2f2`，并报告覆盖率。

**T1-1 实现枚举器（新文件，例如 `src/xtbflow/v1/enumeration.py`）。** 先断键、再只在有剩余价的原子之间成键，以免组合爆炸。最终合法性一律以 `valid_endpoint` 为准；你的预筛只能用 `valid_endpoint` 必然蕴含的条件，否则会漏掉合法产物。按 `canonical_event` 去重。

**单元测试（必须）**：对训练集每个落在所选枚举域内的目录事件，在其母体上运行枚举器，结果必须包含该事件（按 `canonical_event` 比较）。漏掉任何一个都是 bug，不允许存在。

**T1-2 冻结配置 `configs/review/enum_baseline.json` 并推送。** 必须包含：

- 枚举域；
- 打分器：`S_rand` 随机、`S_E` 冻结 h_E、`S_N` 即 N_clean；
- k 网格 {1, 2, 4, 8, 16, 32, 64}；
- AUC 定义：k ∈ {1, 2, 4, 8, 16}，按 log₂k 梯形权重 [1, 2, 2, 2, 1]/8；超过删失上限的点剔除后重新归一化权重；
- 删失规则；
- 比较对象和读法（见 5.3 节）；
- 平局处理：分数相同时按 `channel_id` 的 sha256 排序，保证确定性。

**T1-3 在评估母体上枚举并打分。**

- **主分析**：screen 342 个母体。生成臂数据取自 R 的提取文件。
- **辅分析**：dev 62 个母体，只算枚举侧和覆盖率，作为健全性检查。

记录：每个母体的合法事件数 N_enum；目录通道覆盖率；最佳通道覆盖率（枚举域的天花板）。

打分器的用法：

- 输入用锚定查询（母体 `x_r`、`b_r`）和枚举出的 `b`，`x = x_r`。
- h_E 用 `t_b = 1`；N_clean 用 `t_b = t_x = 1`。
- 批量在 GPU 上算（经 Slurm）。

**T1-4 计算召回曲线。**

- **枚举侧**：recall@k = 1，当且仅当分数前 k 个事件里含最佳通道。N_enum < k 时取全部，不算删失。
- **随机打分不要模拟，用解析期望**：P = 1 − C(N−m, k)/C(N, k)，其中 m 是最佳通道中被枚举到的个数。
- **生成侧**：每个（母体、臂、训练 seed），按完成顺序取前 k 个**不同的非空** `channel`，含最佳通道即命中。流里不同事件少于 k 个时，该格记为删失。
- 先在母体内对未删失的 seed 求平均，再求母体差。某个 k 上任一比较臂的删失格比例 >5% 时，该点不进 AUC，只作描述性报告。
- 预期会有大量删失（只看不同事件个数，不涉及任何命中标签）：V1a 每条流的预算上限是 3,200 单位，screen 上每条流的不同事件数中位数 B0 为 18、A1 为 10、A2 为 6；B0 有约 32% 的流不足 16 个。A1、A2 在 k ≥ 8 时多半会被剔除。这是生成模型受预算限制的真实情况，照规则处理并如实报告即可。

### 5.3 比较与预写读法

- **主比较**：AUC(枚举 + S_N) − AUC(B0)。
  - 双侧下界 > 0 → `ENUMERATION_BEATS_GENERATION`；
  - 上界 < 0 → `GENERATION_BEATS_ENUMERATION`；
  - 否则 `INCONCLUSIVE`。
- **次要比较**（同样报告区间，不设闸门）：枚举 + S_N 对 A1、B1、A2、A0；枚举 + S_E 对 B0；枚举 + S_rand 对 B0（"盲目穷举是否已经够用"）。
- 另报：各 k 的命中率表；N_enum 分布（中位数和 95% 分位）；最佳通道覆盖率；最佳通道不在枚举域内的母体上，生成模型是否命中（这些是枚举做不到、生成做得到的情形，非常重要）。

### 5.4 加载 N_clean 的方法

```python
from xtbflow.v1.scores import BarrierHead, BarrierEnsemble
members = []
for m in (101, 102, 103):
    obj = torch.load(f'{H}/reactant_{m}.pt', map_location=dev, weights_only=False)
    head = BarrierHead('X', **obj['meta']['score_config']).to(dev)
    head.load_state_dict(obj['ema'])
    members.append(head)
ens = BarrierEnsemble(members, obj['meta']['median'], obj['meta']['scale'])
# 用法：ens.components(query, b, x_r, t_b=1, t_x=1)['mean_kcal']
```

### 5.5 必须通过的复现检查（失败就停下来问）

用 R 的提取文件，在"完成候选数 n"轴上重算 B0 的最佳事件命中率（母体内对 3 个 seed 求平均，再对 342 个母体求平均），必须与 R 报告（PR #107 的 `docs/evidence/v1a_r/reanalysis.json`，路径 `analysis_a.completed_candidates.event_best.arm_rates.B0`）逐位一致，|Δ| < 1e-9：

| n | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| B0 | 0.107212 | 0.191033 | 0.309942 | 0.468811 | 0.625731 |

再验证：对每个母体，枚举 + S_rand 的 recall@k 等于第 5.2 节的超几何解析值。

### 5.6 交付

- 证据目录 `docs/evidence/review_enum/`：`REPORT_ZH.md`、`results.json`、manifest、`SHA256SUMS`。
- 一张图：k 对召回率，各臂和各枚举打分器各一条线。
- 预算：GPU ≤2 小时，CPU ≤50 核时。超出预估就停下来问。

---

## 6. 任务 T2：物理预言机基准

### 6.1 问题

在**真实 TS 几何**（T2a）和**模型生成的几何**（T2b）上，哪种便宜的能量方法最会在同一母体内给各通道按势垒排序？

### 6.2 预言机清单与许可规则

| 预言机 | 获取方式 | 规则 |
|---|---|---|
| GFN2-xTB | tblite 0.7.0（`xtbflow` 环境），`xtb_energy_kcal` | 必做，作为复现锚点 |
| GFN1-xTB | tblite，`Calculator('GFN1-xTB', ...)` | 必做 |
| g-xTB | 先查是否有 conda-forge/PyPI 包，或可经 SSH 获取的源码/二进制 | 可得就做，记录版本和二进制哈希；拿不到就记 `UNAVAILABLE` 和原因，继续做其他的 |
| AIMNet2 / AIMNet2-rxn | 先查许可证和获取方式 | 许可证允许、且不需要接受任何条款或登录就做；否则停下来问 |
| UMA、MACE-OMol25 等 OMol25 模型 | HuggingFace 上需要接受许可的门控模型，而且 n2 访问不了 huggingface.co | **停下来问负责人**。接受许可只能由负责人本人操作 |

所有预言机都按中性单重态计算，坐标单位 Å，在 API 边界把能量换成 kcal/mol。

### 6.3 T2a：真实几何（dev，558 条参考记录）

- 输入是 `check_a_rows.jsonl` 的 558 条记录：TS 几何用 `T1xCache.reaction(cache_index)['x_ts']`，原子序数用 `['z']`；锚定反应物用母体目录的 `atomic_numbers`、`x_r`。
- 每种预言机的势垒 = E(TS) − E(锚定反应物)。指标与 PR #110 的检查 A 完全相同：先在通道层面汇总（每个通道恰好一条记录），算同母体 Spearman ρ（主指标）、top-1、Δ ≥ 2 kcal/mol 的一致对比例。另报 MAE、中位有符号误差、每次调用的墙钟时间。
- **复现检查（必须）**：
  - 你重算的 GFN2-xTB 势垒与 `check_a_rows.jsonl` 的 `xtb_ts` 逐条相差 ≤1e-6 kcal/mol；
  - GFN2 的 ρ 必须等于 0.7491015704725383（|Δ| < 1e-9），双侧区间 [0.6524459560799384, 0.8457571848651383]。

### 6.4 T2b：生成几何（screen，B0，训练 seed 0）

- 从 `efficiency_s0.candidates.jsonl` 中取 `arm == 'B0'`、`proxy_status ∈ {PROXY_MATCH, KNOWN_EVENT_GEOMETRY_MISS}` 的候选（事件在目录内）。按 `sha256(candidate_id)` 升序，每个母体最多取 8 个。几何 `x` 从 `efficiency_s0/streams.jsonl` 按 `candidate_id` 取。
- 目标值是该候选事件的通道级目录势垒（该 `channel_id` 所有参考记录里的最小值）。预言机势垒 = E(x_生成) − E(锚定反应物)。
- 用 `within_parent_rho` 算同母体 Spearman（至少 4 个候选、至少 2 个不同目标值），再取母体平均并算聚类区间。

### 6.5 比较与读法

- 每种预言机对 GFN2-xTB 做配对差（T2a、T2b 分别算）：双侧下界 > 0 → `BETTER_THAN_GFN2`；上界 < 0 → `WORSE_THAN_GFN2`；否则 `INCONCLUSIVE`。
- 报告一张"排序质量 × 每次调用成本"表。**不要替项目选定预言机**，只给数据。

### 6.6 交付与预算

- 证据目录 `docs/evidence/review_oracle/`。
- 新环境的 `pip freeze` 和各预言机的许可证摘要放进证据目录。
- 预算：GPU ≤1 小时，CPU ≤20 核时。

---

## 7. 任务 T3：开放世界终点，第一阶段（xTB 层面，只用 CPU）

### 7.1 问题

V1a 的 B0 候选里，被当作"未命中"的目录外事件，有多少在 GFN2-xTB 势能面上确实是连接反应物和所预测产物的一阶鞍点？这些新通道的 xTB 势垒，比该母体已知最佳通道高还是低？

已有的旁证：V1b 的 D1 阶段用 DFT 认证过 2 个目录外候选，一个是 `STRICT_EVENT_MISMATCH`（连通，但事件不同），一个是 `IRC_REACTANT_MISMATCH`。样本太小，不能下结论。

### 7.2 样本（先冻结后计算）

- **来源**：`efficiency_s0.candidates.jsonl`，条件 `arm == 'B0'`、`training_seed == 0`、`decode_status == 'VALID'`；几何从 `efficiency_s0/streams.jsonl` 取。
- **三组**，各自按 `sha256(candidate_id)` 升序、每个母体最多 1 个：
  - `OUTSIDE_CATALOGUE_EVENT` 40 个（主样本）；
  - `PROXY_MATCH` 20 个（阳性对照）；
  - `KNOWN_EVENT_GEOMETRY_MISS` 20 个。
- 对 OUTSIDE 组涉及的每个母体，再加一条**基准链**：从该母体最佳参考记录的 `x_ts`（参考目录里的锚定顺序）出发，事件取该记录的 `b_p`。
- 把样本清单（`candidate_id` 列表和基准 `reference_id` 列表）写进 `configs/review/open_world_o1.json`，在**任何量化计算之前**提交并推送。

### 7.3 xTB 认证链（每个样本一条）

1. **引擎**：geomeTRIC 1.1.1 的自定义引擎，能量和梯度来自 tblite GFN2-xTB（Hartree，Hartree/bohr）。Hessian 用解析梯度的中心差分（步长 0.005 bohr，对称化），单位 Hartree/bohr²。频率分析调用 `qc_protocol.harmonic`。
2. **TS 优化**：从生成的几何出发，`transition=True`，初始 Hessian 用数值 Hessian，收敛阈值用 `qc_protocol.TIGHT`。然后算最终 Hessian，用 `classify_saddle` 分类，阈值 −30 cm⁻¹。
3. **IRC**：从同一个 TS 和它的最终 Hessian 出发，双向，每个方向最多 300 步，trust 0.2（与 `qc_protocol.IRC` 一致）。
4. **端点**：两端各自优化到极小点，再做频率检查（`classify_minimum`）。
5. **身份判定**：直接调用 `qc_protocol.endpoint_identity(z, b_r, perms, b_predicted, x_end1, x_end2)`，不要改写它。`STRICT_JOINT_GRAPH_VALID` 表示一端是反应物、另一端正好是所预测事件的产物。
6. **状态**：每条链只能落在以下一种：`STRICT_JOINT_GRAPH_VALID`、`STRICT_EVENT_MISMATCH`、`IRC_REACTANT_MISMATCH`、`EVALUATION_UNRESOLVED`、TS 阶段失败（`TS_OPT_LIMIT`、`MULTIPLE_NEGATIVE_MODES`、`NO_NEGATIVE_MODE`、`FREQUENCY_GRAY_ZONE`）、`IRC_LIMIT`、端点失败。每条链都写一个 `verdict.json`，含各阶段计时。

**代码位置**：分支基于 `codex/v1b-20261007`，因为 `qc_protocol` 只在那里。新建 `src/xtbflow/v1/xtb_chain.py`，只 import `qc_protocol` 里的纯函数，不修改 `qc_protocol.py`。环境用你新建的 `xtbflow-xtbchain`（tblite + geomeTRIC + PySCF，PySCF 只用于原子质量表）。

### 7.4 指标

- 三组各自的认证率，即 `STRICT_JOINT_GRAPH_VALID` 的比例，附 Wilson 95% 区间。
- **阳性对照闸门**：PROXY_MATCH 组的认证率 < 50% 时，说明这条 xTB 链本身有问题。**停下来报告，不要解读 OUTSIDE 组的结果。**
- 对 OUTSIDE 组中通过认证的链：ΔEa_xTB = E_xTB(新 TS) − E_xTB(同一母体基准链的 TS)。基准链必须先通过认证；没通过的母体不进这一项，并报告数量。报告 ΔEa_xTB < 0 的比例和 ≤ +5 kcal/mol 的比例。
- 每条链的墙钟时间和梯度调用次数。

### 7.5 第二阶段（只写提案，不运行）

基于第一阶段的结果写一页提案：用 V1b 的 DFT 链（ωB97X/6-31G(d)，gpu4pyscf，协议 `v1b-h-gpu4pyscf-3`，D1 实测每条约 0.68 GPU 小时）认证至多 10 个"xTB 层面的新通道"，给出样本规则和算力预估。**交给负责人批准，你不要运行。**

### 7.6 交付与预算

- 证据目录 `docs/evidence/review_open_world/`：报告、`results.json`、每条链的 verdict 汇总表、`SHA256SUMS`、环境 `pip freeze`。
- 预算：CPU ≤50 核时；单个作业 ≤120 分钟；可以按样本分成多个串行作业。

---

## 8. 什么时候必须停下来问负责人

- 任何复现检查不通过：5.5 节、6.3 节的数字，或输入文件哈希与第 4 节不符。
- 任何预估会超出本文的算力预算，或需要 DFT、或需要碰 V1b。
- 需要接受任何软件或模型许可、登录任何网站，或下载 >1 GB 的东西。
- 需要修改本文规定的定义、比较对象、读法或样本规则。
- T3 阳性对照闸门失败。
- 发现与另一个会话（2.6 节）的作业、分支或目录冲突。

问的时候写清：卡在哪里、有哪些选项、各自的代价。

---

## 9. 交付与汇报

每个任务一个 PR，基线分支见第 0 节的表，**不合并**。三个任务都完成后（或在任何停止点），给负责人写一段不超过 12 行的汇总：

- T1：主比较的数值、区间和读法；最佳通道在枚举域内的覆盖率；枚举做不到、生成做到了的母体数。
- T2：每种预言机在 T2a/T2b 上的 ρ，以及相对 GFN2 的读法；哪些预言机因许可或网络没做成。
- T3：三组认证率；OUTSIDE 组通过认证者的 ΔEa_xTB 分布；第二阶段提案的 PR 链接。
- 所有 PR 链接、Slurm 作业号、实际算力用量。

---

## 10. 验收清单（Claude Code 会逐项核对）

**所有任务通用**

1. PR 的基线分支正确、未合并；`git diff --name-status <基线>...<头>` 只有**新增**文件，外加 `bootstrap_inventory.json`。
2. 配置文件的首次提交早于所有结果文件；每个结果 manifest 记录了 `source_commit`（已推送）、`dirty=false`、配置的 SHA-256；n2 上存在对应的 `source-<sha>` 干净克隆。
3. 我会在 n2 上，从你的源码克隆、用你的原始输出重跑分析步骤；得到的 `results.json` 必须与提交的**逐字节相同**。所以分析必须确定性（固定随机种子，排序稳定）。
4. 在 n2（Linux）上通过：`python scripts/validate_bootstrap.py`、`python -m unittest discover -s tests/reusable -p "test_*.py"`、`python -m pytest -q tests --ignore=tests/reusable`。新增的纯函数必须有单元测试。
5. `REPORT_ZH.md` 第一句符合第 3 节第 1 条；报告里的每个数字都能在 `results.json` 里找到；`SHA256SUMS` 与文件一致。
6. 算力在预算内，作业号齐全；没有在 Slurm 之外用 GPU；没有取消别人的作业；没有 DFT。

**T1 专项**

7. 枚举域只依据训练集选定，有覆盖率统计；"训练集目录事件必被重新生成"的测试存在且通过。
8. 5.5 节的 B0 复现数字逐位一致；随机打分与解析期望一致。
9. 删失规则按配置执行，被剔除的点有列表；主比较读法按预写规则给出。

**T2 专项**

10. GFN2 逐条复现（≤1e-6 kcal/mol）和 ρ 复现（<1e-9）通过。
11. 每种预言机都有版本、来源、许可状态；没有接受任何许可条款。

**T3 专项**

12. 样本清单在任何量化计算之前提交；样本规则可以从文件独立重现（我会重抽一遍核对）。
13. `endpoint_identity`、`classify_saddle` 等是 import 自 `qc_protocol`，没有改写；数值 Hessian 有单元测试（对称；对一个已优化的小分子极小点，6 个零频之外全为正频）。
14. 阳性对照闸门按规则执行；ΔEa_xTB 只对"基准链也通过认证"的母体计算。
15. 第二阶段只有提案，没有任何 DFT 作业。

验收不通过的项，我会在 PR 上用行内评论指出，修好后再验。

# xtbflow 交接：生成过程中的物理筛选（SMC 剪枝—克隆，探索性）

2026-10-08 · 交给一个从未接触过本项目的 AI 执行 · 计算在 n2 上 · 完成后由 Claude Code（本文作者）验收

本文自成一体：读完就能开工，不需要读之前的对话。凡是写着"停下来"的地方，必须停下、写明原因并等待负责人，不要自行绕过。负责人外出半天，你可以连续推进耗时任务，但只限本文范围之内。

---

## 0. 一页摘要

**问题。** 在联合生成的中途（检查点 s_c），用 GFN2-xTB 给同一母体的 32 条轨迹的预测终点打分，只保留更有希望的一半，再从保留者"重启克隆"出另一半，生成到终点。这样做，是否比"不干预地生成完，再用同样的 xTB 打分做事后筛选"更容易找到最低势垒的已知通道？

**为什么问它。** 前几轮（PR #110–#112）已经得到：
- 事件定型前，物理信号有一个窗口；
- 几何推动能改写 27–33% 的事件；
- 但"沿力推几何"几乎不偏向低势垒，事件效用只 +0.003 到 +0.008。

推动依赖方向，筛选不依赖方向，只直接比较同一母体的不同通道。"在生成过程中用物理选择"是联合（并联）模型独有、串联模型做不到的能力。所以必须和事后筛选比：事后筛选两种模型都能做。

**结果读法**（第 6 节写死）：
- `IN_GENERATION_SELECTION_HELPS`：比事后筛选好，且比随机剪枝好；
- `NOT_PHYSICS_SPECIFIC`：比事后筛选好，但不比随机剪枝好；
- `NO_GAIN_OVER_POSTHOC`：和事后筛选没有可检出的差别；
- `SELECTION_HURTS`：比事后筛选差。

**范围。**
- 只用 V1a 开发集（62 个母体、21 个分子式组），全部为探索性。
- 不做 DFT，不碰 V1b，不碰 screen 集，不合并任何 PR。

**算力预估。** GPU 约 15–30 分钟（两次短作业），CPU 约 10–20 核时。

**分支。**
- 新建 `explore/smc-selection`，基于 `explore/physics-guidance`（提交 `ffca40f`，PR #112）。
- 开一个 PR，base 设为 `explore/physics-guidance`，**不合并**。

---

## 1. 项目背景（最少必要）

xtbflow 的目标：只给反应物，用流匹配生成模型同时提出"发生什么反应"（事件）和"过渡态（TS）长什么样"（几何），再用便宜的量子化学验证。

- **母体（parent）**：一个反应物分子。中性闭壳层，元素 C/H/N/O，≤23 个原子。
- **事件**：键—电子矩阵（BE 矩阵）的变化，b_p − b_r。
  - 对角元是孤对电子数，非对角元是键级。
  - 规范事件 ID `channel_id` 由 `xtbflow.v1.data.canonical_event(b_r, b_p, perms)` 给出。
- **目录**：数据来自 Transition1x（T1x，ωB97X/6-31G(d)）。
  - 每个母体有若干已知通道，每个通道有目录势垒（kcal/mol）。
  - 最佳通道 = 目录势垒最低的通道，见 `best_channel_ids`。
- **联合模型**：同时生成 b（事件变量）和 x（几何，Å），各有一个时钟。
  - 时钟 `geometry_lead3`：t_b = s³，t_x = s，共 50 步，s = k/50。
  - 其中 b 晚于 x 定型，所以几何先成熟。
- **预测终点**：x̂₁ = x_t + (1 − t_x)·v_x，b̂₁ = b_t + (1 − t_b)·v_b。v 是网络在当前状态给出的速度。
- **事件效用**（V1a 代理）：
  - 已知事件为 max(0, 1 − (Ea_事件 − Ea_最佳)/10)；
  - 目录外事件或非法事件为 0。

已确立的相关事实（PR 均未合并）：

| PR | 结论 |
|---|---|
| #110 | GFN2-xTB 在真实 TS 上的同母体排序 ρ = 0.749；在预测终点上，s ≈ 0.7 起才有信息 |
| #111 | geometry_lead3 下，事件定型中位 s = 0.62，xTB 窗口占 34.6% |
| #112 | 鞍点感知推动改写 12–27% 事件，但效用增益 +0.003 [+0.000, +0.006] / +0.008 [−0.001, +0.017]；朴素下降让合法率掉 11 个百分点 |

---

## 2. 环境与"不影响其他任务"（硬性规则）

### 2.1 机器与仓库

- **仓库**：`git@github.com:askusk521-cyber/xtbflow.git`，默认分支 `main`。
  - 你只开 PR，不合并、不开自动合并、不直推 main。
  - 开工前读仓库根目录的 `AGENTS.md`。
- **n2**：用户 `lhshen`，192 核，4×RTX PRO 6000，Slurm 21.08。单节点，登录 shell 就在计算节点上。

### 2.2 GPU 与 CPU 的使用方式

- **GPU：必须经 Slurm**（`srun`/`sbatch`），在登录 shell 上不许碰 GPU。
  - 在登录 shell 跑任何 Python 时，都设置 `CUDA_VISIBLE_DEVICES=`，防止误用 GPU。
- **CPU：不走 Slurm，直接在登录 shell 上跑**，用 `nohup ... &` 或 `setsid` 挂后台，日志写到运行目录。
- **CPU 用量上限（同机还有另一个 AI 在跑任务）：**
  - 每个进程池 `--workers` ≤ 48，所有命令前加 `nice -n 10`；
  - 启动重任务前先看 `uptime`：1 分钟负载 > 120 时，等它降下来再启动；
  - 同一时刻你自己最多只跑一个重 CPU 任务。
- **Slurm 规则：**
  - 分区 `main`，单卡 `--gres=gpu:pro6000:1`；
  - 不要手动设 `CUDA_VISIBLE_DEVICES`，Slurm 会分配；
  - 项目规定同一时刻只能有 1 个 Slurm 作业。提交前先 `squeue -u lhshen`；已有作业（包括别人的）就加 `--dependency=afterany:<jobid>` 排队；
  - **绝不 `scancel` 不是你提交的作业**；
  - 单个作业墙钟 ≤ 60 分钟，作业名以 `smc-` 开头；
  - GPU 作业里可以用分配给它的 CPU（`--cpus-per-task=48`）。
- **`sbatch --wrap` 用的是 `/bin/sh`**，没有 `source`/`conda activate`，直接用解释器的绝对路径：

```bash
sbatch --parsable --partition=main --gres=gpu:pro6000:1 --cpus-per-task=48 --mem=64G --time=00:60:00 \
  --job-name=smc-g1 --output=$OUT/g1-%j.log \
  --wrap "cd $S; export PYTHONPATH=src:vendor/mechai_reusable OMP_NUM_THREADS=1 PATH=/home/lhshen/miniconda3/envs/xtbflow/bin:\$PATH; python scripts/explore_smc.py g1 ..."
```

### 2.3 不能动的东西

- **另一个 AI 的任务**：目录 `/home/lhshen/xtbflow-runs/review-align-20261009/`，分支 `review/*`，以及它的进程和作业。只读，连 `ls` 以外的操作都不要做。
- `/home/lhshen/xtbflow/` 下的主 checkout 和所有 worktree：不切分支、不重置、不 stash、不改文件。
- `/home/lhshen/xtbflow-runs/` 下已有的运行目录（`v1a-20261007`、`explore-geoinfo-20261008` 等）：只读。
- conda 环境 `xtbflow`、`xtbflow-qc`、`xtbflow-cp2k`：不往里装任何包。本任务不需要新包。
- 仓库里已有的文件：**只新增，不修改**，`bootstrap_inventory.json` 除外。
  - 不改 `scripts/explore_geometry_information.py`、`src/xtbflow/v1/explore_rollout.py` 等；
  - 需要它们的函数时 import（脚本可以 `sys.path.insert` 后 import，但不编辑）。

### 2.4 你的目录与代码同步

- 运行根目录：`R=/home/lhshen/xtbflow-runs/explore-smc-20261009`，只在这里写东西。
- **开发**：在 `$R/dev-clone` 里开发。

```bash
mkdir -p $R && cd $R
git clone -q --no-checkout /home/lhshen/xtbflow-runs/explore-geoinfo-20261008/source-ffca40f dev-clone
cd dev-clone && git fetch -q git@github.com:askusk521-cyber/xtbflow.git explore/physics-guidance
git checkout -q -b explore/smc-selection FETCH_HEAD
git remote set-url origin git@github.com:askusk521-cyber/xtbflow.git
```

- **正式运行**：一律从已推送提交的干净克隆跑，目录 `$R/source-<短SHA>`，方法同上但用 `--detach <SHA>`。`git status --porcelain` 必须为空。
- n2 的 git 身份已配置；GitHub 走 SSH，HTTPS 不稳定。
- **环境**：`/home/lhshen/miniconda3/envs/xtbflow/bin/python`（PyTorch、tblite 0.7.0、RDKit 都在），`PYTHONPATH=src:vendor/mechai_reusable`。
- **提交规范**：
  - subject 用 `feat:`/`fix:`/`test:`/`docs:` 前缀，祈使句；
  - 新增文件后运行 `python scripts/validate_bootstrap.py --refresh`，`bootstrap_inventory.json` 放进同一个提交；
  - commit 末尾加四行 trailer：`Coding-Agent`、`<agent>-Version`、`Model`、`Reasoning-Effort`，填你的真实值，查不到写 `unspecified`；
  - PR 描述 = 英文说明 + `中文说明` 段（至少三句完整中文），末尾四行署名；
  - 不提交 > 1 MB 的数据文件，大文件留在 n2，把 SHA-256 写进 `SHA256SUMS`。

---

## 3. 科研纪律（硬性规则）

1. **全部探索性。** 报告第一句写"本报告全部结果为探索性"，并给出主结果的数值和区间。
2. **先冻结、后看数据。**
   - 先提交并推送配置 `configs/explore/smc_selection.json`（版本 `smc-1`），把第 5–6 节的全部定义写进去，再跑任何正式计算；
   - 冒烟测试只检查流程能否跑通，不看指标；
   - 结果出来后若要改，开 `smc-2`，两版都报告。
3. **不调参。** R = 16、检查点网格、打分器参数、K 网格全部按本文执行，不按结果挑选。
4. **区间方法。** 指标先在（母体, 训练 seed）层面算，再对 3 个 seed 求平均得到母体值；配对差按母体算，再用 `xtbflow.v1.metrics.cluster_summary(values, formula_of)`（分组键是母体的 `split_group`）给双侧 95% 区间。
5. **不声称化学结论。** 代理命中不等于认证过的 TS。

---

## 4. 输入（n2，只读）

- V1a 根目录 `V=/home/lhshen/xtbflow-runs/v1a-20261007`。
  - 用 `scripts/explore_geometry_information.py` 的 `load_inputs(cfg)` 读开发集：配置传 `configs/explore/geometry_information.json`，它会校验划分哈希。
  - 查询行在 `$V/data/development_queries.jsonl`。
- **生成器**：`xtbflow.v1.assets.load_generator(V, 'joint', seed, device)`，seed ∈ {0, 1, 2}。返回 `(net, meta)`，meta 里有 `sigma_b`（1.0）、`sigma_x`（0.5）、`dual_time`。
- **pg-1 的不推臂轨迹**（复现基准）：`/home/lhshen/xtbflow-runs/explore-geoinfo-20261008/pg-1efad90/none-geometry_lead3/check_b_trajectories.jsonl`。
  - 5,952 行，sha256 `b842f4c418f8cbd30f60d8090100a9855334ef7f59b2fc76d0095b3bf3a39b62`；
  - 每行有 `parent_id`、`seed`、`proposal`、`events`（51 个逐步解码事件，最后一个是终点事件）、`proxy_status`、`event_utility`、`xtb_delta_kcal`（网格 t = 0, 0.1, …, 1.0 上 x̂₁ 的 xTB 能量减锚定反应物能量）。
- **检查 A 的 558 条参考记录**：分支 `explore/physics-guidance` 的 `docs/evidence/explore_geoinfo/check_a_rows.jsonl`。
  - 字段 `parent_id`、`formula`、`channel_id`、`cache_index`、`catalog_barrier_kcal`、`xtb_ts`；
  - TS 几何用 `xtbflow.m0.t1x_data.T1xCache('/home/lhshen/data/t1x/t1x_m0_v1.npz').reaction(cache_index)`。注意它是原始原子顺序；锚定顺序的 `x_ts`、`b_p` 在 `$V/data/reference_catalog.jsonl`，按 `reference_id` 连接，优先用这个。

**要复用的函数**（import，不复制改写）：

| 函数 | 用途 |
|---|---|
| `xtbflow.v1.sampler.initial_state(q, proposals, seed, sampling_seed=0, namespace='explore_time_window', sigma_b, sigma_x, group='joint')` | 初始噪声，与 pg-1 完全相同 |
| `xtbflow.v1.explore_rollout.recorded_rollout(net, q, st, path='geometry_lead3', n_steps=50, dual_time=meta['dual_time'])` | 不推的完整轨迹，返回逐步 `b_hat`/`x_hat`/`b_trace`/`x_trace` |
| `xtbflow.v1.sampler.rollout(net, q, state, role='joint', path='geometry_lead3', n_steps=50, start=k_c)` | 从第 k_c 步接着生成 |
| `xtbflow.v1.explore_rollout.event_direction(b_hat, b_r, x, mask)` | 由预测键级变化得到的反应坐标 u（单位向量） |
| `xtbflow.v1.explore_rollout.saddle_reflect(F, u)` | (I − 2uuᵀ)F |
| `xtbflow.m0.flow.symmetric_noise` / `geometry_noise` | 克隆用的新噪声 |
| `xtbflow.v1.guidance.centered` | 去质心 |
| `xtbflow.m0.sampler.decode_be(b, element_index, n)` → `valid_endpoint` → `canonical_event` → `CatalogueMatcher(parent, refs).match(bp, x, ...)` | 终点解码与目录匹配，写法照抄 `explore_geometry_information._trajectory_job` 的逻辑 |
| `xtbflow.v1.geometry_information.xtb_energy_kcal(z, x, cfg)`、`min_distance` | xTB 设置取 `geometry_information.json` 的 `xtb` 块 |
| `explore_geometry_information._force_job((z, x, cfg))` | xTB 梯度，单位 Hartree/bohr |
| `xtbflow.v1.clocks.clock_grid`、`observation_index` | 时钟与检查点步号 |
| `xtbflow.v1.metrics.cluster_summary`、`geometry_information.within_parent_rho` | 区间与同母体相关 |

---

## 5. 设计（写进 `smc-1` 配置）

### 5.1 种群与检查点

- 每个（母体, 训练 seed）一个种群：P = 32 条轨迹，proposal j = 0..31，初始噪声与 pg-1 相同。
- 时钟 `geometry_lead3`，50 步。
- **批次布局**：批大小 256，按（母体, proposal）顺序切块，与 pg-1 的 `cmd_rollouts` 完全一样。一个种群永远不跨批次。
- **检查点**：s_c ∈ {0.6, 0.7, 0.8, 0.9}，步号 k_c = `observation_index(tx, s_c)` = 30, 35, 40, 45。对应 t_b = 0.216, 0.343, 0.512, 0.729。

### 5.2 打分器（输入只用连续预测，不需要解码）

对候选几何 x（Å）和连续事件预测 b̂，锚定能量 E_R 取 `xtb_energy_kcal(母体 x_r)`。

- **`raw`**：E_xTB(x) − E_R。
- **`relaxed`（主打分器）**：
  1. 从 x 出发做 20 步鞍点弛豫：每步算 xTB 力 F（换算成 kcal/mol/Å），在当前 x 上重算 u = event_direction(b̂, b_r, x)，令 G = (I − 2uuᵀ)F，Δx = 0.001 Å²/(kcal/mol) × G；任一原子位移超过 0.05 Å 时，整步按比例缩小；去质心。
  2. 弛豫后算能量 E_rel − E_R。
  3. 再算沿 u 的曲率 c_u = −[F(x + hu) − F(x − hu)]·u / (2h)，h = 0.005 Å。
  - 每个候选约 23 次 xTB 调用。
  - 任何一步出现原子间距 < 0.5 Å、SCF 失败或 u 无效，就记失败。
- **分层排序**：
  - 第 0 层：成功且 c_u < 0；
  - 第 1 层：成功但 c_u ≥ 0；
  - 第 2 层：失败。
  - `relaxed` 按（层，弛豫势垒）升序排；`raw` 只分成功/失败两层，按 raw 势垒排；
  - 平局按 `sha256(f"{parent_id}|{seed}|{proposal}")` 排。
- **预写的分支（S0 决定，不看 SMC 结果）**：如果 S0 中真实 TS 上 c_u < 0 的比例 < 50%，`relaxed` 去掉曲率分层，只按弛豫势垒排，并在报告里写明。

### 5.3 剪枝—克隆（每个种群，在第 k_c 步）

1. 用打分器给 32 条轨迹在第 k_c 步的预测终点（x̂₁, b̂₁）打分并排序，保留前 R = 16 条（幸存者），其余 16 条在 k_c 停止。
2. 每个幸存者 a 生一个克隆，克隆从同一时刻的条件分布重新加噪：
   - b′ = (1 − t_b)·(b_r + σ_b·ε_b) + t_b·b̂₁⁽ᵃ⁾
   - x′ = centered((1 − t_x)·(x_r + σ_x·ε_x) + t_x·x̂₁⁽ᵃ⁾)
   - t_b、t_x 取第 k_c 步的值；ε 只填前 n 个原子，写法照抄 `initial_state`；
   - ε 的生成器种子用 `addressed_seed('explore_smc_clone', noise_role='joint_b'（或 'joint_x'）, query=qid, training_seed=seed, proposal=a 的 proposal, checkpoint_step=k_c)`。不依赖臂名，所以不同臂里同一幸存者的克隆噪声相同，可以配对。
3. 幸存者从自己第 k_c 步的原状态（b_trace[k_c], x_trace[k_c]）继续；克隆从 (b′, x′) 继续。都用 `sampler.rollout(start=k_c)` 生成到第 50 步。
   - 批次布局与 G1 相同：幸存者在原行，克隆占被剪掉的行，按秩次顺序填。
4. **随机对照 `random`**：幸存者改为按 `sha256(f"{parent_id}|{seed}|{proposal}|{k_c}")` 取前 16，其余相同。

### 5.4 臂（共 13 个）

| 臂 | 含义 |
|---|---|
| `none` | 不干预生成完（= 事后筛选基线） |
| `relaxed_{0.6,0.7,0.8,0.9}` | 用 relaxed 剪枝—克隆；**`relaxed_0.7` 是主臂**（0.7 是 PR #110 测得 xTB 开始有信息的时刻，已披露为开发集选择） |
| `raw_{0.6,0.7,0.8,0.9}` | 用 raw 剪枝—克隆（消融：弛豫有没有用） |
| `random_{0.6,0.7,0.8,0.9}` | 随机剪枝—克隆（对照：重启本身的效果） |

每个臂的网络调用次数相同：16 条在 k_c 停止，16 个克隆补上剩下的步数。另外每个检查点多一次前向。检查点 xTB 调用次数如实报告。

### 5.5 终点评估（所有臂相同）

对每个终点候选：
1. 解码终点 b，得到 `canonical_event`、`proxy_status`、`hits_best_event`、`event_utility`；
2. 用 relaxed 打分器给终点打分（x = 终点 x，b̂ = 终点 b）。

每个种群（32 个终点）：
- 只取合法事件，事件分数 = 该事件所有候选里最好的（层，弛豫势垒），平局按 `sha256(channel_id)`；
- 按事件分数给不同事件排序；
- **hit@K** = 前 K 个不同事件里含最佳通道，K ∈ {1, 2, 4}。

**主指标** H = mean(hit@1, hit@2, hit@4)。

**次要指标：**
- hit@∞（种群里有没有最佳通道，不依赖排序）；
- 种群平均事件效用（与 pg-1 可比）；
- 不同合法事件数；
- 合法率；
- best-reference 命中率。

**诊断：**
- `none` 臂：各检查点打分与终点事件目录势垒的同母体 Spearman，用 `within_parent_rho`（≥ 4 条、≥ 2 个不同目标），raw 和 relaxed 各算一遍；
- 误剪率：none 种群里有最佳通道、但幸存者终点都不含它的比例；
- 克隆的终点事件与其幸存者不同的比例；
- 最佳通道只出现在克隆中的种群比例；
- 打分器统计：分层计数、失败数、弛豫降能分布、每候选调用数与耗时。

---

## 6. 比较与读法（运行前写死）

**比较**（按母体配对，聚类 t 区间）：每个 `relaxed_s` 和 `raw_s` 对 `none` 和对同一 s 的 `random_s`；每个 `relaxed_s` 对 `raw_s`；每个 `random_s` 对 `none`。

主指标 H 的读法（逐臂给出，主结论看 `relaxed_0.7`）：

| 读法 | 条件 |
|---|---|
| `IN_GENERATION_SELECTION_HELPS` | 对 none 下界 > 0，且对 random 下界 > 0 |
| `NOT_PHYSICS_SPECIFIC` | 对 none 下界 > 0，对 random 区间触及 0 |
| `NO_GAIN_OVER_POSTHOC` | 对 none 区间含 0 |
| `SELECTION_HURTS` | 对 none 上界 < 0 |

---

## 7. 执行步骤

| 步 | 内容 | 在哪跑 |
|---|---|---|
| T | 新文件：`src/xtbflow/v1/smc.py`（排序、剪枝—克隆索引、重启加噪、hit@K）、`src/xtbflow/v1/xtb_score.py`（raw/relaxed 打分器）、`scripts/explore_smc.py`（子命令 s0/g1/c1/g2/c2/analyze）、`tests/test_smc.py`；n2 上全量测试通过 | CPU |
| F | 写配置 `configs/explore/smc_selection.json`（smc-1）和设计文档 `docs/explore/SMC_SELECTION_DESIGN_ZH.md`，**提交并推送**；然后用 `--limit-parents 2 --proposals 32 --seeds 0` 在 CPU 上冒烟（不看指标） | CPU |
| S0 | 打分器健全性：在 558 条真实 TS 上算 raw 和 relaxed（u 用参考记录的 b_p − b_r）。**raw 必须复现** `xtb_ts`（逐条 ≤ 1e-6 kcal/mol），以及检查 A 的 ρ = 0.7491015704725383（|Δ| < 1e-9），算法用 `analyze_a` 的 `min` 汇总。报告 relaxed 的 ρ、c_u < 0 比例和弛豫降能 | CPU |
| G1 | `none` 全量：3 seed × 62 × 32，用 `recorded_rollout`；保存每条轨迹在 k ∈ {30, 35, 40, 45, 50} 的 b_trace/x_trace 和 b_hat/x_hat（存 .pt，留 n2） | GPU（Slurm） |
| C1 | 解码 none 终点并与 pg-1 对照（见下方复现检查）；四个检查点上的 raw/relaxed 打分（5,952 × 4）；生成 12 个臂的续跑计划（幸存者、克隆初始状态） | CPU |
| G2 | 12 个臂从 k_c 续跑到 50，批次布局同 G1 | GPU（Slurm） |
| C2 | 所有臂终点解码、匹配和 relaxed 打分；survivor 复现检查；analyze 生成 `results.json` | CPU |
| D | 证据目录 `docs/evidence/explore_smc/`（`REPORT_ZH.md`、`results.json`、各阶段 manifest、`SHA256SUMS`），开 PR | — |

**复现检查（任一不过就停下）：**
1. C1：none 终点事件和 `proxy_status` 与 pg-1 none 行 100% 一致；第 35 步的 raw 打分与 pg-1 `xtb_delta_kcal` 在 t = 0.7 处逐条相差 ≤ 1e-6 kcal/mol（t = 0.6/0.8/0.9 同样适用）。
2. C2：每个臂幸存者的终点事件与 G1 同一 proposal 的终点事件一致率 ≥ 99.9%，并报告最大 |Δx|。低于该值说明续跑有 bug。
3. S0：两项复现数字。

所有 manifest 记录 `source_commit`（已推送）、`dirty=false`、配置 sha256、输入文件 sha256 和 Slurm 作业号。分析必须确定性：我会从你的原始输出重跑 analyze，结果要逐字节相同。

---

## 8. 什么时候停下来

- 任何复现检查不通过，或输入文件哈希与第 4 节不符；
- 预估 GPU > 1 小时或 CPU > 40 核时；
- 需要修改已有文件、第 5–6 节的定义，或需要新装包；
- 与另一个 AI 的作业、目录或负载冲突：例如负载持续 > 120，或你的作业排队超过 2 小时；
- 想用 screen 集、做 DFT、碰 V1b。

停下时写清：卡在哪里、有哪些选项、各自的代价。写在 PR 描述或 `$R/STATUS.md` 里，负责人回来会看。

---

## 9. 交付与汇报

一个 PR：`explore/smc-selection` → base `explore/physics-guidance`，不合并。完成后写不超过 10 行的汇总：
- `relaxed_0.7` 的 H 对 none、对 random 的差值、区间和读法；
- 四个检查点的剂量—响应；relaxed 对 raw；
- 误剪率与克隆发现率；
- 复现检查结果；
- PR 链接、作业号、实际算力。

---

## 10. 验收清单（Claude Code 会逐项核对）

1. PR 的 base 正确、未合并；`git diff --name-status explore/physics-guidance...HEAD` 只有新增文件，外加 `bootstrap_inventory.json`。
2. 配置 `smc-1` 的首次推送早于所有正式结果；冒烟运行没有输出指标。
3. 第 7 节三项复现检查全部通过，数字写在 `results.json`。
4. 我从 `source-<sha>` 和你的原始输出重跑 analyze，`results.json` 逐字节相同。
5. 新增纯函数有单元测试，至少包括：
   - 剪枝—克隆索引（16 个幸存者、每个恰好一个克隆、平局确定）；
   - 重启加噪在 t = 1 时等于终点、在 t = 0 时等于初始分布的形式；
   - hit@K 在手工构造的种群上正确；
   - relaxed 打分器在一个小分子上的单位与符号（力 = −梯度，换算因子正确）。
   在 n2 上 `python scripts/validate_bootstrap.py`、`python -m unittest discover -s tests/reusable -p "test_*.py"`、`python -m pytest -q tests --ignore=tests/reusable` 全部通过。
6. 所有 GPU 计算都有 Slurm 作业号；没有在登录 shell 上用 GPU；CPU worker ≤ 48 且用了 nice；没有动另一个 AI 的目录、作业和分支；没有 DFT。
7. `REPORT_ZH.md` 第一句符合第 3 节第 1 条；每个数字都能在 `results.json` 里找到；读法按第 6 节给出；偏差与限制写全，至少包括：
   - 开发集已多次使用；
   - 检查点 0.7 和时钟 lead3 都用过开发集结果；
   - 闭世界指标会把目录外通道算作未命中；
   - xTB 与 ωB97X 的势垒排序不一致。

# xtbflow 项目开发规范

这份文件是 `xtbflow` 仓库的项目级工作规范。进入仓库后的源码修改、测试、
数据审计、n2 作业和 PR 描述都按这里执行；若它与用户当前任务冲突，以用户的
明确要求为准。历史文档和 `archive/` 只提供背景，不能放宽这里的边界。

## 工作目录和仓库边界

- Codex 的项目工作目录固定为 `/home/lhshen/xtbflow`。执行命令时显式使用这个
  目录作为 `workdir`，不要在 `/home/lhshen` 根目录直接修改项目文件。
- 开始每项工作先检查 `pwd`、当前分支、工作树、worktree、远端和相关 PR：

  ```bash
  cd /home/lhshen/xtbflow
  pwd
  git status --short --branch
  git worktree list
  git remote -v
  ```

- 一个工作线使用一个分支；堆叠 PR 必须明确记录 base 分支和依赖顺序。不要把
  一个工作线的合同、科学数据、训练结果和总进度文档混在同一提交中。
- 不覆盖别人的未提交修改，不在 detached worktree 上直接开发。需要在 n2 运行
  时创建按提交或分支命名的独立执行目录。
- `vendor/mechai_reusable/` 是只读快照，不在原地修改；`archive/` 是历史材料，
  不作为当前实现入口。

## 本地环境和 n2 环境

### 本地开发机

- 本地源码目录是 `/home/lhshen/xtbflow`，适合阅读源码、编辑合同和文档、运行
  Git/静态检查以及准备可复现命令。
- 本地 Python、PyTorch、xTB、tblite、CP2K、Slurm 和 GPU 不视为已安装或已认证。
  当前本地 shell 可能没有 `python`；不要把本地成功的 mock、语法检查或单元测试
  写成真实训练、量化计算或 GPU 结果。
- 本地可做的检查包括 `git diff --check`、JSON/YAML 语法、源码审阅和不依赖科学
  后端的静态合同检查。完整 CPU/torch 测试优先在 CI 或 n2 的 `xtbflow` 环境执行。

### n2 真实执行环境

- SSH 别名为 `n2`，主机名为 `node2`。所有真实公开数据下载、数据解析、训练、
  GFN2/xTB、CP2K、GPU 基准和物理验证都必须通过 `ssh n2` 执行。
- Conda 根目录为 `~/miniconda3`；常用环境为 `xtbflow`。执行 Python 前显式加载：

  ```bash
  source ~/miniconda3/etc/profile.d/conda.sh
  conda activate xtbflow
  ```

- 历史参考基线（不是所有运行的统一环境）是 Python 3.10.19、PyTorch
  2.10.0+cu128、NumPy 1.26.4、SciPy 1.15.3、RDKit 2024.03.3、ASE 3.29.0、
  tblite 0.7.0 和 xTB 分发包 22.1（导入模块版本可能显示 20.2）。每次重要作业
  必须以证据文件记录实际 `python --version`、包版本、可执行文件和
  `git rev-parse HEAD`；不得用这组历史参考值替代实际运行环境。
- Slurm 为 21.08.5，分区是 `main`；GPU GRES 是 `gpu:pro6000:4`。GPU 作业必须
  申请有限时长和明确 GPU 数，例如：

  ```bash
  srun --partition=main --gres=gpu:pro6000:1 --nodes=1 --ntasks=1 \
    --time=00:05:00 bash -lc 'source ~/miniconda3/etc/profile.d/conda.sh; \
    conda activate xtbflow; python -c "import torch; print(torch.cuda.get_device_name(0))"'
  ```

- CP2K 使用独立的 `xtbflow-cp2k` 环境；没有在作业中明确激活并记录可执行文件、
  版本和协议哈希时，不得声称 CP2K 结果。当前 n2 的默认 shell 不保证 CP2K 在
  `PATH` 中可用。
- n2 原始数据、缓存、模型权重和私有路径不得提交到仓库。Reaction-QM/RGD1 原始
  字节保留在 n2 缓存，仓库只提交来源 URL、大小、MD5/SHA-256、字段审计和摘要。
- `~/xtbflow/xtbflow-exec-*` 可能是历史或 detached 执行目录，含有未提交修改时
  绝不覆盖。新作业应从目标提交创建新的干净目录，或明确记录现有目录的状态。

开始 n2 工作前至少检查：

```bash
ssh n2 'hostname; source ~/miniconda3/etc/profile.d/conda.sh; conda activate xtbflow; \
  python --version; python -c "import torch; print(torch.__version__)"; \
  command -v sbatch; command -v srun; git --version; squeue -u "$USER"'
```

## 工具和文档使用规范

- 文件搜索优先使用 `rg`/`rg --files`；命令执行通过项目目录的工具调用完成，避免
  在工具参数里隐式切换到其他目录。并行读取只用于彼此独立的只读检查，编辑、提交、
  推送和依赖前一步输出的动作按顺序执行。
- 修改前先阅读目标模块、相关测试、`configs/resources.yaml`、数据合同和对应的
  `docs/` 文档。新增入口必须写明输入、输出、单位、电子态、失败状态和资源上限。
- 文档中的命令必须能在记录的环境中重放。每条 n2/Slurm 命令都应包含工作目录、
  分支或 commit、环境激活、时间/调用/磁盘预算和输出位置；不要只写“已在 n2 运行”。
- 每个真实阶段产生独立证据文件，优先放在 `docs/evidence/`，文件名带日期或唯一
  运行标识。不得覆盖历史证据；失败报告同样保留。报告至少包含执行主机、源码 SHA、
  配置/协议哈希、来源资产哈希、环境版本、资源消耗、状态、claim limits 和未完成项。
- `docs/ENVIRONMENT_N2.md` 记录主机观察结果，`docs/CP2K_SETUP.md` 记录 CP2K，
  `docs/data_contract.md` 和 `docs/architecture/` 记录公共合同；如果实现改变了
  这些边界，先同步文档和测试再提交代码。
- 工具输出的事实和科学结论分开写。软件 smoke/fixture 只能证明可运行性，不能推出
  化学准确性、架构领先或泛化能力。模型选择必须等待冻结协议、独立划分和物理验证。

## 数据、合同和科学边界

- 缺失 charge、multiplicity、单位、坐标映射、母反应/反应族分组、来源许可证或
  overlap audit 时使用明确的 `unknown`/quarantine 状态，不猜测、不默认中性单重态、
  不用元素序列代替 atom-map 证据。
- `ReactionQMRecord` 只负责来源解析和审计，必须保持 `quarantine`；不得在该对象
  上新增 development/confirmatory 旁路。统一 Track-B 适配器通过完整来源、环境、
  协议、输入隔离和重叠审计后才能授予准入。
- 训练输入不能含 product、TS、事件标签、sealed test label 或从参考结果选择的环境。
  测试标签不得反馈训练、模型选择或阈值选择。真实训练与物理验证使用独立预算。
- 失败必须按真实阶段记录：未执行、执行后计算失败、结算失败、报告写出失败和预算耗尽
  不得合并成一个 `not_run`。账本必须保留已经消耗的调用、时间和资源。
- 没有可信反应族或独立来源标签时，只能限制 claim；不要把较弱开发数据宣传为
  confirmatory 或反应族外泛化。

## 测试、提交和 PR

项目结构和测试入口：

- `src/xtbflow/`：可安装包；`tests/`：pytest 合同测试；`tests/reusable/`：
  标准库/PyTorch 可复用测试；`configs/`：协议、资源和清单；`scripts/`：有界入口。
- Python 3.10+、四空格、类型注解和小而明确的函数；单位、charge、multiplicity 在
  API 边界显式表达。测试名使用 `test_*.py` 和 `test_*`。
- 目标环境可用时运行：

  ```bash
  PYTHONPATH=src:vendor/mechai_reusable pytest -q
  PYTHONPATH=src:vendor/mechai_reusable \
    python -m unittest discover -s tests/reusable -p 'test_*.py'
  python scripts/validate_bootstrap.py
  ```

- 有意修改受清单保护的文件后，在同一提交用 `python scripts/validate_bootstrap.py --refresh`
  更新 `bootstrap_inventory.json`，然后重新运行验证。不要手工伪造哈希。
- 提交标题使用 `feat:`、`fix:`、`test:`、`docs:` 或 `chore:`；提交保持单一目的。
  PR 必须说明改动、依赖/目标 issue、测试命令、n2 命令和 commit、数据/配置哈希、
  真实结果、claim limits 与未完成项。堆叠 PR 还要明确 base/head 关系。
- 不自动合并 PR，不强推覆盖他人分支，不提交凭据、私有数据、模型权重或无界作业配置。
  推送和评论可以按项目授权执行，但合并前保留人工审查和当前 main 的 CI 证据。

## 开始和结束检查清单

开始前：确认当前目录是 `/home/lhshen/xtbflow`，确认分支/工作树和 PR 依赖，确认
任务是否需要 n2，确认资源上限和输出文件名不会覆盖历史证据。

结束前：运行与改动匹配的测试，执行 `git diff --check`，检查 bootstrap 清单，记录
n2 主机/环境/commit/哈希/资源/失败状态，更新必要文档，再提交并推送独立分支。任何
未完成项、未验证假设和科学 claim 限制都必须留在 PR 和证据中。

## 研究路线执行合同（#129）

XTBFlow 的联合研究路线使用四个明确角色：J 是事件与几何联合内核，P 是生成过程中的物理反馈，C 是有限预算下的计算决策 Agent，V 是统一的 TS、模式、双向 IRC 和实际端点认证服务。代码可以分层交付，但每个主张必须说明它属于哪个角色以及使用了哪些其他角色。

研究证据按四级记录：`software_available` 只证明代码和合同可运行；`real_closed_loop` 要求真实反应物、物理调用、候选、认证和费用能够重放；`development_comparison` 要求冻结输入、对照和总预算后得到可复核比较；`independent_confirmation` 要求未参与选择的独立母组和冻结协议支持主张。较低等级不得替代较高等级。

主指标是固定总预算内、按独立母体系统计的 `unique strict certified channels`。至少一条通道的母体系比例、首次认证成本、失败/重试/重复/删失和未知成本都必须同时报告。事件改变率、物理窗口、候选数量和模型损失是机制诊断，不能替代严格认证通道增量。

每轮研究只改变一个主要因素。开始前冻结 source、模型/权重、数据划分、输入允许范围、物理协议、随机设置、总预算、重试规则、停止规则、对照方法和主要指标；结束时保存 source/config/protocol/input 哈希、实际账本、失败状态、claim limits 和下一步交接。新增模块须记录“观察到的瓶颈、已有证据、最小替代方案、有限预算、关闭方式和采用条件”。

下一轮默认顺序是：先完成一条真实的 reactant-only → 计量物理反馈 → 有界动作/重新提议 → V 认证 → 实际端点结算闭环；再在同一生成器、工具集合、输入面板和总预算下比较联合生成、允许回溯的强串联、末端物理处理和生成内物理反馈；随后才比较固定调度、简单启发式和反馈驱动 Agent。部分条件补全、校正势和多步网络只有在上述比较指出明确瓶颈后才进入主线。

真实结果必须与软件 fixture、缓存重放和配置声明分开。任何“pipeline completed”报告仍须逐候选核对 TS、模式、双向 IRC、实际端点、重复通道和费用结算；没有这些证据时使用 `unknown`、`attempted_unresolved` 或 `pending_certification`，不得把候选或排队作业写成新化学通道。所有公开 PR 继续遵守人工合并审查规则。

### 科学分类与比较边界

- `PROTOCOL_FAILURE` 且 `evidence_status=attempted_unresolved` 统一归类为
  `failure_classification=certification_unresolved`，不是化学负例。它计入
  attempted count、失败分母和实际 calculator/CPU/GPU/wall/storage 账本，严格
  通道增量为零；若公开证据无法定位失败阶段，`failure_stage` 和
  `failure_reason` 必须为 `unknown`，不能把 `pipeline_completed` 写成认证成功。
  公开结构化结果同时保留 `chemical_negative=false`、
  `diagnostic_retry_allowed=true`、`diagnostic_retry_requires_new_attempt=true`
  和 `excluded_from_same_panel_tuning=true`。
- 严格通道必须同时有 TS 驻点/梯度收敛、唯一目标虚频模式、双向 IRC、端点的
  电荷/自旋/原子映射/身份合同、实际 `observed_event`，并且证据属于同一候选和
  同一认证协议。`planned_event` 不计数。`valid_alternative` 只有在通过相同认证
  且实际 observed event 或端点连接关系不同才算另一条通道；去重键为
  `parent_identity + normalized_observed_event + endpoint_connectivity +
  stereochemical_or_state_identity`。构象、随机种子、重复尝试、重复认证和协议
  版本变化都是 lineage 或分层证据，不得重复计数。
- 主指标固定为固定 operational budget 内、每个独立母体系和方法臂的
  `unique_strict_certified_channels`。预算必须以向量预先冻结：
  `max_model_forward_evaluations`、`max_candidates`、
  `max_calculator_primitives`、`max_gpu_hours`、`max_cpu_core_hours`、
  `max_wall_seconds`、`max_retries`、`max_diagnostic_attempts`。失败、超时、
  重试和未解析候选消耗的费用保留在账本；缓存重放新增成本为零，不能产生新通道。
- `development_comparison` 只能使用预先冻结的输入、面板、方法、协议和预算，且
  结果仍限于开发范围。`independent_confirmation` 需要新来源或新反应家族、未参与
  训练/选择/调参的母组、冻结后盲化的方法臂和独立预算；开发结果不得升级为确认结论。
- W08 先做 2×2：联合生成 J 与允许重启、重采样、回溯和合理调参的强串联，分别配
  末端/无生成内 P 与生成内 P。四臂匹配输入、工具、随机设置和预算，用于估计 J
  主效应、P 主效应及 J×P 交互。只有真实闭环后才做 W09/W10 的 fixed policy、
  simple heuristic、feedback Agent 比较；C 的增量不能吸收 J 或 P 的收益。
- 双时钟同时登记机制指标（反馈到达时事件是否仍可修改、intervention window、
  冻结前有效步数、反馈后事件改变率）和主科学指标（严格认证通道、至少一条通道
  的母体系比例、首次认证成本）。窗口扩大或事件改变只能证明时序机制，不得单独
  宣称反应发现能力提高。
- 诊断重试必须使用新的 `attempt_id`、独立预算并保留原始失败；不得覆盖 evidence、
  修改同一面板的模型/阈值/策略/停止规则或重选母体系。若确认是全局协议问题，修复
  后须对所有受影响方法臂一致重跑，并建立新的实验身份。

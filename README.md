# xtbflow

**守恒事件—几何流匹配与校准半经验势的反应步骤／过渡态提议研究。**

远程仓库：`git@github.com:askusk521-cyber/xtbflow.git`。该仓库被指定为后续可版本化项目资产的远程主仓库；旧工程尚未迁入，不代表已完成训练、量化计算或目标机理验证。

## 当前研究安排

第一阶段采用公开参考数据与 xTBloom / GFN2-xTB，测试守恒事件流、同构型能量／力校正和几何生成。新增 DFT 不是首次模块训练的先决条件。CP2K 作为首选新增参考后端；VibeQC 保留为需要重新验证能力的替代候选。新增湿实验暂缓。

核心假设：守恒事件与三维反应方向的联合流匹配，结合经参考数据校准的半经验势，能在同等物理计算预算下优于强规则、独立串联和统一物理后处理。该假设尚未被证明。

## 当前授权与读取顺序

仓库所有者于 2026-09-25 在项目对话中指定本仓库并授权 AI 完整项目操作权限。Agent 可自主修改、提交、推送、测试、维护问题和执行有硬上限的项目计算，无需逐项重复请求许可；但合并操作必须遵守下述人工审批规则，日常授权不包含自行合并或绕过审批的权限。完整授权不等于已经取得集群连接、无限算力或科学验证结论。

1. `AGENTS.md`：当前执行指令。
2. `configs/resources.yaml`：当前授权、资源状态和预算政策。
3. `configs/task.yaml` 与 `configs/public_first_plan.yaml`：当前任务与阶段验收。
4. `docs/model_design.md`、`docs/data_contract.md`：候选方法与数据合同。
5. `archive/`：历史记录，不是现行权限或启动顺序。

旧文档中“未批准”“预算为零”“只能写 starter 目录”等操作限制已由本次授权替代；科学适用范围、测试隔离和数据完整性要求仍然有效。

## 仓库管理与合并审批

**所有合并默认必须先经人工明确审批；唯一例外是仓库所有者或其明确委托的人类审批者清楚说明指定范围内“无需审批”。** 该规则适用于所有 PR、目标分支及合并方式，不因改动较小、仅修改文档、测试通过或 AI 自审通过而豁免。

审批与明确豁免须对应具体 PR／变更范围，并在 PR 中记录；未获得批准或明确豁免时，只提交待审变更，不执行合并，也不得通过自动合并、直接推送或其他方式绕过审批。旧有“完整操作授权”“无需逐项许可”不覆盖此合并审批要求。详见 [仓库管理规则](docs/REPOSITORY_MANAGEMENT.md)。

## 第一项实际工作

读取已授权且实际可访问的旧工程与数据，复用已有审计，生成真实资产索引和公开数据子集清单；实现并验证 xTBloom 接口；在同一公开参考构型上补算半经验能量／力，开展最小校正势与流匹配消融。访问不到的旧路径记录为未连接，不得伪造数据记录或无理由重做全部审计。

在新构型／新通道需要参考验证时接入 CP2K。扩大训练前需要开发集证据、可复现代码、真实资源上限；测试标签不得反馈训练。

## 文件与执行边界

初始化包包含启动文档、配置、历史材料及完整性检查，不包含旧工程代码、模型权重、真实数据集、计算器安装或性能结果。`docs/PROVENANCE.md` 记录来源。主机私有路径通过环境变量或忽略的本地配置提供，不发布凭据、受限原始数据和封存测试标签。

在远端完成实际推送后，普通克隆方式为：

```bash
git clone git@github.com:askusk521-cyber/xtbflow.git
cd xtbflow
python scripts/validate_bootstrap.py
```

该命令只检查初始化文件完整性，不运行量化计算或训练。后续修改清单覆盖的文件时，应在同一提交更新 `bootstrap_inventory.json`。

本次连接状态见 `docs/HANDOFF_STATUS.md`。GitHub 仓库授权不会自动提供 SSH、Slurm、GPU 或软件安装。

## Current software and research status

The packaged software slice is available under `src/xtbflow` and exposes an
explicit adapter to the immutable `vendor/mechai_reusable` snapshot.  New
algorithms stay under `src/xtbflow`; the vendor snapshot is read-only.  The
bounded Workflow B controls, runtime ledger, input firewall, calculator
contracts, and source/grouping audits have software evidence.  The joint-flow
fixture is still an untrained software prototype, and the paired event–TS
manifest remains blocked until every record has mapping, electronic-state,
license, step-pairing and overlap evidence.

The repository also contains a separate SPICE2 E/F development lane with
reconstructible grouped learning-curve evidence and a limited GFN2/tblite
qualification scope.  Those E/F labels are not event–TS supervision.  The
xTBloom and CP2K records document adapter/parser and bounded smoke behavior;
production qualification and path evidence remain separate gates.  No line
above is a claim of a trained scientific model, broad chemical accuracy, or a
confirmed mechanism.  The module-level matrix and evidence links are kept in
[`docs/PROJECT_STATUS.md`](docs/PROJECT_STATUS.md).

The 2026-10 parallel work packages use one artifact-level dispatch and shared
ledger.  Read [`docs/research/execution_20261001.md`](docs/research/execution_20261001.md)
and [`configs/runs/research_dispatch_20261001.json`](configs/runs/research_dispatch_20261001.json)
for the twelve owners, write scopes, status matrix, handoff fields and
dependency gates.  A structurally valid offline dispatch deliberately reports
`science_ready: false`; unknown or blocked artifacts cannot be promoted to a
passed result.

Install the package and run the available test slices with:

```bash
python -m pip install -e ".[test]"
python -m unittest discover -s tests/reusable -p "test_*.py"
python -m pytest -q tests/test_package.py tests/test_research_dispatch.py
```

## Runnable MVP

The bounded MVP uses one fixed reactant fixture and a serial candidate path.
From a clean checkout, run the dependency-free software demo with:

```bash
PYTHONPATH=src python -m xtbflow.mvp run --config configs/mvp_fixture.yaml
```

The command creates `artifacts/mvp_run_<timestamp>/` containing
`manifest.json`, `candidates.jsonl`, `ledger.json`, `summary.md`, and
`summary.png`.  The default `backend=mock` is a deterministic harmonic proxy
and is labelled `evidence_level=software_demo`; it is not a physical or
chemical result.  To request the real xTB adapter explicitly, run:

```bash
PYTHONPATH=src python -m xtbflow.mvp run --config configs/mvp_fixture.yaml --backend xtb
```

If the native `xtb` module is unavailable, the xTB command writes the same
structured artifacts with `status=blocked_backend` and does not silently fall
back to the mock backend.  Candidate records always include status,
`reason_code`, attempt, wall seconds, CPU hours, and GPU hours.  Replays with
the same config, backend, and seed expose a stable
`replay.deterministic_summary_sha256`; timing and native build identity may
vary by environment.

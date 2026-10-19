# xtbflow

**守恒事件—几何流匹配与校准半经验势的反应步骤／过渡态提议研究。**

任务是只给反应物，由模型同时生成电子事件（键级／电子记账变化）和三维过渡态几何。GFN2-xTB 等半经验方法有三种用途：校正势的底座、生成过程中的物理信号、低成本认证链。最终认证用 DFT（PySCF + gpu4pyscf）。

- **现状**：见 [`docs/STATUS_ZH.md`](docs/STATUS_ZH.md)，内容包括各阶段结论、PR 叠放关系和待负责人决定的事项。
- **项目主页**：<https://askusk521-cyber.github.io/xtbflow/>，包括论文式说明、实验结果和每日更新的运行状态。
- **历史材料**：2026-10-06 之前的 README 与状态页已原样存于 [`archive/status_20260930/`](archive/status_20260930/)；更早的启动与交接材料见 [`archive/`](archive/)。

## 研究问题

核心假设：守恒事件与三维反应方向的联合（并联）流匹配，结合经参考数据校准的半经验势，能在同等物理计算预算下优于强规则、独立串联和统一物理后处理。**该假设尚未被证明。**

联合模型与串联对照已在 Transition1x 上正式训练并评测过（M0，2026-10-07）。在无引导、候选数相同的条件下，联合的单候选命中率没有高于串联，差值区间包含 0。之后的检查都属探索性，读法见 `docs/STATUS_ZH.md`。

## 授权、读取顺序与合并审批

仓库所有者于 2026-09-25 指定本仓库，并授权 AI 完整的项目操作权限：Agent 可以修改、提交、推送、测试、维护 issue，并执行有硬上限的项目计算。完整授权不等于已经取得集群连接、无限算力或科学验证结论。

读取顺序：

1. `AGENTS.md`：当前执行指令。
2. `configs/resources.yaml`：授权、资源状态和预算政策。
3. `configs/task.yaml` 与 `configs/public_first_plan.yaml`：任务与阶段验收。
4. `docs/model_design.md`、`docs/data_contract.md`：方法与数据合同。
5. `archive/`：历史记录，不是现行权限或启动顺序。

**所有合并默认必须先经人工明确审批；唯一例外是仓库所有者或其明确委托的人类审批者清楚说明指定范围内“无需审批”。** 该规则适用于所有 PR、所有目标分支和所有合并方式，不因改动较小、仅改文档、测试通过或 AI 自审通过而豁免。审批与豁免须对应具体的 PR 或变更范围，并在 PR 中记录。未获批准时只提交待审变更，不得通过自动合并、直接推送或其他方式绕过审批。详见 [仓库管理规则](docs/REPOSITORY_MANAGEMENT.md)。

## 软件现状

`main` 上的 `src/xtbflow` 包含：

- **模型**：守恒事件流与约束解码（`models/event_flow.py`、`event_decoder.py`）、等变几何流（`geometry_flow.py`）、联合与串联事件—几何流（`joint_flow.py` 中的 `JointEventGeometryFlow`、`SerialEventGeometryFlow`）、Delta 能量／力模型（`delta_energy.py`）；
- **训练**：事件流、几何流和联合流的训练，支持断点续跑（`training/`）；
- **计算器**：GFN2-xTB／tblite 适配、xTBloom、CP2K（`calculators/`），单位边界为 Å、Hartree、Hartree/Å；
- **运行时**：有硬上限的计费、重试与缓存账本（`runtime/`）。

`vendor/mechai_reusable/` 是旧工程的不可变快照，通过适配层调用，不在原处修改。

以下内容都还在**尚未合并**的 PR 分支上：M0、V1a、V1b 的实验代码（数据管线、训练与采样脚本、闸门评测、重分析），以及所有探索性检查。叠放关系见 `docs/STATUS_ZH.md` 第 2 节。模型权重、原始数据和运行输出不入库，只保存在 n2 的运行目录里，仓库中只提交它们的哈希。

## 本地检查（CPU，与 CI 一致）

```bash
python -m pip install -e ".[test]"
python scripts/validate_bootstrap.py
python -m unittest discover -s tests/reusable -p "test_*.py"
python -m pytest -q tests --ignore=tests/reusable
```

`validate_bootstrap.py` 只检查清单内文件的完整性，不运行量化计算或训练。修改清单覆盖的文件后，要运行 `python scripts/validate_bootstrap.py --refresh`，并在同一提交里更新 `bootstrap_inventory.json`。

n2 运行环境见 [`docs/ENVIRONMENT_N2.md`](docs/ENVIRONMENT_N2.md)（2026-09-30 核查），之后的变化见 `docs/STATUS_ZH.md`。主机私有路径通过环境变量或被忽略的本地配置提供。不发布凭据、受限原始数据和封存的测试标签。

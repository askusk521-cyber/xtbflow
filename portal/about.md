# 关于本站

## 内容来源与原则

- **只汇总，不产生新结论。** 所有数字摘自所链 PR 的描述、评论或报告；文字版来源是 [`docs/STATUS_ZH.md`](https://github.com/askusk521-cyber/xtbflow/blob/main/docs/STATUS_ZH.md)。
- **一处更新。** 实验结论、证据路径、运行状态、参考文献和存档条目都写在 [`portal/records/`](https://github.com/askusk521-cyber/xtbflow/tree/main/portal/records) 的 YAML 里；图表、数据目录和证据框在构建时由 [`scripts/portal_hooks.py`](https://github.com/askusk521-cyber/xtbflow/blob/main/scripts/portal_hooks.py) 生成。
- **链接可追溯。** 证据链接固定到 PR 的头提交；[`scripts/check_portal.py`](https://github.com/askusk521-cyber/xtbflow/blob/main/scripts/check_portal.py) 检查每个路径在对应提交中存在，并检查记录的头提交与 GitHub 上一致。

## 如何更新

| 场景 | 改哪里 |
|---|---|
| 新实验出结果 | `experiments.yaml` 加一条（问题、结论、PR、证据路径、n2 目录）；有双侧区间的配对差值加进 `forest` |
| 运行进度变化 | `progress.yaml` 的 `running` 与 `updated` |
| 负责人做出决定 | `progress.yaml` 的 `next`，同时更新相关实验的状态 |
| PR 推了新提交 | `experiments.yaml` 中该 PR 的 `sha`（`check_portal.py` 会提示不一致） |
| 新增参考文献 | `references.yaml` |

本地预览：

```bash
pip install -e ".[site]"
python scripts/check_portal.py --refresh-github
mkdocs serve
```

## 仓库治理

所有合并须负责人明确批准（[`docs/REPOSITORY_MANAGEMENT.md`](https://github.com/askusk521-cyber/xtbflow/blob/main/docs/REPOSITORY_MANAGEMENT.md)）。本站的源文件在 main 上，修改同样走 PR 审批。

GPU 计算在集群节点 n2 上经 Slurm 提交；原始输出留在 n2，仓库只提交报告、结果文件与哈希。

# Joint generation, physics feedback and chemistry decisions

The active objective is [#129](https://github.com/askusk521-cyber/xtbflow/issues/129)
and all thirteen execution issues below. Old task closures archive dispatch;
they do not erase research results or establish completion of the new tasks.
Each work package needs both working software and the scientific comparisons
specified in its issue. None is declared complete by this navigation update.

中文说明：当前路线研究联合生成、生成中的物理反馈和根据证据选择计算的 Agent。
旧任务与旧报告作为可复用资产保留，后续开发按下面十三项任务验收。
软件测试、真实案例、开发比较和独立确认分别记录，缺少任何所需证据时保持任务开放。

## Module responsibility and handoffs

Responsibility below is assigned to work packages, not to unconfirmed human
assignees. W01 is currently implemented on `codex/w131-asset-replay-20261107`
from main `d1aee8f`; the other packages have no writer assigned in this change.
When taking another package, record its branch and actual writer in its issue.
Changes spanning package boundaries require an explicit interface handoff.

| Work package | Primary module responsibility | Reusable PRs / input | Required next deliverable |
|---|---|---|---|
| [W01 #131](https://github.com/askusk521-cyber/xtbflow/issues/131) | asset registry, integration entry points, navigation | #111/#113/#114/#119/#120 | hashes and archival replay, then fresh reactant→candidate→certification→ledger chain |
| [W02 #132](https://github.com/askusk521-cyber/xtbflow/issues/132) | data records, task views, splits, candidate identity | main data contracts, W01 assets | real task counts and manifests; hidden-label invariance; matched-geometry labels |
| [W03 #133](https://github.com/askusk521-cyber/xtbflow/issues/133) | joint model, clocks, local coupling and training | #111, W02 | trainable clocks and trajectories, then certified paired comparisons |
| [W04 #134](https://github.com/askusk521-cyber/xtbflow/issues/134) | partial conditioning and water roles | W02/W03 | explicit conditions, masks and real conditional completions |
| [W05 #135](https://github.com/askusk521-cyber/xtbflow/issues/135) | reaction-region scalar correction potential | main Delta, #114, W02 | matched labels, grouped learning curves and finite reference acquisition |
| [W06 #136](https://github.com/askusk521-cyber/xtbflow/issues/136) | physics feedback in sampling | W03/W05/W07 | physics→geometry→event trace with common certification |
| [W07 #137](https://github.com/askusk521-cyber/xtbflow/issues/137) | certification service, cache and recovery | #119, main validation and ledger | real replay, live bounded service and actual alternate events |
| [W08 #138](https://github.com/askusk521-cyber/xtbflow/issues/138) | controls, full accounting and paired attribution | #113/#120, W01/W02/W07 | executable strong controls and equal-budget paired results |
| [W09 #139](https://github.com/askusk521-cyber/xtbflow/issues/139) | finite decision actions, feedback memory | W02/W07 | deterministic offline replay, then two real feedback loops |
| [W10 #140](https://github.com/askusk521-cyber/xtbflow/issues/140) | allocation policies and policy evaluation | W07/W08/W09 | fixed/heuristic/feedback comparisons with identical tools |
| [W11 #141](https://github.com/askusk521-cyber/xtbflow/issues/141) | fixed-pool water campaign inputs | W02/W07, later W03/W06/W09/W10 | independent CHNO reactants, screening cost, then frozen campaign |
| [W12 #142](https://github.com/askusk521-cyber/xtbflow/issues/142) | endpoint network and frontier expansion | W02/W07/W09/W10 | observed endpoint network, then new intermediate→next certified edge |
| [W13 #143](https://github.com/askusk521-cyber/xtbflow/issues/143) | independent confirmation and publication evidence | W02/W07/W08/W10–W12 | claim/evidence map, frozen independent groups, reproducible figures |

## Current executable increment

[Asset replay contract](architecture/research_asset_replay.md) and
`configs/research/assets_20261107.json` pin five research heads independently
of changing PR summaries. The replay initially reconstructs existing O2
analysis and candidate/cost links. It is not fresh quantum chemistry, a new
reactant-only benchmark or a claim that these issues are finished.

## Persistent routing

`configs/run_status.json` pins #130 as the operational endpoint and #129 as
the task roadmap. `scripts/update_run_status.py` reconstructs the bilingual
roadmap explanation on each update while retaining `RUN_STATUS_JSON` unchanged.
An unavailable or closed configured endpoint fails rather than silently
creating a new task tracker. Queue observations do not change issue acceptance.

The homepage workflow reads the issue labeled `run-status`; this label is
preserved. Its deployed content comes from the separate `site` branch, so
changing main navigation does not by itself publish a new website. Homepage
content synchronization remains a later delivery; no website is published by
this PR. Existing scientific reports and PR histories remain intact.

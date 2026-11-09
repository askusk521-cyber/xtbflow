# Certification evidence correction — 2026-11-10

The earlier statement that `37e42e0` completed strict TS/IRC certification was
incorrect. The program did return `CERTIFIED`, but the producer did not supply
the independent path and endpoint checks required by
[#137](https://github.com/askusk521-cyber/xtbflow/issues/137). A passing boolean
validator cannot establish checks its producer did not perform.

At that commit, `path_connected` was computed from two optimizer convergence
flags and path lengths of at least two frames. The method was explicitly
`bounded_cartesian_gradient_following_v1`, followed by BFGS endpoint
optimization. Endpoint Hessians/frequencies were absent. The code treated the
candidate geometry as the reactant graph and selected the lower-energy end as
the product. This does not establish mapped reactant identity or an observed
chemical event. The R0 output's empty event is not a certified channel.

The raw manifest for `37e42e0` records configuration hash
`6aa438af414600d97997a7c823bda0a803a90de90d9a5a1536d4e7b5c769641f`.
The public `live_reactant_only_certified_exact_37e42e0_20261110.json` instead
contains `ad10fe307f1cf2d92f60b3f02d49d8953c6a9ea4ec4deefceccbc32f49e42ae2`,
from a different run configuration. The raw manifest and the retained n2
configuration agree. Historical source commits and dirty flags are not
rewritten to conceal this mismatch.

| Historical record | Correct interpretation |
| --- | --- |
| `xtb_irc_backend_smoke_exact_a149763_20261110.json` | xTB/Hessian/endpoint smoke; strict-certification claim unsupported |
| `live_reactant_only_certified_exact_37e42e0_20261110.json` | Unresolved path diagnostics; published config hash mismatched |
| `reactant_only_saddle_refinement_probe_exact_1a7c72d_20261110.json` | Exploratory external-script diagnostic; not an implementation qualification |
| `xtb_irc_backend_preflight_failure_20261110.json` | Preflight software failure; literal zero timings were not copied from the raw run and are not measured evidence |

All historical files are retained unchanged. The audit command below emits a
new report, records hashes of its inputs and its own exact source, and copies
original consumed calls/costs without another calculator invocation:

```bash
PYTHONPATH=src python scripts/audit_live_certification_evidence.py \
  --manifest "$RAW_RUN/manifest.json" --config "$RUN_CONFIG" \
  --published-evidence docs/evidence/live_reactant_only_certified_exact_37e42e0_20261110.json \
  --output "$AUDIT_OUTPUT"
```

Exit 1 means the historical claim is invalidated; exit 2 means audit input or
execution failure. The script never grants a strict certificate. Evidence and
commands for the new exact audit are recorded separately under `docs/evidence/`.

The [exact audit and rerun receipt](evidence/certification_evidence_correction_20261110_fcec27e.json)
was produced by clean auditor source `fcec27ed9cdd6f58e7b20eb9f22c73f14d4f9b0a`.
Corrected xTB run/replay source was
`a8e4a88069ba363f8e76f8378a58795800718648`, also clean. Both runs used the
unchanged raw configuration hash above. The arms retained 314/306 diagnostic
calls plus one feedback call each. Both return `attempted_unresolved` with
`INCOMPLETE_IRC_CERTIFICATION`. The deterministic summary, including costs in
calls and validation identity, matched:
`e8d7ff5ed6bc56b37205901b47596356d701e81ba00b31f8b8814d06e175e079`.
The targeted contract/accounting regression at the auditor source passed
68 tests. This is software evidence, not independent chemistry validation.

Exact read-only audit replay on n2 (30 seconds, zero calculator calls; retained
raw outputs required; the timestamp selects a new output and preserves history):

```bash
ssh n2 '
  set -e
  cd "$HOME/xtbflow/xtbflow-exec-cert-audit-fcec27e-20261110"
  test "$(git rev-parse HEAD)" = fcec27ed9cdd6f58e7b20eb9f22c73f14d4f9b0a
  test -z "$(git status --porcelain)"
  source ~/miniconda3/etc/profile.d/conda.sh
  conda activate xtbflow
  audit_root=/tmp/xtbflow-cert-audit-a8e4a88-20261110
  audit_output=/tmp/xtbflow-certification-review-$(date -u +%Y%m%dT%H%M%SZ).json
  timeout 30s python scripts/audit_live_certification_evidence.py \
    --manifest /tmp/live-loop-reactant-modefollow-37e42e0/manifest.json \
    --config "$audit_root/frozen_config.json" \
    --published-evidence docs/evidence/live_reactant_only_certified_exact_37e42e0_20261110.json \
    --rerun-manifest "$audit_root/run/manifest.json" \
    --replay-manifest "$audit_root/replay/manifest.json" \
    --output "$audit_output"
'
```

The corrected adapter returns `attempted_unresolved` with
`INCOMPLETE_IRC_CERTIFICATION` after successful proxy diagnostics. It preserves
the two actual endpoint distance graphs without assigning chemical roles,
sets `path_connected=false`, and reports `observed_event=null`. Candidate,
verification and cache identities are populated; the policy revision prevents
reuse of old success records. The live replay identity now includes diagnostic
call counts and validation identities, and its consolidated ledger includes
the separate diagnostic costs. No QC threshold, parent denominator or baseline
is changed.

The goal remains incomplete. It still needs the existing independent reference
service's full TS refinement, final modes, bidirectional IRC, verified endpoint
minima and mapped identity checks, with bounded costs and raw-artifact replay.
The current producer is not a substitute for that service. No new parameter
search or alternative fixture is authorized by this correction.

## 中文说明

此前“严格 TS/IRC 已完成”的结论需要撤回。现有代码只做了短距离下坡和端点优化，
没有完整路径与端点极小点证明，因此程序打印成功不等于独立认证成功。
历史文件保留，新增审计记录会指出证据和配置哈希的缺口，已消耗费用仍然计入。
接下来应接入仓库已有的独立认证服务，不能通过调整阈值、挑选更容易的构型或增加
免责声明，把这些代理指标升级为严格证据。

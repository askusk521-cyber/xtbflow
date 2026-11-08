# Project status and evidence boundary

审阅基线：`cc11455403f1d8d3a2cf7a9a3fe35fd4168cfc47`（2026-10-01）。本页把
“规格”“软件”“真实开发证据”和“化学／确认结果”分开记录；一个模块的
状态不能覆盖另一个模块。并行工作包与机器可读交接以
[`research_dispatch_20261001.json`](../configs/runs/research_dispatch_20261001.json)
为准。

## Module matrix

| Module | interface specified | implemented | software tested | real development run | chemically validated | confirmatory |
| --- | --- | --- | --- | --- | --- | --- |
| Canonical event–geometry boundary | observed — [`architecture_v0.1.md`](architecture/architecture_v0.1.md) | observed — contract/schema tests | observed — [`workflow_b_software_acceptance_20260929.json`](evidence/workflow_b_software_acceptance_20260929.json) | blocked — paired event/TS admission | blocked — no chemical claim from smoke | blocked — W11 freeze required |
| Bounded Workflow B controls | observed — [`workflow_b_baseline_protocol.md`](architecture/workflow_b_baseline_protocol.md) | observed — merged control prototype | observed — joint-flow and workflow tests | blocked — no admitted event/TS training at this base | blocked | blocked |
| Track-B event–TS data | observed — [`data_contract.md`](data_contract.md), issue #76 | observed — quarantine/admission contracts | observed — source/grouping audits | blocked — no paired admitted manifest | blocked | blocked |
| SPICE2 E/F scalar/Delta lane | observed — public manifest and split contracts | observed — E/F pair and learning-curve code | observed — 9 grouped development points | observed — development evidence only | blocked — no broad chemical qualification | blocked |
| GFN2/tblite pilot adapter | observed — explicit charge/spin/units boundary | observed — bounded adapter | observed — independent finite-difference evidence | observed — limited tested H/C/N/O/S closed-shell scope | blocked outside the recorded scope | blocked |
| xTBloom adapter | observed — adapter contract | observed — diagnostic bridge | observed — CPU source smoke | blocked — native production qualification open | blocked | blocked |
| CP2K reference path | observed — protocol/parser contract | observed — parser and artifact retention | observed — calibration/convergence smoke records | blocked — path/force qualification gate | blocked | blocked |
| Research dispatch (W00) | observed — this contract | planned/branch-local | planned until focused tests run | not_run by design | blocked | blocked |

`observed` in this table means that the cited software or audit evidence was
recorded; it does not mean that the next scientific gate passed.  `blocked`
and `not_run` preserve the denominator and the reason.  In particular, E/F
learning curves are useful development evidence for the scalar correction lane,
but they are not event–TS supervision and cannot unlock the paired training
gate by themselves.

## Current handoff state

The first-wave packages can prepare contracts, fixtures, source audits, model
controls, evaluation rules, reference preflight, independent seed inputs and
research hypotheses in parallel.  Real computation remains globally bounded
by the shared stage ledger and `max_concurrent_jobs: 1`.  The principal gates
are:

```text
W01 paired manifest + W02 control factory + W03 metric protocol
    -> W04 S1 separately trained generator comparison
W04 generator + W06 reference qualification (+ W05 for Delta)
    -> W07 S2 guidance placement
W10 pre-registered candidate protocol -> W09 independent seeds
W09 seeds + W06 driver -> W09 attempts -> W10 application choice
W04/W07/W08/W09/W10 frozen evidence + W06 -> W11 confirmation
```

The graph is artifact-level, so a pre-registration artifact can precede an
input artifact while the eventual result returns to the research-positioning
package without forming an issue-level cycle.  Validate it offline with:

```bash
python scripts/validate_research_dispatch.py \
  --config configs/runs/research_dispatch_20261001.json \
  --offline --output /tmp/dispatch-audit.json
```

The report is valid only when every producer, consumer, path/type, handoff
field and dependency passes.  `science_ready: false` is expected until real
upstream artifacts exist; a pseudo `passed` state or an unknown artifact marked
ready is rejected.

## Evidence and claim limits

- Synthetic joint-flow fixtures establish routing, conservation and recovery
  behavior only.  They are not a trained model or a chemistry result.
- Public E/F records and GFN2 diagnostics have their own source, split, unit
  and scope evidence.  They cannot be silently re-labelled as paired event–TS
  data, transition-state validation, or independent reaction discovery.
- Missing charge, multiplicity, coordinate mapping, license, step pairing,
  protocol or overlap proof stays `unknown`/`quarantine`.
- Remote PR state in the dispatch is a dated `snapshot_only`; an offline run
  never asserts current CI, mergeability or branch state.
- Confirmation consumes a new, unused batch only after method, application,
  protocol, weights, budget and stopping rule are frozen.  No package may
  merge another package or bypass the repository's human approval rule.

Historical evidence remains in `docs/evidence/` and is not rewritten to make
the current project appear further along.  New runs must add a dated report
with host, commit, config/data/protocol hashes, resource accounting, failure
states, claim limits and unfinished work.

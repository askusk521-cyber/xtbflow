# 2026-10-01 research execution dispatch

This document is the human-readable companion to
[`configs/runs/research_dispatch_20261001.json`](../../configs/runs/research_dispatch_20261001.json).
It records the coordination contract for issues #74–#86.  It is an interface
and handoff record; it does not claim that the model, data, calculator, or
confirmation stages have completed.

The dispatch was prepared from base commit
`cc11455403f1d8d3a2cf7a9a3fe35fd4168cfc47` on branch
`codex/issue75-dispatch`.  The #74 review baseline was
`a0c959358b988f220f66a3641c53bb9d997069da`.  A later integration branch must
record its actual base and head again rather than treating either SHA as a
permanent main-branch pointer.

## Evidence vocabulary

Every work package has one row for each of the following stages:

`interface_specified → implemented → software_tested → real_development_run → chemically_validated → confirmatory`

The row state is one of `observed`, `planned`, `blocked`, `unknown`,
`not_run`, or `failed`.  `passed`, `complete`, and `success` are deliberately
not state values.  An observed row must name an evidence path and a commit;
planned and blocked rows carry a reason.  This makes a software contract
reviewable without promoting it to a scientific result.

The current dispatch is `interface_contract_frozen_science_incomplete`.
Running the validator reports `valid: true` for a structurally sound
contract, while `science_ready: false` remains explicit.  In the current
snapshot, the admitted paired manifest, S1 results, reference qualification,
S2 results, independent attempts, application selection, and confirmation
result are blocked or not yet run.

## Single responsibility and write scope

The `owner` values below are roles, not fabricated user accounts.  A real
claimant records a session and branch in the issue/PR handoff.  A path appears
in only one `write_scope`; shared core files require a change request to the
owning role.

| Work package | Issue | Single responsibility | Ready now | Real execution gate |
| --- | ---: | --- | :---: | --- |
| W00 | #75 | integration contract, status, DAG and handoff | yes | none; no training/QC |
| W01 | #76 | Track-B paired event–TS admission and input views | yes | source/map/electronic-state evidence |
| W02 | #77 | model control factory and strong baselines | yes | real training is W04 |
| W03 | #78 | paired metrics, attribution and cost protocol | yes | protocol freeze before S1 |
| W04 | #79 | separately trained S1 generator comparison | yes (runner/fixture) | W01 + W02 + W03 + shared balance |
| W05 | #80 | Delta learning curves and domain checks | yes (eligible cache lane) | qualified E/F rows and bounded seeds |
| W06 | #81 | reference driver, TS/path qualification and ledger fixes | yes (software) | finite calculator protocol and reservation |
| W07 | #82 | matched-budget S2 guidance placement | yes (interface/toy) | W04 generator + W06 qualification |
| W08 | #83 | one selected coupling object/timing question | yes (pre-registration) | S1 failure classification |
| W09 | #84 | independent seeds and rule-first attempts | yes (input side) | W06 driver for physical attempts |
| W10 | #85 | literature/competitive hypotheses and application choice | yes (source/protocol) | W04/W07/W09 development evidence |
| W11 | #86 | development/confirmation isolation and final decision | yes (contract) | frozen method, application, new batch, W06 |

The JSON `work_packages[*].write_scope` is authoritative.  In particular,
W00 does not write model, training, evaluation, data-admission, or calculator
core files; it only owns the coordination files listed in its scope.

## Artifact-level DAG

Dependencies are between concrete artifacts rather than whole issue numbers.
This permits a package to publish a pre-registration artifact and later
consume a result without creating a false issue-level cycle.  The validator
checks every edge, producer/consumer declaration, and cycle.

The principal path is:

```text
W00.dispatch_contract
 ├─ W01.paired_manifest + W01.input_view
 ├─ W02.baseline_factory + W02.control_manifest
 └─ W03.metric_protocol
        └─ W04.s1_plan → W04.s1_results
                                ├─ W07.s2_plan → W07.s2_results
                                ├─ W08.coupling_selection
                                └─ W10.application_selection
W10.candidate_protocol → W09.seed_manifest → W06.reference_qualification
                                       └→ W09.attempt_records → W10.application_selection
W04/W06/W07/W08/W09/W10 artifacts → W11.confirmation_contract/result
```

The complete 34-edge list, required status for each edge, schema, logical
path, and artifact hash placeholder is in the JSON manifest.  A required
status is a gate, not an assertion that the status is already present.  For
example, `paired_manifest_to_s1` requires `software_tested`; the current
manifest reports that dependency as unmet because W01 has not admitted a
scientific record at this base.

`artifact_hash` is `unknown` for planned or blocked outputs.  It must be a
64-character SHA-256 before an output is used for a scientific handoff.
Repeated artifact names with different hashes are rejected; even identical
duplicate names are rejected so that consumers cannot silently choose a
different version.

## Handoff contract

Every package uses the same payload.  Values that do not exist are written as
`unknown` or a concrete failure state; they are never replaced with a passed
placeholder.

```text
work_package:
run_id:
evidence_level:
code_commit:
config_hash:
data_hash:
split_hash:
input_view_hash:
checkpoint_hash:
physical_protocol_hash:
ledger_id:
output_index:
status:
failure_reason:
```

`ledger_id` must remain `shared-stage-ledger`; this work package does not
create a second budget database.  `output_index` names logical repository or
run outputs and never publishes private raw paths, credentials, model weights,
or unlicensed source bytes.  The template in the JSON manifest retains the
same fields for all twelve packages.

## Pull-request snapshots and offline boundary

The coordination review captured the following dated observations while
reading the associated pull requests.  They are stored as
`verification.pull_requests[*]` with `claim: snapshot_only`:

| PR | Observed state | Head | Base | Merge observation |
| ---: | --- | --- | --- | --- |
| #61 | OPEN | `2baa5ce9` | `main` | no merge commit |
| #62 | MERGED | `6c88173e` | `main` | `abee6248`, 2026-09-29 |
| #63 | OPEN | `be52a939` | `main` | no merge commit |
| #66 | MERGED | `32a763c9` | `main` | `ec803556`, 2026-09-30 |
| #67 | OPEN | `3a5f10a3` | `codex/workflow-a-track-b-20260929` | no merge commit |
| #70 | OPEN | `5638ace2` | `codex/c2-reference-bridge-20260930` | no merge commit |
| #71 | OPEN | `343893a8` | `main` | no merge commit |

The short hashes above are display forms; the machine-readable manifest keeps
full commit identities.  `--offline` never calls GitHub and sets
`remote_currentity: unknown_offline`; it cannot turn a stored snapshot into a
current mergeability or CI claim.  A future live check must be a separately
recorded observation with a new date and source.

## Validation and handoff commands

The implementation is standard-library only and does not launch a scientific
job:

```bash
python scripts/validate_research_dispatch.py \
  --config configs/runs/research_dispatch_20261001.json \
  --offline \
  --output /tmp/dispatch-audit.json

PYTHONPATH=src:vendor/mechai_reusable \
  pytest -q tests/test_research_dispatch.py
```

The focused contract tests cover an intentional cycle, same-name/different-
hash collision, unknown/ready and pseudo-passed claims, missing handoff
fields, typed paths, and an attempted live remote claim in offline mode.
Unavailable local Python/test dependencies are an environment limitation, not
evidence of a scientific failure.  Full CPU/reusable tests and bootstrap
verification remain integration-stage checks; W00 does not refresh or hand-edit
`bootstrap_inventory.json`.

## Claim limits and stop rules

No GPU, QC, training, data download, or n2 job is authorized by this dispatch.
The shared P1 limits remain those in `configs/resources.yaml`; the single
concurrency limit applies across all packages.  A missing upstream artifact is
reported as `blocked`/`unknown` with its denominator and next evidence request.
It does not justify changing an input, treating a fixture as a real record,
or declaring a model or chemical result complete.

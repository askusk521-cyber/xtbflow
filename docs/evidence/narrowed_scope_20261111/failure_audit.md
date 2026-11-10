# S3 raw-stage audit and S2 cached trained handoff

The original #181 auditor, copied byte-for-byte from `721a9f2`, ran from the
clean **2772c7bedb9c674144d7dab702a29178eeee699a** execution checkout in n2's
`xtbflow-qc` environment. [Its new receipt](legacy_cache_audit.json) passes the
original cache/protocol/build/result/ledger checks, with zero new calculator
calls and unchanged zero-channel verdict. The new projection never invokes
`CertificationService.run`; its source hashes must not be used to refresh old
cache records.

The separate read-only projection at
`e2d670313c0f530c4c141b96e24f2848496e5263` verifies 107 asset entries
(10,863,148 bytes) across five retained attempts and the cached trained handoff.
It takes 0.084590 s wall / 0.082266 s CPU. See the
[new audit](failure_audit.json) for every original byte hash, unchanged verdict,
historical resource record, v2 stage reason and geometry/mode summaries.

| Historical candidate | Calls retained | Original verdict | Recovered stage/reason |
| --- | ---: | --- | --- |
| Rule no-feedback | 500 | same_endpoint | graph: both_ends_reactant |
| Rule physical-feedback | 402 | same_endpoint | graph: both_ends_reactant |
| Trained joint, job 4101 | 468 | attempted_unresolved | graph: endpoint_bond_perception |
| Trained independent serial, job 4101 | 638 | reference_ts_connected | graph: STRICT_JOINT_GRAPH_VALID |
| Trained feedback, job 4106 | 526 | attempted_unresolved | TS modes: FREQUENCY_GRAY_AT_CAP |

No attempt is replaced. The 4101 joint record reached TS, IRC and endpoint
calculation; the observed obstacle is endpoint bond perception. The 4106 record
has no completed IRC/endpoints and its final minimum TS frequency is about
-19.364 cm⁻¹ at the existing gray-zone cap. These are unresolved certification
outcomes, not proof that a reaction does not exist.

The rule chain's final TS frequencies begin at approximately -71.654 and
-71.671 cm⁻¹. Its two endpoint structures have fixed-map, proper-rotation aligned
RMSD about 0.927 Å, pair-distance RMS change about 0.859 Å and maximum pair-distance
change about 1.856 Å, despite both mapping to the reactant graph. All mode
amplitudes by original atom index and full frequency lists are retained in the
audit. These summaries establish geometric change, not a torsional/orientational
mechanism or causation by the placement bug; no such mechanism is guessed.

## Cached chemical bridge

The original mapped reactant query is reduced to the exact six-field worker
whitelist. Both trained saved outputs pass the existing chemically constrained
`certification_candidate` bridge. Their decoded BE, mapped atom order,
coordinates, charge/spin, environment and full content identity match the actual
archived reference requests. The original joint/event/geometry checkpoint
identities remain recorded. The original journal retains **150 historical
forwards**, not new work; the connected serial candidate is an exposed positive
calibration control. A null proposal stays valid as an unknown hypothesis and
cannot alone imply same_endpoint.

This is cached chemical handoff qualification. It does not complete a new
live trained geometry→event→decoded product→physical endpoint intervention.
S1 measures local continuous response; no chemical event change is fabricated.
The rule fixture's score vector remains distinct from decoded BE and observed
endpoint events. Its reference input consumes geometry, so changing only that
score cannot establish chemical feedback.

The first projection run at `0ef2cd7` verified all five raw records but used the
worker configuration in place of the archived handoff protocol; the resulting
content-hash mismatch correctly stopped the bridge audit. Its unchanged
[failed receipt](reference_audit_failed_0ef2cd7.json), 0.070882 s wall and all
read costs remain retained. The correction reads the hash-verified certification
plan. It is a separately committed read-only correction within the original
600-second/256-MiB envelope, with no physical/model retry.

## Reproduction and validation

Set the following n2 variables to retained sources/inputs and **new** output
filenames. Use the legacy auditor from the original evidence commit, not this
branch's certification module, for the first command.

```bash
ssh n2 '
  cd "$XTB_REFERENCE_CODE"
  test "$(git rev-parse HEAD)" = 2772c7bedb9c674144d7dab702a29178eeee699a
  test -z "$(git status --porcelain)"
  source ~/miniconda3/etc/profile.d/conda.sh
  conda activate xtbflow-qc
  ulimit -f 32768
  timeout 600 python "$XTB_REFERENCE_AUDITOR_721a9f2" \
    --run-root "$XTB_REFERENCE_RUN_ROOT" \
    --config configs/validation/live_feedback_reference_20261110.json \
    --out "$XTB_NEW_LEGACY_AUDIT"
'
ssh n2 '
  cd "$XTB_STAGE_AUDIT_CODE"
  test "$(git rev-parse HEAD)" = e2d670313c0f530c4c141b96e24f2848496e5263
  source ~/miniconda3/etc/profile.d/conda.sh
  conda activate xtbflow
  export PYTHONPATH=src:vendor/mechai_reusable CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1
  timeout 599 python scripts/audit_narrowed_reference.py \
    --service live_rule "$XTB_RULE_SERVICE" \
    --service trained_initial "$XTB_TRAINED_INITIAL_SERVICE" \
    --service trained_feedback "$XTB_TRAINED_FEEDBACK_SERVICE" \
    --generation-root "$XTB_GENERATION_ROOT" --out "$XTB_NEW_STAGE_AUDIT"
'
```

Final exact-source n2 validation: 387 pytest tests/68 subtests, bootstrap 408
files. Preceding projection source: 386/68 and 56 reusable tests. Log hashes,
environment and measurement gaps are in the separate software receipts. Mock
tests do not establish chemical accuracy. No historical scheduler accounting is
invented; prior measured program costs and missing measurements stay distinct.

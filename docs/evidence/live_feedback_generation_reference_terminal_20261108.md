# Live feedback generation and reference terminal — 2026-11-08

The dependent n2 chain is now terminal. Job 4101 completed the original two
candidate reference batch and retained one strict certified channel. Job 4105
then consumed the verified reference feedback and generated one new candidate
with 50 model forward evaluations and zero new calculator calls. Job 4106 ran
the independent certification service for that candidate, charged 526
calculator primitives, and completed with process exit code 0.

The new candidate did not certify: its final feedback status is
`attempted_unresolved` with raw status `PROTOCOL_FAILURE`, and the new-stage
strict channel count is zero. Its measured cost was 2112.462 seconds wall,
0.6603476 CPU-core-hours, 0.5865953 GPU-hours and 526 calculator calls. The
failure and all consumed calls remain in the ledger.

The structured classification is `certification_unresolved`, with
`failure_stage` and `failure_reason` retained as `unknown` because the public
report does not localize the failure further. This is not a chemical negative:
`chemical_negative` is false and `strict_channel_increment` is zero. A
diagnostic retry is allowed only as a new attempt under an independent budget;
the original attempt is preserved and excluded from same-panel tuning.

This is the first real generation-to-certification attempt from the feedback
Agent, including a negative result. It does not show an Agent efficiency gain,
an additional certified channel, or independent confirmation. The machine
readable projection records source/config/protocol/job/report hashes and keeps
raw coordinates and logs private on n2.

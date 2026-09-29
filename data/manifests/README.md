# Public record manifest

`public_records.v1.jsonl` is a source-audit handoff, not a training split. All
six rows are explicitly quarantined because this checkout has source identities
and inherited audit claims but no verified per-record geometry bytes, charge /
spin evidence, or reference E/F labels. No record is admitted to train,
validation or test, and no synthetic row is used to fill the gap.

When an accessible source file is obtained, add one row per real record with
its exact locator and SHA-256, parent/family grouping, units, label mask,
charge/spin evidence, license record and overlap audit. Run the record contract
and group-split firewall before changing `admission` from `quarantine`.

## Track-B manifests

`track_b_reaction_ts.v0.1.jsonl`, `independent_seeds.v1.jsonl` and
`search_attempts.v1.jsonl` are intentionally empty at introduction. Empty files
make the zero-admission state explicit and prevent example or guessed records
from entering the scientific pipeline. Their contracts and current source
audit are documented in `docs/TRACK_B_DATA_GATE.md`.

A Track-B record may be added only after its real bytes, units, electronic
state, atom correspondence, product supervision, event label, transition
geometry, reference protocol and grouping are pinned. Seed records must be frozen before search;
attempt records are append-only and retain failures and alternative channels.

The zero-state audit reports are:

- `docs/evidence/track_b_manifest_audit_20260929.json`
- `docs/evidence/independent_seed_manifest_audit_20260929.json`
- `docs/evidence/search_attempt_manifest_audit_20260929.json`

They deliberately report that model training and search may not start yet.

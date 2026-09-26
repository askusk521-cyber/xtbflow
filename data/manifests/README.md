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

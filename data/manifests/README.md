# Public data manifests

`public_records.v1.jsonl` is the repository-wide record manifest. It contains
262 rows: six inherited source-audit rows remain explicitly quarantined, while
256 SPICE2 OpenFF configurations are admitted as a bounded development E/F
pilot. The admitted rows cover 64 independent parent records, with four
configurations per parent and group-preserving `train`/`validation`/`test`
assignments of 48/8/8 parents (192/32/32 configurations). This is a
development corpus, not a final paper test set or a reactant-only
reaction-discovery set.

The SPICE2 source manifests are:

- `spice2_openff_pilot_20260929.jsonl`: one frozen configuration for each of
  64 selected parents;
- `spice2_openff_256_20260929.jsonl`: the same 64 parents with configuration
  indices 0, 3, 6 and 9, yielding 256 rows.

Each SPICE2 row uses a stable `source_collection` ID. The IDs
`spice2_openff_dipeptides_v1.2` and
`spice2_openff_solvated_amino_acids_v1.1` identify the upstream collection and
version without embedding a path from the machine that produced the manifest.
The upstream archive identity remains pinned by `source_file_sha256`; a local
filesystem path is not provenance.

Historical source-audit rows must retain `admission: quarantine` until their
record-level geometry, charge/spin evidence, labels, provenance and license
contract are verified. For every admitted record, keep an exact locator and
SHA-256, parent/family grouping, units, label mask, charge/spin evidence,
license record and overlap audit. Run the record contract and group-split
firewall before changing `admission` from `quarantine`.

## Track-B manifests

`track_b_reaction_ts.v0.1.jsonl`, `independent_seeds.v1.jsonl` and
`search_attempts.v1.jsonl` are intentionally empty at introduction. Empty files
make the zero-admission state explicit and prevent example or guessed records
from entering the scientific pipeline. Their contracts and current source
audit are documented in `docs/TRACK_B_DATA_GATE.md`.

A Track-B record may be added only after its real bytes, units, electronic
state, atom correspondence, product supervision, event label, transition
geometry, reference protocol and grouping are pinned. Seed records must be
frozen before search; attempt records are append-only and retain failures and
alternative channels.

The zero-state audit reports are:

- `docs/evidence/track_b_manifest_audit_20260929.json`
- `docs/evidence/independent_seed_manifest_audit_20260929.json`
- `docs/evidence/search_attempt_manifest_audit_20260929.json`

They deliberately report that model training and search may not start yet.

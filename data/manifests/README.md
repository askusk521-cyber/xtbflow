# Public data manifests

`public_records.v1.jsonl` is the repository-wide record manifest. It contains
262 rows: six inherited source-audit rows remain explicitly quarantined, while
256 SPICE2 OpenFF configurations are admitted as a bounded development E/F
pilot. The admitted rows cover 64 independent parent records, with four
configurations per parent and group-preserving `train`/`validation`/`test`
assignments of 48/8/8 parents (192/32/32 configurations). This is a development
corpus, not a final paper test set or a reactant-only reaction-discovery set.

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

# S3 stage projection

This draft depends on #181 at `721a9f2075dcc9abba819ecb91c34e256b75c44a`.
The v2 read-only transformation retains TS optimization/gradient/mode, IRC,
endpoint and graph status/reason fields and a canonical raw-evidence hash.
It leaves strict certification criteria unchanged. Missing evidence stays
unknown. Raw artifact byte hashes are distinct and supplied by the audit driver.

For historical records, run the original cache auditor from the **clean
2772c7bedb9c674144d7dab702a29178eeee699a execution source**, using the separately
pinned 721a9f2 audit driver, before applying this new projection. Never call a
modified `CertificationService.run` against an old cache: source hashes change
cache keys and could launch work. A changed reason is a versioned projection,
not a rewrite of the original verdict or ledger.

The new audit/validation envelope is CPU only: at most 600 wall/CPU seconds per
audit or test command, 256 MiB read/16 MiB new audit data, one concurrent command,
zero retries, zero training/model generation/new calculator calls. Existing
500/402 reference primitives and unresolved trained-attempt costs remain intact.
Software tests use fake evidence and cannot establish chemical efficacy.

## Review follow-up: published trained result identities

The initial audit's `original_unchanged` meant unchanged during that read; it
did not independently bind all three trained results to their earlier public
identities. The new audit recomputes canonical identity with only
`content_sha256` excluded, checks the record's own identity, then compares the
three trained candidates to the immutable terminal report at `5f5304b`.
The exact public report is retained byte-for-byte in `anchors/`, with its byte
SHA-256 pinned in the auditor. Missing/duplicate/unexpected trained candidates
fail closed. A same-verdict result whose cost or evidence changed is rejected
even if all its internal hashes were recomputed consistently.

The receipt separates `unchanged_during_audit` from
`published_anchor_verified`; the latter is required for the three trained
records. Rule-chain cache provenance remains the separate exact-2772c7b audit.
No old audit, original result, scientific verdict or consumed cost is rewritten.

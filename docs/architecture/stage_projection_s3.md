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

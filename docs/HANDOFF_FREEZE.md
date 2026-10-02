# Handoff and freeze contract

`schemas/handoff_freeze.schema.json` and `scripts/validate_handoff.py` provide
the small governance boundary shared by #75, #86 and #87.  A manifest records
the exact GitHub base/head, code/config/data/split/input-view/protocol/checkpoint
and ledger identities, the evidence level, and explicit failure reasons.  It is
an identity and readiness record; it does not execute a run or claim a
scientific result.

The validator is intentionally local.  It checks the schema contract, hashes
repository-local file artifacts, rejects unsafe absolute/private paths, and
keeps `contract_valid` separate from `ready`.  A well-formed draft or blocked
manifest is useful handoff evidence and reports `contract_valid=true,
ready=false`.  A freeze can be ready only when every required artifact is
registered, locally or externally identified with a verified SHA-256, the
evidence status is `pass`, and the GitHub ref state is `verified`.

`source_status` is required for every artifact.  `unknown`, `quarantine`,
`external_only`, and `unavailable` are preserved as blocking states; missing
hashes never become passes.  The `--offline` mode never verifies GitHub refs,
so it cannot silently turn a local declaration into a remote check.

Example command:

```bash
python scripts/validate_handoff.py \
  --config configs/runs/handoff_freeze_example_v0.1.json \
  --offline \
  --output /tmp/handoff-audit.json
```

Use `--require-ready` in a preflight gate when a blocked or draft record must
fail the command.  Without it, the command exits zero for a structurally valid
blocked handoff and writes the reasons that require upstream work.  The output
contains the normalized GitHub SHAs, all artifact hashes and source states,
`evidence_level`, `failure_reasons`, and separate `errors` and
`blocking_reasons` lists.

This contract complements the existing run ledger and workflow protocols; it
does not create a second budget database, grant merge approval, or make an
unmerged PR part of `main`.

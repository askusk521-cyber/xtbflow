# Claim, comparison, and evidence manifest

The W13 first increment is a result-blind table contract. It records what a
claim would need, which comparison and budget support it, and where evidence is
stored. It does not turn a planned, blocked, unresolved, or development result
into an observed scientific claim.

`schemas/claim_evidence_manifest_v1.schema.json` defines the machine-readable
shape. `scripts/validate_claim_evidence.py` adds semantic checks that JSON
Schema alone cannot express:

- claim IDs and comparison IDs are unique;
- scientific claim types use `unique_strict_certified_channels_per_parent` as the
  primary metric;
- an observed claim has at least one evidence reference and complete accounting;
- unresolved or blocked claims cannot be marked `ready`;
- evidence paths are relative and cannot escape the repository when path checks
  are requested.

`scripts/render_claim_evidence.py` consumes a validated manifest and emits a
deterministic JSON summary plus a small SVG status chart. The chart is a report
of the table; it does not calculate a gain, choose a method, or infer missing
chemistry evidence.

The current panel manifest may remain null while #129's eligible pool is being
generated. The first useful manifest should therefore contain planned or blocked
rows with explicit claim limits rather than placeholder success values.

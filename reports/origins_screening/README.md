# Origins screening attempt ledger

This directory stores append-only exploratory attempt records for issue #27. It is not a success-only gallery and currently contains no calculation result.

One JSON file should be written per attempt. Required fields:

- `schema`, `attempt_id`, `created_at`, `decision_use`;
- candidate registry version and `candidate_id`;
- source, input manifest, coordinate and atom-pool hashes;
- parent/family/microstate/environment group IDs;
- code commit, clean-tree state, model/checkpoint and protocol IDs;
- exact fields visible to the proposer;
- proposal source and candidate rank;
- reserved and consumed calculator/reference calls;
- wall time, retry count and failure taxonomy;
- evidence tier, stationary/mode/connectivity outcomes;
- duplicate relationship and retained/rejected reason;
- whether the attempt influenced model or application selection.

Use [`attempt_template.json`](attempt_template.json) as a machine-readable blank. Never copy a successful result into the template. Confirmation attempts belong in the separate #28 location and require a frozen confirmation manifest.

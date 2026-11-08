# Recorded policy guidance adapter v0.1

The PR C adapter consumes a policy snapshot, prior ledger summary, visible
tool feedback, and a declared action space. It returns only an event delta
through the PR A provider boundary. It cannot write geometry or mutate the
snapshot, freeze state, candidate registry, or ledger.

The policy request fingerprint includes the immutable snapshot identity, action
ID, action space, feedback, provider revision, configuration, and protocol
fingerprints. Repeating the same request through one adapter runner is recorded
as a duplicate with zero charged cost. Event edits after the freeze boundary
are explicitly rejected by the merge gate and remain in the ledger.

This implementation uses deterministic recorded feedback and does not run a
live policy loop or chemistry. Its outputs support software state-transition,
deduplication, freeze, and cost-accounting claims only.

# Guidance adapter boundary v0.1

Real physical and policy providers enter the #166 system through one boundary.
They receive an immutable `ProviderRequest` containing a snapshot identity,
lane, provider revision, configuration/protocol fingerprints, and target-free
feedback/action-space mappings. They return a `GuidanceOutput`; providers do
not mutate geometry, event, freeze state, candidate state, or ledgers.

Before invocation, the runner asks for an estimated `GuidanceCost` and reserves
that amount in `GuidanceBudget`. After invocation it measures the actual cost,
reconciles the reservation, and writes an `AdapterLedgerEntry`. Reservation
failure prevents invocation. Timeout, invalid output, execution failure,
duplicate request, and budget exhaustion each have explicit statuses and
failure stages.

All identity fields are lowercase hexadecimal SHA-256 values. The request
fingerprint includes snapshot, lane, provider revision, provider/configuration
identity, protocol identity, feedback, and action space. A repeated request is
not invoked or charged again. The merge gate remains the only place where a
provider output can change generation state.

The checked-in mock physical and policy providers are deterministic fixtures.
They prove only the boundary, reservation, reconciliation, and failure
accounting behavior. They make no physical or chemical claim and are not
substitutes for #136 or #139 adapters.

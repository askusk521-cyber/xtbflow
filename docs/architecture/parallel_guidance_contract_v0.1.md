# Parallel guidance contract v0.1

This contract defines the calculator-free first layer of #166. A generation
step begins with one immutable `GenerationSnapshot`. The base, geometry, and
event lanes all receive its `snapshot_id` and return bounded `GuidanceOutput`
records. The deterministic merge gate sorts outputs by the declared lane
priority, applies geometry trust-radius and collision checks, rejects edits
after the event freeze boundary, projects the combined event delta through the
masked conservation constraint, and then recomputes event velocity from the
merged geometry and event.

Every output produces one `GuidanceLedgerEntry`, including failed, rejected,
duplicate, and budget-exhausted actions. A duplicate output has zero charged
cost. The ledger is immutable and can be serialized with the decision and
next snapshot for replay.

The recursive firewall rejects target, product, transition-state, reference,
label, energy, force, and related derived fields in metadata and provenance.
This first layer does not call xTB, CP2K, a GPU calculator, or a model
provider. Synthetic providers and the five-mode offline replay belong to the
next stacked PR. The contract therefore supports software state-transition and
accounting claims only; it does not support chemistry, accuracy, performance,
or generalization claims.

The JSON representation is described by
`schemas/parallel_guidance_contract.schema.json`.

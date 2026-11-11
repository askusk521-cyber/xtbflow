# Live physical-feedback loop v0.1

This is the first bounded runtime loop that calls the existing xTB oracle on
the current immutable generation snapshot. It is a software/proxy experiment,
not a reaction-discovery or chemical-validation result.

The fixed experiment has one reactant-only water parent and one seed. Both
arms use the same input, GFN2-xTB protocol, and one calculator-call budget:

```text
no_feedback
    snapshot -> xTB energy_forces (charge cost) -> snapshot unchanged

physical_feedback
    snapshot -> xTB forces -> bounded geometry guidance
             -> force-derived event delta -> merge gate
             -> next immutable snapshot
```

The provider reads only the current snapshot geometry and the result of the
same xTB call. It returns `GuidanceOutput` through the existing adapter
boundary. The merge gate enforces the geometry trust radius, collision guard,
event freeze, and one conservation projection. Adapter and runtime ledgers
retain estimated/measured/charged cost, provider/protocol fingerprints,
failure stage, snapshot identity, and the changed geometry/event state.

The downstream diagnostic gate is opt-in through an explicit `certification`
configuration block. With the block absent, the runner records
`MISSING_TS_HESSIAN_IRC_EVIDENCE` and keeps the original
`status=blocked_certification` artifact; it never silently selects a backend.
When enabled, each arm receives its own durable budget and the shared xTB
protocol is used for finite-difference Hessians, short Cartesian descents and
endpoint relaxations. These are proxy diagnostics, not independent IRC or
verified endpoint minima. The corrected policy records `attempted_unresolved`
and counts zero strict certified channels even if local gradient/mode checks
pass. The historic R0 `CERTIFIED` claim is withdrawn in
[the evidence review](../CERTIFICATION_REVIEW_20261110.md).

The run artifact is
`artifacts/real_feedback_run_20261110T020000/`. Its claim levels are:

- `software_demo`: immutable snapshot, merge gate, replay and accounting;
- `proxy_metric`: one real GFN2-xTB force/energy call per matched arm;
- `strict_validation`: required for TS/IRC and any chemistry claim.

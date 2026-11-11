# Bounded xTB path diagnostics (certification policy v2)

**Correction, 2026-11-10:** the earlier `CERTIFIED` verdicts from this module
do not satisfy the independent TS/IRC contract in issue #137. See
[the evidence review](../CERTIFICATION_REVIEW_20261110.md). Original records
are retained as audit history, not accepted strict-validation evidence.

This adapter is the next certification slice after the live physical-feedback
loop. It is deliberately a downstream validator and does not generate a
candidate or read a product, reference TS, sealed label, or target-selected
environment.

The input is one explicit `MolecularSystem` snapshot plus a durable
`CalculatorCallToken`. Coordinates are Angstrom, energies are Hartree, forces
are Hartree/Angstrom, and the finite-difference Cartesian Hessian is reported
as unweighted Hartree/Angstrom². Every energy/force evaluation consumes one
token before entering the calculator. The result retains the protocol and
calculator build identity, input hash, call count, endpoint gradients, and
failure reason.

The bounded sequence is:

```text
current candidate snapshot
→ xTB energy/forces
→ finite-difference Cartesian Hessian
→ exactly-one-negative-mode validator
→ ± bounded gradient-following path
→ ASE endpoint minimization using the same calculator protocol
→ provisional distance graphs for both actual endpoints
→ attempted_unresolved / INCOMPLETE_IRC_CERTIFICATION
```

Endpoint optimization is an optional ASE boundary. If ASE is unavailable, the
result is `blocked_backend`; calculator errors and exhausted tokens remain
`execution_failed` or `budget_exhausted` with consumed calls retained. The
runner records wall seconds, CPU seconds/hours, GPU hours, source identity,
configuration hash, input hash, and replay identity. OpenMP/BLAS threads are
fixed to one in the smoke runner so CPU accounting is reproducible.

The Cartesian descent followed by BFGS relaxation does not implement an
independent bidirectional IRC. Low endpoint gradients do not establish
minima without endpoint Hessian/mode checks. Graphs derived from the candidate
geometry cannot stand in for the input reactant, and the lower-energy endpoint
cannot be designated the product by its energy alone. Actual endpoint distance
graphs remain `proxy_metric`, with no assigned reactant/product roles and
`observed_event=null`. Strict certified channel count remains zero.

`CERTIFICATION_POLICY_REVISION` participates in the configuration/cache identity
so old success records cannot be reused under the corrected policy. Numeric
thresholds, fixture, baseline and denominator are unchanged. The local
stationary/mode diagnostics are retained, while `path_connected=false` and
the missing independent checks are recorded explicitly.

Both the water fixture and the reactant-only approach remain
`blocked_certification` until the independent reference chain is integrated.
Passing this module's software tests cannot discharge that gate.

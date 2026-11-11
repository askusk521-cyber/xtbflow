# Recorded physical guidance adapter v0.1

The PR B adapter consumes a `PhysicalFeedback` record bound to one immutable
snapshot. Feedback may contain forces, a reaction direction, and curvature
visible at that step. It cannot contain target, reference-TS, label, or other
derived fields. The provider returns only a bounded geometry `GuidanceOutput`.

The adapter is executed through the PR A boundary: estimate cost, reserve the
budget, invoke the provider, measure actual cost, reconcile the reservation,
then pass the output to the merge gate. The off/on ablation uses the same
snapshot, candidate pool, seed, and budget. The enabled path changes geometry;
the merge gate recomputes event velocity from the post-update geometry. The
disabled path is an identity baseline.

This is recorded-feedback software only. The deterministic provider does not
call xTB/CP2K or any calculator and does not establish a physical or chemistry
gain. A later adapter may replace it only after preserving the same boundary,
reservation order, provenance fields, failure ledger, and claim limits.

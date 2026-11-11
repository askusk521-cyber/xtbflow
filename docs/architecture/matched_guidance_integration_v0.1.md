# Matched guidance integration v0.1

PR D fixes the parent denominator, candidate-pool identity, seed, provider
feedback, and total budget across five synthetic arms:

- `none`;
- `serial_physical`;
- `parallel_geometry_event`;
- `policy_on`;
- `policy_off`.

Every parent receives every arm. Rows retain geometry/event changes, model
forward cost, calculator cost, wall cost, adapter ledger entries, merge
rejections, and explicit status. The parallel arm runs physical and policy
providers from the same snapshot and merges both outputs once; serial physical
uses only the geometry lane; policy on uses only the event lane.

This is a matched software comparison with recorded feedback. It does not run
real calculators, certify channels, or support chemistry superiority. Real
parent-level certification, unresolved accounting, paired intervals, and
stopping/retry protocols are required before any #140 scientific comparison.

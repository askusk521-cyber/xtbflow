# Parallel guidance offline replay v0.1

The second #166 layer runs the same immutable initial snapshot, candidate-pool
identity, random seed, synthetic providers, and total budget through five
deterministic modes:

1. `none` — base lane only;
2. `geometry_only` — base plus geometry guidance;
3. `event_only` — base plus event guidance;
4. `parallel_geometry_event` — base, geometry, and event guidance from the
   same input snapshot, merged once by the deterministic gate;
5. `serial_baseline` — base, geometry, and event guidance in sequential
   substeps, with each later lane receiving the preceding immutable state.

Each artifact records the input snapshot, every lane output, merge decision,
ledger entries, final state, total cost, and claim limits. Synthetic providers
are deterministic and calculator-free. The parallel mode changes both
geometry and event in the next state; the event delta is projected through the
masked conservation constraint after the lane outputs are merged. Budget
exhaustion and rejected actions remain in the artifact instead of disappearing.

This replay proves software state transitions, ablation comparability, and
accounting. It does not establish chemical accuracy, physical usefulness,
performance, or generalization. Real physical and policy adapters remain
separate follow-up work.

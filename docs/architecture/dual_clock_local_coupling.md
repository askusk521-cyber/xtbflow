# Dual clocks and local coupling (W03 / #133)

`DualClockJointFlow` is a new trainable extension of the shared event/geometry
field. It carries separate normalized generation coordinates `t_b` (event) and
`t_x` (geometry), with zero-offset maps so the synchronous case is exactly the
existing `JointEventGeometryFlow` field. The schedule records event freezing,
the coupling window, and the actual grid; a step crossing a boundary is rejected
instead of silently changing the control halfway through an update.

The default software configuration uses an event power of 2, a geometry power of
1, event completion at 0.75, and a coupling window [0.2, 0.8). These are
normalized sampling coordinates, not physical time or a fitted scientific
setting. Event updates remain in the declared conservation subspace after local
gating; geometry updates retain the pairwise SE(3)-equivariant construction.

The local gate is derived only from the current candidate and its reactant event:
an invariant distance contact term and an absolute BE change term. It has no
reference product, TS coordinate, mode, endpoint, atom-ID lookup, or post-hoc
selection. A global gate is available as the exact comparison control. Both
cross-branch messages are gated before the event projection and geometry field;
the trajectory report records message norms, clock coordinates, conservation
residual and model-evaluation count.

The first increment is software-only. `configs/models/dual_clock_runtime_v0.1.json`
is explicitly marked `software_prototype_untrained`; no checkpoint, real
calculator call, certified channel, or performance claim is attached. Training
must use a frozen task view, parent/family split and an independent validation
protocol before any real chemistry comparison. A future matched-budget run must
compare synchronous, global-local, one-way and bidirectional controls under the
same source, weights and physical call ledger.

Example bounded smoke usage on a clean checkout:

```bash
PYTHONPATH=src:vendor/mechai_reusable python -m pytest -q tests/test_dual_clock.py
```

The test uses synthetic tensors to verify clocks, boundary handling, local
state dependence, conservation and finite trajectory accounting. It does not
validate reaction accuracy or generalization.

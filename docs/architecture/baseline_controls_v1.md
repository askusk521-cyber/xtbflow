# Baseline control directory v1

`configs/models/baseline_controls_v1.json` and
`xtbflow.models.BaselineFactory` define one software boundary for the W02
controls. Every learned arm is built with fresh parameters. The first four
learned controls reuse the existing field paths: `shared_no_exchange` maps to
`both_off`, `step_serial` advances the event state before geometry,
`current_one_way` disables only geometry-to-event feedback at the current
state, and `current_two_way` keeps both messages. The no-projection arm keeps
its initial state and field settings while disabling only the conservation
projection; it is recorded as a mechanism intervention.

`separate_independent` contains two distinct parameter spaces and exchanges no
generated state. `full_two_stage` contains distinct event and geometry stage
modules; it freezes the predicted event state before evaluating geometry.
Their actual parameter counts and state hashes are reported by
`BaselineFactory.manifest`, so a later training runner can reject or label a
capacity mismatch. The manifest also records raw candidate caps, no-refill
semantics and field-evaluation counts.

`strong_rule` is a rule-only arm. `RuleGeometryInitializer` perturbs only the
reactant coordinates along declared rule bond edits with an explicit maximum
atom displacement. It performs zero calculator calls and returns
`unsupported` when no reactant-side rule can propose an edit. A geometry guess
is not a transition-state or pathway claim.

The factory is a software contract. Separate training, checkpoint identity,
physical validation and scientific comparisons remain the responsibility of
the downstream runner; this package contains no training or QC execution.

# Scale, data, and blind-generation protocol

This protocol turns the current joint-versus-serial result into a testable
hypothesis. It is a planning contract; it contains no scientific result and
remains blocked until an admitted sample-level event/transition-state manifest
is available.

The machine-readable files are
[`joint_flow_scale_ladder_v0.1.json`](../../configs/models/joint_flow_scale_ladder_v0.1.json)
and
[`joint_scale_data_matrix_v0.1.json`](../../configs/runs/joint_scale_data_matrix_v0.1.json).

## Capacity and fairness

The current graph keeps node, condition, and radial inputs fixed and changes
only `hidden_dim`. The declared widths are 64, 128, 256, and 512, with
23,387, 79,515, 290,075, and 1,104,411 trainable parameters respectively.
The count is recomputed from the constructor and must also be recorded from the
checkpoint. A serial run includes the complete event and geometry parameter
state; an inactive control path is not counted as effective capacity.

The primary factorial matrix has three arms (`both_off`,
`serial_independent`, and `joint_bidirectional`), three nested training-parent
sizes (25, 100, and 400), and three seeds (11, 17, and 23). Its 81 runs are a
starting design, not a guarantee of statistical sufficiency. The 512-width arm
is an extension after the primary matrix under the same frozen protocol. The
current-state one-way and unconstrained controls are diagnostic runs at a
representative point so that serial timing, feedback, and conservation effects
remain separable.

Parent identities, reaction-family labels, source identity, and overlap audits
must be frozen before training. Nested subsets are selected by a public hash of
the parent ID and a fixed seed; labels and validation scores cannot affect the
order. Metrics are averaged per parent before confidence intervals are formed.
Training-seed variability and parent-resampling variability are reported as
different quantities.

## Training diagnostics

Each run records event-only, geometry-only, and total gradient norms before
clipping. The global policy clips the complete model at the declared norm. The
branchwise control clips event-only, geometry-only, and shared encoder
parameters separately. Shared parameters are kept as their own group because
their gradient is the sum of both objectives. A change in the architecture
ordering that disappears under branchwise clipping is an optimization finding,
not evidence for or against the coupling hypothesis.

## Blind generation

The primary score is produced from reactant-side inputs and a frozen model
without reference event, TS geometry, product, or downstream labels. A repeated
attempt is valid only when the protocol declares an actual supported stochastic
source: event noise or geometry noise. Latent inputs are not implemented and
are rejected during protocol validation. Different integer seeds by themselves
do not create different candidates for a deterministic model. Candidates are
decoded, checked for legality, and de-duplicated by metric scope: geometry
metrics use geometry slots, while event metrics use geometry-plus-event slots.
Empty, failed, invalid, and duplicate attempts remain in the parent accounting;
label-assisted event and geometry errors are retained for diagnosis only.

The executable boundary is the target-free `xtbflow-reactant-input/v1` JSONL
contract in [`schemas/reactant_input.schema.json`](../../schemas/reactant_input.schema.json).
`scripts/run_reactant_generation.py` reads only this manifest and a frozen
checkpoint; `scripts/score_reactant_generation.py` is a separate post-hoc stage
that may open reference labels. `label_replacement_invariant` checks that
replacing reference labels leaves candidate fingerprints unchanged. A repeated
attempt must use `GenerationProtocol` with an explicit event or geometry noise
source; nonzero `latent_dim` and a `latent` policy fail before model execution.
Different seeds alone are not counted as samples.
`generate_repeated_candidates` records actual unique geometry slots and never
fills an advertised cap with copies of a deterministic trajectory.

The scale planner (`scripts/plan_joint_scale_runs.py`) materializes the 81 run
records only after an identity-only parent manifest passes the admission gate.
Without that manifest it emits a durable `blocked` plan and performs no model
training or calculator call.

Failure accounting distinguishes `not_run`, `execution_failed`,
`settlement_failed`, `report_write_failed`, and `budget_exhausted`; consumed
calculator calls are retained for every settled attempt. Physical refinement is
a separate matched-budget stage after generator-only results are frozen.

The headline claim remains conditional: joint coupling can be promoted only if
it exceeds the complete serial control on a frozen parent-group test set under
the same candidate, integration, and physical budgets. A crossing learning
curve is evidence for a data-dependent hypothesis, not proof of universal
architectural superiority.

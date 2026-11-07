# Development task views (W02 / #132)

`task_manifest.py` projects an existing audited Track-B source into event-only,
geometry-only and paired-joint supervision. It preserves the original admission,
parent/family grouping, input identity, geometry evidence and claim limits.
Quarantine and confirmatory records are excluded. This is not a replacement for
source admission, atom-map audits, public licensing or overlap checks.

The index records each source row, source/record/input hashes, split, tasks and
distinct parent/family counts. Its content is rebuilt from the original manifest
before loading; modified permissions or invented task eligibility are rejected.
The three views overlap: do not add their counts as independent data. An explicit
empty `energy_force` list is required for this source because the endpoint/TS
adapter does not supply matched-geometry E/F labels. It does not manufacture zero
forces, reference energies or Delta targets.

The index preserves the source protocol's environment declaration and explicitly
marks it unresolved for matched-geometry pairing. It does not infer a gas/solvent
model from the dataset name. Charge, multiplicity, units, mapping and protocol
hashes remain explicit; E/F use requires a separately verified environment.

`load_task_views(index_path, source_manifest_path, cache_root, task=..., split=...,
purpose=...)` reconstructs arrays from the existing private source cache through
`load_paired_pilot`. All source asset, row and geometry hashes are checked there.
Each `TaskBatch` separates `reactant_input` from `targets`; `training_views()`
provides the established `reactant_input` / `training_supervision` interface.
Training purpose requires the train split. Evaluation batches cannot be converted
to training supervision. Existing `ReactionQMRecord` quarantine is unchanged.

`scripts/audit_task_views.py --cache "$SOURCE_CACHE" --out "$NEW_OUTPUT"` runs
only on n2 from a clean commit. The source manifest is fixed to
`data/manifests/paired_pilot_v1/manifest.jsonl`. It reconstructs usable counts,
checks real training batch shapes, and selects the first three sorted distinct
training parents for hidden-label substitution. With reactants, model weights,
seed and inference settings fixed, product graphs/coordinates, event identity
and TS coordinates/identity change while inputs and generated attempts must remain
identical. The random fixed model is an input-isolation check, not a trained
model benchmark. No reference modes enter this source adapter or input boundary.

Run budget: at most 600 wall seconds, 600 CPU seconds, 16 MiB per output file,
three input-invariance cases, two candidates × two integration steps × two
counterfactual views each (24 flow evaluations), zero training, GPU or calculator
calls. Use a new output directory; failures retain their own report. Exact source,
environment, counts and output hashes belong in a dated evidence file after the
real replay. Raw geometries remain on n2; the public index contains identifiers,
hashes and counts only.

Remaining #132 work: partial-label-only source admission, E/F and same-geometry
multi-fidelity ingestion, legal declared conditioning for #134, cross-source and
reverse-reaction grouping, and versioned promotion of certified development
discoveries with confirmation isolation. The existing Diels–Alder source provides
development diagnostics and declared reactant endpoints; this increment creates
no independent confirmatory dataset or chemical validation result.

# Independent reactant pilot v1

This package freezes reactant-side seeds before a model or quantum-chemistry
driver is allowed to run. A seed contains explicit CHNOS atoms (including H),
coordinates in angstrom, integer charge, singlet multiplicity, microstate,
conformer, approach and environment identifiers, plus source and license
evidence. Every seed belonging to one parent keeps the same atom inventory and
electronic state. Unknown charge, spin, source, or license evidence remains a
blocker; it is never filled with neutral or singlet defaults.

The input contract rejects product, TS, IRC, reference-mode, target-selected
water, energy and force fields. The generated `manifest.json` and
`manifest.jsonl` are immutable snapshots identified by `manifest_hash`; later
seed additions require a new freeze directory. The freeze command performs no
QC calls:

```bash
python scripts/build_independent_seeds.py \
  --config configs/seeds/independent_pilot_v1.json \
  --freeze \
  --output "$XTBFlow_RUN_ROOT/independent-seeds-v1"
```

The checked-in configuration is intentionally `blocked` because no
source-qualified public seed bytes are stored in the repository. It is useful
for contract and dry-run testing, but it is not a scientific input set. A
dry-run preserves the block and emits a bounded plan without touching a
calculator:

```bash
python scripts/run_independent_pilot.py \
  --config configs/seeds/independent_pilot_v1.json \
  --dry-run \
  --output "$XTBFlow_RUN_ROOT/independent-plan"
```

`--execute --resume` refuses to proceed until a frozen manifest, qualified
driver, and shared ledger budget are all present. Proposals, attempts and
validation are separate records in `xtbflow.data.attempts`; failures retain
their stage and reason instead of becoming a chemical negative label.


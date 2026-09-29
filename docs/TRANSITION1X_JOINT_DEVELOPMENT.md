# Transition1x joint-flow development run

The repository now contains a bounded real-data development runner:
`scripts/run_transition1x_joint_development.py`. It is intentionally separate
from the Track-B admission path. The runner hash-gates the public
`train_rpsb_all.pkl` asset, loads it through the NumPy-only restricted
deserializer, derives a binary endpoint edit diagnostic from reactant/product
distances, and trains the current joint-flow prototype on reactant-to-TS
coordinate displacements.

The source audit remains authoritative. Transition1x has no explicit formal
charge or multiplicity fields, no source-supplied event labels, no certified
atom mapping or family holdout metadata, and no explicit redistribution
license in the pinned record. The runner therefore writes
`data_status=quarantine_development_only`, zero-fills the unresolved condition
features, and does not create Track-B records or independent-seed records.

The controlled comparison uses one trained parameter state and evaluates the
registered `both_off`, `serial_independent`, and `joint_bidirectional` paths on
the same held-out rows. It reports software/development diagnostics only. The
event label is explicitly product-derived and the result cannot support a
product-free discovery, chemical accuracy, or architecture-leading claim.

On n2, the bounded 32-record/2-step run completed on commit
`190118dc25fa334e82cf75d1dd6a4f653a3a2835`; the complete machine-readable
report is
[`transition1x_joint_development_20260930.json`](evidence/transition1x_joint_development_20260930.json).
The deterministic selection was 24 train, 4 validation and 4 test records,
with a maximum of 9 atoms. All 10,073 source records passed the structural
source audit before selection.

Run the same bounded experiment on n2 with:

```bash
PYTHONPATH=src:vendor/mechai_reusable \
python scripts/run_transition1x_joint_development.py \
  --asset /home/lhshen/.cache/xtbflow/transition1x-rpsb-13119869/train_rpsb_all.pkl \
  --max-records 32 --steps 2 \
  --output docs/evidence/transition1x_joint_development_20260930.json
```

The next scientific gate is still a separately sourced, state-resolved,
independently mapped reaction/TS corpus. This development path is useful for
testing the real-data adapter and control wiring while that gate remains
closed.


# Transition1x endpoint geometry baseline

`scripts/run_transition1x_geometry_baseline.py` evaluates the published
Transition1x transition-state guesses against the audited reference TS rows.
The exact asset is hash-gated before restricted deserialization. The evaluator
keeps the published atom row order and uses a proper-rotation Kabsch alignment
plus all-pair distance MAE.

The 9,000 `use_ind` rows are retained as the source's published split and the
1,073-record complement is reported as a diagnostic split. It is not an
independently audited reaction-family holdout. No bonds, events, charge,
multiplicity, or mechanism are inferred.

The n2 run is recorded in
[`transition1x_geometry_baseline_20260930.json`](evidence/transition1x_geometry_baseline_20260930.json).
On the 1,073-record complement, the published ordinary guesses reached the
following mean atom-mapped RMSD / pair-distance MAE in Å:

| Guess field | RMSD | Pair-distance MAE |
| --- | ---: | ---: |
| `ts_guess_sbv1` | 0.2485 | 0.0957 |
| `ts_guess_NEBCI-xtb` | 0.3526 | 0.1483 |
| `ts_guess` | 0.4746 | 0.2244 |

`ts_guess_true` is near-identical to the reference TS and is retained only as
an oracle-like leakage diagnostic. It is never treated as an independent input.
These numbers establish a reproducible endpoint-conditioned geometry baseline;
they do not establish reactant-only discovery, chemical accuracy, IRC success,
or a model advantage.

Run on n2 with:

```bash
PYTHONPATH=src:vendor/mechai_reusable \
python scripts/run_transition1x_geometry_baseline.py \
  --asset /home/lhshen/.cache/xtbflow/transition1x-rpsb-13119869/train_rpsb_all.pkl \
  --output docs/evidence/transition1x_geometry_baseline_20260930.json
```


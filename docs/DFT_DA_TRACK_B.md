# DFT Diels--Alder Track-B development source

Issue 58 requires reconstructible reactant input, mapped event supervision,
transition-state geometry, electronic state, parent/family grouping, and
provenance before real C/D/E training.  The public Figshare version-5 archive
at [article 29118509](https://figshare.com/articles/dataset/Diels-Alder_reaction_space_for_self-healing_polymer/29118509)
is CC BY 4.0 and contains 1,580 DFT rows with mapped reaction SMILES and
reactant/product/TS XYZ files.  The archive is downloaded only to the n2 cache;
its MD5 is `d37dbd26be16b63038537f8f04868fad` and its SHA-256 is
`4df57289edecdfc518da6aab147b704745a9f42901893671aab05a3f72bd73ed`.

`src/xtbflow/data/dft_da.py` retains a row only when the CHNO atom scope is
met, the TS log explicitly reports neutral singlet (`Charge = 0,
Multiplicity = 1` and `--chrg 0 --uhf 0`), the source terminates normally, and
the mapped reactant/product graphs agree with connectivity reconstructed from
the corresponding XYZ files.  This graph check is the atom-map evidence;
matching element sequences alone is rejected.  The reactant endpoint is
marked `declared_reactant_endpoint`, so these records are development-only.
The product, event, and TS fields are never present in the model-visible
reactant view.

The first audit yielded 201 rows across 41 parent groups.  State evidence comes
from both same-directory monomer Gaussian logs; transition-state atom order is
derived under the published TS-tools ordering contract (commit
`c1ba9cdd124f1fc4bbef4fee8cdd410e5703cfdc`) and is labelled
`derived_under_contract`.  Family IDs are derived from canonical
mapped-reactant components and family groups are kept whole across
deterministic 70/15/15 train/validation/test splits.  The adapter will
quarantine any row that fails a later check rather than guessing missing state
or correspondence.

## Replaying the bounded training

Run from a clean commit on n2, with the public archive already in
`~/.cache/xtbflow/dft-da-full-29118509-v1`:

```bash
cd /home/lhshen/xtbflow/xtbflow
git rev-parse HEAD
source ~/miniconda3/etc/profile.d/conda.sh
conda activate xtbflow
PYTHONPATH=src:vendor/mechai_reusable python scripts/run_dft_da_joint_training.py \
  --cache-root ~/.cache/xtbflow/dft-da-full-29118509-v1 \
  --output ~/xtbflow-runs/dft-da-joint-20260930 \
  --runtime-config configs/models/joint_flow_runtime_v0.1.json \
  --epochs 12 --batch-size 8 --max-wall-minutes 110 \
  --seeds 11 17 23 --device cuda
```

The entry point records the reactant-only strong-rule diagnostic and trains
conserved-independent, serial, same-integrator one-way, bidirectional joint,
and unconstrained projection-toggle controls from the same initial state for
each seed.  It writes the manifest, optimizer checkpoints, per-parent held-out
diagnostics, configuration hash, checkpoint hashes, environment, split/leakage
audit, and resource ledger under the output directory.  The run is bounded to
one job and 30 wall minutes; no xTB/CP2K refinement is part of this stage.  Results must
remain labelled development evidence until the independent input and physical
refinement stages in issues 45, 56, 57, and 58 are complete.

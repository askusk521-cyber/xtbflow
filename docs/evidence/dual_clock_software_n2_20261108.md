# Dual-clock software validation — 2026-11-08

The clean checkout `c16577b1a21be14f8e4bfbeaca828ce2779290a5` was tested on
`node2` in conda environment `xtbflow` (Python 3.10.19, PyTorch 2.10.0+cu128).
The frozen configuration hash is
`06e95997dce4a35ef475eeb33e85461ec8f1faa38ac436113d7693e319c94ca6`.

The focused dual-clock suite passed 5 tests. The existing joint-flow suite plus
the new suite passed 17 tests. The bootstrap inventory verified 387 files. The
checks used synthetic tensors and made zero GPU or calculator calls. The exact
commands and machine-readable record are in
`dual_clock_software_n2_20261108.json`.

This is a software contract result. The configuration is explicitly an
untrained prototype; no chemical accuracy, certified channel, training result,
or generalization claim follows from this run. Scientific adoption still needs
frozen data, matched physical budgets, independent controls, and certification.

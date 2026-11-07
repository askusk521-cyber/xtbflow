# Dual-clock software validation — final 2026-11-08

The exact clean head `d8a44da79aa3004eb8b035a3d8283507ea5546e3` was cloned into
`~/xtbflow-runs/w133-dual-clock-20261108-final2` on `node2` and run with conda
environment `xtbflow`, Python 3.10.19 and PyTorch 2.10.0+cu128. The complete
pytest suite passed 369 tests and 68 subtests; the reusable unittest suite passed
56 tests; and the bootstrap inventory verified 389 files. One existing warning
in `test_energy_forces.py` remains.

The run used synthetic/software fixtures, no GPU, and zero calculator calls.
The dual-clock configuration remains explicitly `software_prototype_untrained`.
The result therefore validates implementation and accounting contracts only; it
does not establish reaction accuracy, certified channels, training quality,
generalization, or an advantage over a control.

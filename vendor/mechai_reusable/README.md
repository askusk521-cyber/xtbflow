# Reusable mechai contracts

This directory contains a deliberately small source snapshot copied from the
mechai project on 2026-09-26. It is a reusable reference implementation, not a
claim that the xtbflow method is already trained or chemically validated.

Included contracts:

- explicit-H CHNOS Lewis bookkeeping and conserved event application;
- bounded proton/water relay proposal enumeration;
- task/provenance manifest validation with explicit unknown and blocked states;
- the untrained event--geometry compatibility prototype and label-free decoder.

The code keeps the original `mechai` import namespace so its existing tests can
run with `PYTHONPATH=vendor/mechai_reusable`. The source was copied from a
dirty working tree; the exact source-relative files and SHA-256 values are in
`docs/MECHAI_REUSE_MANIFEST.json`.

These modules intentionally do not provide a quantum-chemical energy, a force,
a transition-state guarantee, a reaction probability, or a trained checkpoint.

The event/proposal/manifest tests run with the standard library. The
event--geometry compatibility model and decoder additionally require PyTorch;
those tests were not run in the current CPU environment because `torch` is not
installed.

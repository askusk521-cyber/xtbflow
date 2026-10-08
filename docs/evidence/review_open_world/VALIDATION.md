# Verification and resource record

## Frozen sampling and provenance

- Branch base: codex/v1b-20261007 at 5998dfd.
- Sample selector pushed at 8944f0c; exact sample configuration pushed at 6507aaa before any quantum calculation.
- Chain source: 621b900, clean pushed clone. Analysis/report source: 7521aeb.
- Independent sample re-selection from the stated inputs using source-a2f79c2 and `cmp configs/review/open_world_o1.json open-selection-recheck.json`: exit 0.
- Positive controls executed first: 17/20 passed, so the >=50% gate permitted OUTSIDE interpretation. All 120 frozen chains have verdicts; no failed chain was silently discarded or replaced.

## Physical and software checks

- Central-difference quadratic Hessian test passed.
- Optimized water check from source-dcb16bb passed: exact Hessian symmetry, six-dimensional translation/rotation span, all three internal modes positive (hessian_check.json).
- Pure `qc_protocol` harmonic/classification/identity functions are imported unchanged; the DFT engine is never invoked.
- Source-a2f79c2 bootstrap and both complete test suites exited 0. Final evidence-commit tests and byte-identical analysis comparison are recorded in the final PR update.

## Actual resources

Main chain process: CPU user 2760.74 s + system 10.94 s = 2771.68 s, or 0.769911 core-hours. Wall 3016.71 s (50:16.71). CPU was directly executed under the owner's explicit amendment; no Slurm job, GPU use or DFT computation. The small water physical check, tests, imports and installation are additional preparation costs, not included in the chain-process total; the 50-core-hour cap is far above measured use.

The single process ran within the original two-hour job limit. Every chain records wall time and energy/gradient-call counts; totals may differ from process wall time because loading, logging and sample joins are outside chain timing. Environment was cloned into the owned run root, not installed into existing conda environments.

## Evidence limits

A strict xTB graph-connected saddle is not a DFT-certified channel. Endpoint frequency gray zones are treated as endpoint failures, not silently accepted. EVALUATION_UNRESOLVED includes explicitly recorded numerical/perception exceptions. Sample selection is deterministic and limited to one training seed and B0; Wilson intervals do not establish performance for all model outputs. The proposal is not permission to run DFT.

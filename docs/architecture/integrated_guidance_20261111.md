# Integrated guidance entry — #197

The common implementation is `xtbflow.sampling.integrated_guidance`. Its input
is an explicit mapped reactant query plus current BE/geometry/clocks. One forward
observes the old predicted endpoint, then constructs the ordinary Euler base
with updated BE and clocks. A guide receives both identities and changes only
geometry. The following normal forward consumes that geometry and updates BE.
A final decoder returns a chemical BE matrix or an explicit invalid result;
`candidate_from_state` passes it through the common mapped candidate contract.
Unfinished execution, unknown termination or unsettled costs block certification.

Coordinates are Angstrom, E/F are Hartree and Hartree/Angstrom, clocks are
dimensionless. Charge, multiplicity, atom IDs, reactant BE, source evidence and
gas condition must be explicit. No product labels enter this interface.

The bounded executable traverses these same interfaces using a synthetic
coupled model, synthetic chemical-BE decoder and fake reference backend:

```bash
PYTHONPATH=src:vendor/mechai_reusable python scripts/check_integrated_guidance.py --mode fake --out /tmp/xtbflow-new-integration-fixture
PYTHONPATH=src:vendor/mechai_reusable python scripts/check_integrated_guidance.py --mode cached --out /tmp/xtbflow-new-integration-fixture
```

Use a new output directory. The config is
`configs/research/integrated_entry_20261111.json`: two synthetic forwards, one
synthetic observation/update, at most two fake reference calls, 60 s wall,
0.02 core-h, 0 GPU-h and 0.01 GB. Cached mode retains the candidate and cumulative
reference ledger with zero new reference calls. Neither the synthetic BE decoder
nor the historical four-score rule fixture is a trained chemical decoder.

The real trained model/decoder remains isolated at
`e84e330390a519bf4ac12048b297403c09717646`; S0 asset file hashes are retained in
`configs/research/narrowed_assets_20261111.json`. Later stages must verify that
source, checkpoint and saved tensors and use a new stage permission manifest.
#197 authorizes zero trained forwards, zero E/F and zero reference primitives.

## Dependency selection

`configs/research/integration_sources_20261111.json` maps each port to its exact
source commit and original file hash. #182 was merged once, including #175.
S0 (#190), latest S1 (#194), S2/S3 provider/rotation/accounting fixes (#191/#193),
reference/bridge/audit closure (#181/#195), and only #196's own report/evidence
changes are combined. Historical branch histories are not wholesale merged.
The older PVO script in #192 is not part of this entry and remains on its branch.
The newer #194 entry checks take precedence over #196's old parent.

All old 90-forward receipts and original scientific results remain byte-pinned.
Historical 468/638/526 and 500/402 results are read-only evidence; their pinned
original auditors must be used for cache verification, never a new service run.
Bootstrap inventory is regenerated from the selected staged tree.

## Operational authorization

The owner authorized this integration and in-scope fixes/merges on 2026-11-11:
https://github.com/askusk521-cyber/xtbflow/issues/129#issuecomment-6789105000.
`configs/resources.yaml` records this scoped waiver; other scopes retain the
standing policy. Legacy uncommitted work is preserved in the original worktree.

Software validation and CI receipts are recorded separately after exact-source
execution. No numerical, chemical or method-advantage claim follows from this
integration. #198 remains gated on merged tested source and retained identities.

## Exact-source validation

On node2, source `aa01c85580e1b5fcdfb80915c1f97b8ee9970a33` passed 42 focused
tests, full pytest (419 tests, 68 subtests), 56 reusable tests, inventory (495
files), diff check and executable fake/cached entry. The first fake entry used
one synthetic reference primitive; cache replay used zero additional calls.
The immutable receipt is `docs/evidence/integration_197_20261111/software_aa01c85.json`.
The earlier 418-pass/1-failure run is separately preserved and never relabelled.
No new scientific calculation or trained checkpoint forward occurred.

Reproduce in a fresh branch-named clean n2 worktree at the SHA above. Activate
`~/miniconda3/etc/profile.d/conda.sh` and `conda activate xtbflow`; use
`CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PYTHONPATH=src:vendor/mechai_reusable`. Reserve `srun --partition=main --nodes=1
--ntasks=1 --cpus-per-task=1 --mem=8G --time=00:10:00`, wrap the validation in
`timeout 598`, and retain logs in a new output directory (128 MiB cap). Run
`python -m pytest -q`, `python -m unittest discover -s tests/reusable -p
'test_*.py'`, `python scripts/validate_bootstrap.py`, `git diff --check
028d2aaed2c4bfbed173dc8f6b06106b3686f700 HEAD`, and the two fixture commands above.
The receipt records actual per-command process/child CPU, wall, source, versions,
Slurm job and log hashes. Allocated limits are distinct from measurements.

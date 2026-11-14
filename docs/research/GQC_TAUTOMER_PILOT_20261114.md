# GenerativeQC tautomer pilot: registered execution

The owner explicitly requested real n2 calculations using the precompiled
GenerativeQC installation and Slurm GPU allocation, followed by a reusable
quantum-chemistry skill. This permits a bounded GPU extension of the earlier
CPU-only planning envelope. No earlier result or source-data admission changes.

Protocol: `configs/research/gqc_tautomer_pilot_20261114.json`.
Entry: `scripts/gqc_tautomer_pilot.py`.
Base/head: `codex/chemistry-candidates-20261114` (PR #206) →
`codex/gqc-tautomer-20261114`; no dependency on the unfinished #203 experiment.

Independent input graphs represent explicit gas-phase charge-0 singlets.
Prepare at most 32 ETKDG/MMFF conformers with finite E/Z enumeration; select one
MMFF starting geometry per stereoisomer and retain that sampling limitation.
Input coordinates are Angstrom; native GQC coordinates are Bohr, energies are
Hartree, and forces are Hartree/Bohr. Gradients equal negative forces. No source
TS, product geometry, barrier labels or trained model are used.

Read-only reconnaissance found the 2026-10-29 bundle, an incomplete `latest`
build, and the reusable 2026-11-08 wheel/venv at upstream commit
`2bf3fd98da9c6cfba46ca83345e63e1d934e4780`. The latter is the selected runtime;
actual Python/native-library hashes are measured at execution. Existing builds
are not modified or recompiled. Old CUDA PBE force failures motivate selecting
B3LYP/def2-SVP provisionally, conditional on a new organic-system derivative
and independent CPU PySCF gate. This choice is for a minimal exploratory
screen and does not replicate the source dataset's M06-2X protocol.

Stages, run sequentially:

1. Prepare in the n2 `xtbflow` environment, zero QC, 120-second wall limit.
2. Qualify the exact runtime under `main`, `gpu:pro6000:1`, four CPU cores,
   16 GiB RAM, 10 minutes. Water and both R6 tautomers receive two central
   directional finite-difference checks plus an independent CPU PySCF E/F
   comparison. Every state must pass the predeclared tolerances. Maximum 40
   QC operations, 480 internal wall seconds, no retry.
3. Only after a matching build/config/input gate, optimize the finite cohort
   on one GPU with a 30-minute allocation and 1,500 internal wall seconds.
   BFGS requires maximum absolute force ≤3e-5 Hartree/Bohr, at most 80
   iterations/start and 800 QC operations across the stage. The budget counts
   failures. Retain every start, including unresolved/budget-exhausted starts.
4. Check final stereochemical graph identity in the RDKit environment.
   A force-converged structure is not yet a verified molecular minimum.
   Independent Hessian/frequency checks, if performed, have a separate frozen
   bounded continuation and are labelled by the actual backend. Reaction
   transition states, IRC, rate constants and solution populations are outside
   this pilot.

Maximum retained output is 256 MiB/stage; one job at a time; external retries
zero. Slurm allocation and process timing are reported separately. This budget
is a ceiling, not a prediction of runtime. A failed native call remains in the
append-only started/completed/failed journal. An externally killed call without
a closing record stays unresolved and is never silently rerun.

All stages use an exact clean source commit in a new n2 execution directory,
with separate output directories that must not already exist. `GQC_PYTHON`
selects the existing wheel's interpreter; `GQC_SOURCE_SHA` identifies its source.
Set these on n2, along with `PILOT_CODE` and `PILOT_RUN`, before executing:

```bash
ssh n2 '
  cd "$PILOT_CODE"
  git rev-parse HEAD
  test -z "$(git status --porcelain)"
  source ~/miniconda3/etc/profile.d/conda.sh
  conda activate xtbflow
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
  timeout 120s python scripts/gqc_tautomer_pilot.py prepare \
    --config configs/research/gqc_tautomer_pilot_20261114.json \
    --out "$PILOT_RUN/prepare"
  srun --partition=main --gres=gpu:pro6000:1 --nodes=1 --ntasks=1 \
    --cpus-per-task=4 --mem=16G --time=00:10:00 \
    "$GQC_PYTHON" scripts/gqc_tautomer_pilot.py qualify \
    --config configs/research/gqc_tautomer_pilot_20261114.json \
    --inputs "$PILOT_RUN/prepare/inputs.json" --out "$PILOT_RUN/qualify"
'
```

Actual job scripts will pin the execution SHA, log files and runtime variables;
raw data and personal runtime paths remain on n2. Existing source caches are
not a prerequisite for these independently generated molecular inputs.

中文说明：本轮实际计算先核验已有 GPU 程序，再优化少数互变异构体。输入来自独立
生成的小分子结构，并明确采用气相中性单重态，不能把这些条件补写成来源数据的事实。
本轮最多给出有限构象范围内的稳定性线索，不据此宣布新反应机理、溶液种群或合成产率。

## Registered bounded continuation and minimum checks

The original runtime launch (Slurm 4618) failed before QC because its existing
OpenBLAS provider was not on the loader path. The preserved failure used zero
QC calls. Exposing that existing directory produced a fresh successful gate,
with 18 counted operations. This was an environment correction, not a repeated
scientific calculation or a replacement of the old receipt.

The R6 carbonyl Cartesian BFGS start reached the registered 80-iteration limit
at 8.24e-5 Hartree/Bohr maximum force. Without changing that criterion, a separate
TRIC continuation is now frozen in
`configs/research/gqc_tautomer_continuation_20261114.json`. It visits unresolved
or budget-unreached starts in the original input order, at most once each, and
leaves completed starts unchanged. It may use at most 200 new calls/650 seconds
on one GPU, while subtracting all prior QC from a shared 800-operation envelope
and reserving 32 operations for independent checks. Original statuses/costs and
the entire input denominator remain visible even if the continuation stops.

The separate CPU PySCF minimum checks are frozen in
`configs/research/gqc_tautomer_frequency_20261114.json`. They select the lowest
qualified sampled geometry for each connectivity graph, compare B3LYP E/F,
compute up to eight analytic Hessians, check one Hessian-vector derivative at
two finite-difference scales, and evaluate M06-2X/def2-TZVP single points as a
method/basis sensitivity check. Maximum 32 QC operations, four CPU cores,
1,100 internal wall seconds, 20-minute CPU Slurm job, no GPU. Previous QC costs
are subtracted before execution. No new GQC Hessian qualification is implied.

Harmonic Gibbs quantities, when valid, use 298.15 K/1 atm gas and a single
sampled minimum per form. Shallow imaginary modes remain gray-zone results.
The alternative-method energies are single points on B3LYP geometries, not
M06-2X optimized structures or free energies. Missing forms yield incomplete
comparisons; a failed structure is never silently replaced by another molecule.

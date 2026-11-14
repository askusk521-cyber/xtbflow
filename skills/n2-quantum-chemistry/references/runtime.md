# Verified n2 runtime and API

Observed 2026-11-14: the reusable GenerativeQC wheel uses Python 3.11.17 in a
venv based on Conda `xtbflow-qc`. The `xtbflow` environment has Python 3.10 and
RDKit; use it for independent structure preparation and graph checks. Invoke
the GQC venv's explicit Python for native calculations, even after activating
Conda. The two environments are not interchangeable.

The selected upstream source is `2bf3fd98da9c6cfba46ca83345e63e1d934e4780`.
The observed native library `libgenerativeqc.so` SHA-256 is
`67bc98a4549fe8854ed6ae65791c824024bd77a9e8e498c6891dea3b8de670c4`.
It was compiled for CUDA 12.9 / sm_120 and used on an RTX PRO 6000 Blackwell.
These are identities of an observed build, not a requirement to use this
version forever. A replacement build needs its own bounded qualification.

## Dynamic library requirement

This wheel needs the `scipy_openblas32` LP64 provider directory on the loader
path. Slurm job 4618 failed before any QC with `libscipy_openblas.so` missing.
The provider already existed in the venv; exposing its directory fixed loading
without recompilation or package installation. Use the recorded
`GQC_OPENBLAS_DIR`, not a guessed system BLAS ABI.

The private n2 environment file defines `GQC_PYTHON`, `GQC_OPENBLAS_DIR`,
`GQC_SOURCE_SHA` and the declared native-library hash. When absent, locate the
existing venv and inspect its `pyvenv.cfg`, package files and library dependencies
read-only before creating a local configuration. Do not run old all-in-one
benchmark/build scripts; they may rebuild dependencies or overwrite logs.

Inside the n2 job:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate xtbflow-qc
source ~/.config/n2-qc/runtime.env
export LD_LIBRARY_PATH="$GQC_OPENBLAS_DIR:$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
cd "$QC_CODE"
test "$(git rev-parse HEAD)" = "$QC_SHA"
test -z "$(git status --porcelain)"
"$GQC_PYTHON" "$QC_DRIVER" --config "$QC_CONFIG" --out "$QC_OUT"
```

`QC_CODE`, `QC_SHA`, `QC_DRIVER`, `QC_CONFIG` and `QC_OUT` are task-specific
values established on n2, not fields automatically supplied by this skill.
The driver argument pattern is illustrative; use that driver's actual CLI.
Use fresh output/log paths. A small interactive probe can use `srun` with
`--partition=main --gres=gpu:pro6000:1 --nodes=1 --ntasks=1
--cpus-per-task=4 --mem=16G --time=00:10:00`; longer work uses a finite `sbatch`
script with the same environment setup and persistent `%j` stdout/stderr.

## Native Python units

Coordinates are **Bohr**, energies **Hartree**, forces **Hartree/Bohr**:

```python
from generativeqc import Calculator
BOHR_ANGSTROM = 0.529177210903
calc = Calculator(method="b3lyp-rks", basis="def2-svp", device="cuda",
                  precision="fp64", basis_representation="spherical",
                  density_fitting="none", max_iterations=150,
                  energy_tolerance=1e-10, density_tolerance=1e-7)
atoms_bohr = [(s, tuple(v / BOHR_ANGSTROM for v in xyz))
              for s, xyz in zip(symbols, coordinates_angstrom)]
result = calc.singlepoint(atoms_bohr, charge=charge, multiplicity=multiplicity,
                          properties=("energy", "forces"))
```

Obtain `symbols`, coordinates, charge and multiplicity from the actual task;
do not use API defaults to fill missing scientific information. Check
`converged`, `executed_backend` and finite outputs. Copy returned arrays before
subsequent native calls if retaining them. For an optimizer in Bohr, gradient
is `-forces`; forces in Hartree/Angstrom are native forces divided by
`BOHR_ANGSTROM`.

Official API source: <https://github.com/jinzhezenggroup/generativeqc>.
Prefer the installed/pinned source when behavior differs from current upstream.

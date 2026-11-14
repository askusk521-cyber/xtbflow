# What has actually been checked

The 2026-11-08 build at upstream `2bf3fd98…` has method-specific force behavior.
Historical PBE/LDA/PBE0/r2SCAN CUDA force failures remain restricted for that
build; a method appearing in the registry or reporting `forces` does not remove
those failures. Historical CPU PBE and CUDA B3LYP/RHF checks covered small
fixtures only. Do not extrapolate either a failure or a pass to every future
build, spin state or molecule.

On 2026-11-14, B3LYP/def2-SVP, spherical, FP64, direct (no density fitting),
gas-phase neutral singlets passed a fresh gate on perturbed water and the R6
enol/carbonyl initial geometries. Two directional central differences used
1e-3 and 3e-4 Bohr. Maximum directional force discrepancy was 2.79e-8
Hartree/Bohr. CPU PySCF 2.14.0 with the same basis/XC gave energy agreement
within 5.9e-12 Hartree and maximum Cartesian force difference 3.28e-5
Hartree/Bohr. The predeclared tolerances were 2e-5, 1e-4 and 3e-4 respectively.
This qualifies a bounded organic pilot, not general chemical accuracy.

The reproducible project driver is `scripts/gqc_tautomer_pilot.py` at xtbflow
commit `323fd156987322ed7181dd48b4fe514a162f702d`, with configuration
`configs/research/gqc_tautomer_pilot_20261114.json`. The gate requires the same
build, input hash and protocol before geometry optimization. Adapt the
qualification cohort when another molecular class or electronic state is
requested; do not require these exact molecules for unrelated tasks.

For a new force check, compare the analytic projected force to
`-(E(x+h*d)-E(x-h*d))/(2*h)` for a normalized direction, at more than one
reasonable step. Keep energy/coordinate units consistent. An independent
backend comparison supplements this internal derivative check. Do not tune
tolerances after seeing an inconvenient result.

The current GQC analytic Hessian/IRC interfaces have not been qualified by this
force gate. If PySCF or another program provides frequencies or a reaction path,
retain its method/build, actual call counts and claim boundary separately.
Small-system stationary points do not by themselves predict solution kinetics.

## Completed optimization example

The same pilot subsequently reached the recorded force threshold for all 13
finite E/Z starts. Two Cartesian BFGS starts hit their iteration limits and a
time-limited batch left later starts incomplete. A separately registered
geomeTRIC TRIC continuation, using the same GQC forces, resolved the remaining
starts with 91 new calls while retaining the original 456 calls and failures.
For similar molecular optimizations, the internal-coordinate driver is a useful
starting example; this observation does not require changing every optimizer.

All 13 final graphs/stereoisomers matched their intended identity. Eight
representative minima had positive projected frequencies using **CPU PySCF**
B3LYP/def2-SVP Hessians. One Hessian-vector finite-difference probe had relative
error about 2.01e-4 at two scales. This is not a GQC-native Hessian qualification.
The complete pilot recorded 562 GQC CUDA energy/force calls and 31 independent
PySCF operations, including eight Hessians, with no model training.

Exact continuation/reference source:
`aad1b5362883c5c2364acd2eec5c6c1e90ff4d13` in the xtbflow repository.
Evidence and command recipe: `docs/evidence/gqc_tautomer_20261114/RESULTS_ZH.md`.
Repository: <https://github.com/askusk521-cyber/xtbflow>.
Read the source before adapting the standalone driver; retain its budget and
input/build identity checks, but choose chemistry and tolerances for the task.

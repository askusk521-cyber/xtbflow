---
name: n2-quantum-chemistry
description: 在 n2 通过 SSH 和 Slurm 运行真实量化计算，优先复用已编译的 GenerativeQC，核验能量、力和几何优化结果并保存可复现证据。适用于 n2 上的小分子计算、后端检查和计算任务执行，不用于纯理论问答或其他集群的配置。
---

# n2 quantum chemistry

Use the existing compiled runtime for the requested chemistry. The observed
n2 host is `node2`; GPU partition/GRES are `main` / `gpu:pro6000:1`.
Verify the live queue and available resources before allocating. Every GPU
calculation, including capability and derivative probes, runs inside Slurm.
Use explicit finite time, GPU count and CPU count; the partition itself has
an unlimited time policy.

Read [runtime.md](references/runtime.md) for the verified interpreter,
dynamic-library requirements, API units and submission pattern. A private
runtime configuration may exist on **n2** at `~/.config/n2-qc/runtime.env`.
Load it inside the remote job; do not assume locally defined variables reach
SSH. Keep personal paths out of public repository files.

## Choose an actual supported calculation

- Inspect the active project's instructions, chosen method, molecular charge,
  multiplicity, units, environment, outputs and budget before execution.
- Reuse the compiled wheel. A directory named `latest` is not proof of a
  complete build. Compare source/runtime hashes and import the chosen package
  with its own interpreter. Do not rebuild or upgrade a working installation
  merely to run a molecule.
- Consult [qualification.md](references/qualification.md) when selecting a
  method or property. Capability flags establish availability, not correct
  derivatives. Keep qualification scoped to the measured build/method/basis,
  spin, elements and numerical settings.
- Energy-only work needs an energy check appropriate to the requested method.
  Optimization requires qualified forces. Frequencies require a qualified
  Hessian route; a successful force check does not qualify Hessians or IRC.
  If another backend supplies a stage, identify it explicitly in the result.

## Execute and finish

Create a clean execution directory named for the source commit and a fresh
output directory on n2. Preserve existing dirty checkouts, compiled packages
and old run artifacts. Record the source SHA, configuration hash, input hash,
native-library identity and actual environment versions.

Submit a small probe before scaling a new method/system class. Count started,
failed and completed operations, plus elapsed/allocated resources. Write a
started record before entering a native call so a scheduler kill leaves an
unresolved attempt rather than an apparent zero-cost failure. Runtime repair
after a pre-calculation launch failure is a new recorded attempt, not a
replacement of the old failure. Recheck remaining authorized budget.

Read both the calculation report and job exit/terminal state. Queue disappearance
alone is not success. If Slurm accounting is unavailable, keep the program's
measured time and any retained scheduler receipt; label missing quantities
unknown rather than estimating them as observed.

For a relaxed structure, check residual forces, chemical identity and relevant
stereochemistry. Claim a minimum only after frequency/curvature validation.
Compare energies only with a matching composition, electronic state, method
and reference convention. Label limited conformer sampling and gas-phase
screening as such; these results do not establish solution populations,
reaction rates, reaction connectivity or experimental feasibility.

Return the chemical result or actual failure, its limits, and links to retained
evidence. Updating the skill should capture a demonstrated reusable detail,
not turn the current molecule or pilot budget into a universal prescription.

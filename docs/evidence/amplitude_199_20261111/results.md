# #199 registered amplitude-policy result — 2026-11-11

Exact scientific source `61e8d709dba213fe4c0d32daa7cc8431879e906c` follows merged
#198. It reuses only the exact hash-verified #198 caches. No direction, model,
clock, event, dose schedule, threshold or input was changed after results.

| State | Baseline Phi | lambda=1 Phi | Selected lambda | Applied RMS (Å) | New E/F / forwards |
| --- | ---: | ---: | --- | ---: | ---: |
| 10 | 6.255553719 | 6.256833543 | no push | 0.000000 | 4 / 3 |
| 25 | 3.220909806 | 3.172234391 | 1.0 | 0.002040 | 2 / 1 |
| 40 | 0.451710587 | 0.440399492 | 1.0 | 0.003240 | 2 / 1 |

Two states accept the first trial, so adaptive and fixed amplitude coincide
there. At k=10 all three trials fail only the residual criterion: Phi is
6.256833543 / 6.256181965 / 6.255864914 at lambda 1 / 1/2 / 1/4, all above
6.255553719. Projection, linearization and displacement checks pass. Smaller
steps reduce the size of this worsening but do not yield an acceptable decrease.
The complete policy therefore suppresses that update after paying for all three
trials. **Zero of three qualified states accepts a smaller nonzero step.**
This does not demonstrate a backtracking efficacy advantage or establish that
overshoot caused this trained flow's earlier outcomes.

At k=25/40 residual falls by about 1.51% / 2.50%. This is a selection criterion,
not an unbiased outcome estimate, TS certificate or connected channel. No final
discrete event or reference endpoint was evaluated here. The two accepted updates
meet the registered technical gate for the four-arm #188 mechanism pilot.

Actual #199 work: 8 E/F, 5 trained forwards, no JVP/reverse, 10.441969 s driver
wall, 10.081024 s process+child CPU. No failures, unresolved work, retries,
training, GPU, reference, quantum HVP or Hessian. Fixed lambda=1 queries were
shared, not repeated; k=10 incurred two extra forward/E/F trials. Together
#198+#199 used 11 E/F and 50 forwards (ceilings 15/54), plus the original #198
6 JVP/12 reverse passes. Original historical receipts remain unchanged.

Exact-source n2 validation: 431 pytest tests/68 subtests, 56 reusable tests,
fake/cached common entry, bootstrap 512 and diff check pass. See
`software_61e8d70.json`. All attempted trials and their individual rule outcomes,
raw/cache hashes, state identities and costs are in `results_61e8d70.json`.
Command and resource limits: `docs/architecture/amplitude_backtracking_199_20261111.md`.
Raw tensors and calculator outputs remain on n2.

Decision: freeze this one policy for #188. Recompute forces and JVPs on NEW actual
F states there; do not reuse these saved-state derivatives after intervention.
No stronger serial comparison, independent confirmation or method superiority
has been demonstrated.

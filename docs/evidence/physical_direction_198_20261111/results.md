# #198 real physical-direction result — 2026-11-11

Exact scientific source: `76028d54fb5da4a16200fa67ceaf061a0d27befd`, derived from
merged common source `baa50bc0531985f92121690ee297c7c934eeca9f`. All three states,
the packet and joint checkpoint match original S0/S1 identities. No trajectory
was regenerated. Float64 CPU, exposed ID63623 / dft-da:R0, seed 0, event_lead2.

| State | Min distance (Å) | F gain / cosine | G gain / cosine | G BE response norm |
| --- | ---: | ---: | ---: | ---: |
| 10 | 0.541164 | 0.348264 / -0.020733 | 0.402579 / 0.033231 | 0.474723 |
| 25 | 0.770104 | 0.756177 / 0.013379 | 0.843202 / 0.176138 | 2.887273 |
| 40 | 0.949252 | 0.359853 / 0.394586 | 0.366696 / 0.402640 | 0.838983 |

All six derivatives satisfy the registered adjacent-scale numerical check.
Largest FD/JVP relative error is 2.44e-6. Centered/mapped invariants hold;
all continuous-event sensitivities exceed the recorded numerical floor.
F has a small negative local projection at k=10. G remains positive at all
three states, with weak alignment at k=10 and stronger alignment later. This
supports a locally nonidentity transport map and state-dependent direction
quality; it does not establish universal reversal or a useful chemical event.

The old-to-unpushed-next endpoint L2 drift is 0.228928 / 0.171522 / 0.131142 Å.
The #199 residual baseline must therefore be the cached NEXT endpoint, not the
OLD force observation. Each unit L2 probe has per-atom RMS 0.25 for 16 atoms.
The three finite force norms are 14.930225 / 10.883509 / 4.600707 Hartree/Å;
these noisy early geometries are not optimized or chemically validated structures.

Actual new work: 3 completed E/F, 45 started/completed trained forwards,
6 completed JVPs and 12 reverse passes; 4.953304 s driver wall and 4.637265 s
process+child CPU, zero GPU/training/reference/Hessian. No unresolved started
work or retry. The immutable result retains every state, direction, FD scale,
source/config/cache hash and work row. Raw coordinates/forces/tensors remain on
n2. The historical 90-forward receipt is unchanged.

Exact-source software validation passed 423 pytest tests/68 subtests,
56 reusable tests, full fake/cached handoff, bootstrap 504 and diff check.
See `software_76028d5.json`. The bounded reproducible invocation and caps are
in `docs/architecture/physical_direction_198_20261111.md`; private asset paths
are intentionally supplied through environment variables on n2.

Decision: all three G states qualify numerically for #199. Test only its
registered alpha=0.1 and lambda=1, 1/2, 1/4 policy. No final event, TS, connection,
channel benefit, strong-serial comparison or independent confirmation is claimed.

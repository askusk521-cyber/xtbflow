# S1: amplitude, transport and BE response

The retained 81 rows give 27 matched pairs on nine exposed parents. Median
standard displacement is 0.000207692 Å; normalized displacement is 0.05 Å.
The paired ratio median is **240.74094**, range 151.986–316.735. Every standard
step improves energy and force RMS; every normalized step worsens them. This
supports amplitude/overshoot confounding. It does not establish a sign error,
better TS convergence or channel yield; these parents are not used for tuning.

For `E(y)=(-y1²+y2²)/2`, the correct unstable mode with an identity endpoint map
moves toward the saddle. A wrong mode moves away. With `y=(-x1,x2)`, equivalent
to `t=.5, v=(-4x1,0)`, directly adding the correct endpoint saddle direction to
x increases |y1|. These are deliberate counterexamples. Alpha-zero identity
and positive force-unit-rescaling invariance pass. They do not justify using
J-transpose on a reflected saddle field as a general correction.

## Frozen trained response

All three states use the S0 parent/seed/checkpoint and historical addressed
float32 noise, followed by float64 CPU arithmetic. Event-lead2 has
`t_b=s, t_x=s²`; it is not pg-1's geometry-lead3 clock.

| Step | t_b / t_x | endpoint gain | cosine(Jd,d) | norm(dv_BE/dx d) | next-step BE change |
| --- | --- | ---: | ---: | ---: | ---: |
| 10 | .2 / .04 | .463835 | .154987 | .355203 | 7.56e-6 |
| 25 | .5 / .25 | .497083 | .300453 | .460705 | 1.32e-5 |
| 40 | .8 / .64 | .410983 | .534367 | .574462 | 5.86e-6 |

The direction is a fixed centered numerical probe, **not a physical force**.
Central differences at 1e-4, 3e-4 and 1e-3 Å agree with JVP: largest relative
error is 1.51e-6. The actual trained conservation projector preserves BE
symmetry (zero observed error) and electron-sum response (≤1.67e-16); endpoint
centering error is ≤6.1e-18. All 16 atoms are active; this run is not an
independent padded-atom qualification. Alpha-zero continuations are bitwise
equal. A post-step geometry update leaves that step's event unchanged and
changes the next step, confirming the one-step delay. Extra forwards are charged.

The continuous two-step states do not decode to valid chemical events in these
three probes. This is **not** final predicted-endpoint decoding or a failed
terminal chemistry experiment. The cached terminal chemical bridge is audited
separately in the #181-based S3/S2 replay. Transport is nontrivial and event
response exists, but useful physical direction and connected-channel improvement
remain untested. No guidance variant is promoted.

## Failure and cost preservation

The first exact-source run `9a5599c180c5640668ee349260ab6180d304c68f` failed
when a NumPy clock array reached a tensor operation. Fifty reconstruction
forwards completed; forward 51 started and failed. Its one JVP attempt reserved
two backward operations but completed **zero** backward operations; the original
receipt's `backward_calls=2` is a conservative scheduled count, not measured
completed work. The [failed receipt](numerical_failed_9a5599c.json) is unchanged.
It consumed 1.402990 s wall / 1.275235 s process CPU and saved the three states.

The [registered recovery](../../../configs/research/numerical_recovery_20261111.json)
fixed only clock conversion and consumed those hash-pinned states, with no
reconstruction or new selection. Source
`0bdd8b83e73597bf21c324d9a766010872e1cd33` is recorded in the receipt.
It used 39 forwards, three JVPs/six reverse passes, 1.728739 s wall and
1.628181 s CPU. Total **90 started forwards**, 89 completed, four JVP attempts,
six completed reverse passes (eight conservatively reserved), 3.131729 s wall
and 2.903416 s CPU. The original cumulative 192-forward/600-second cap did not
increase. Zero new training, GPU, xTB, HVP, Hessian or reference work occurred.

The [publication receipt](numerical_diagnostics.json) retains the original raw
report hash; decoded intermediate matrices are replaced by validity/hash fields.
Raw state tensors and original reports remain on n2. Missing Slurm historical
accounting is not reconstructed as measured resource use.

## Reproduction

Use a new output directory, exact clean source from the receipt, and n2
`xtbflow`. `XTB_RESEARCH_SOURCE` must be the clean pinned e84e330 checkout;
the model root and worker packet must match S0 hashes. For recovery,
`XTB_SAVED_STATES` points to the immutable failed run directory. The source's
registered recovery caps apply; this is not permission to repeat the run now.

```bash
ssh n2 '
  cd "$XTB_NUMERICAL_CODE"
  test "$(git rev-parse HEAD)" = "$XTB_NUMERICAL_SHA"
  test -z "$(git status --porcelain)"
  source ~/miniconda3/etc/profile.d/conda.sh
  conda activate xtbflow
  export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
  srun --partition=main --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=8G --time=00:10:00 \
    timeout 598 python scripts/diagnose_narrowed_numerics.py \
      --qualified-rows "$XTB_RETAINED_ROWS" --research-source "$XTB_RESEARCH_SOURCE" \
      --model-root "$XTB_MODEL_ROOT" --packet "$XTB_WORKER_PACKET" \
      --resume-states "$XTB_SAVED_STATES" --out "$XTB_NEW_DIAGNOSTIC_OUT"
'
```

Software validation at the recovery source: 274 pytest tests/68 subtests and
bootstrap 340 files. The preceding source passed 273/68 plus 56 reusable tests.
These checks establish software behavior only. Full command/log hashes and
environment are recorded in the validation receipt.

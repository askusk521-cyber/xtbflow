# Review follow-up for #184 / #194

The original `alpha_zero_bitwise` compares two unguided calls, and
`same_step_event_unchanged` follows from replacing only x. Those archived
booleans are retained, but they did not qualify installed guidance or actual
update ordering. New diagnostic output keys say precisely what those controls
check under the new `xtbflow-narrowed-numerical/v2` output schema; existing v1
receipts are untouched. The old trained FD/JVP values and 90-forward ledger are unchanged; this
follow-up does not load a trained checkpoint or rerun those derivatives.

`scripts/check_narrowed_guidance_entry.py` runs two isolated, exact-source checks:

- `score`: e84e330's actual `v1.sampler.rollout`/`score_direction` with nonempty
  recording and failing guides. At alpha zero both trajectories and all cost
  counters must equal the unguided baseline, with zero guide evaluations or
  backwards. Positive-alpha controls must consume the recording guide and reach
  the failing guide. This specifically tests the installed score-guidance path.
- `physical_update`: ffca40f's actual `recorded_rollout` push callback and Euler
  update. A synthetic coupled network records the geometry it consumes. The
  single physical-fixture push changes geometry at step one, leaves that step's
  event equal to the baseline, is consumed by the next network call, and changes
  the next event. Masking, centering and conservation are checked on a padded
  input. No outside replacement of b/x is used to assert the one-step delay.

These checks qualify software ordering, not a physical direction or chemistry.
The fake score represents one aggregate differentiable function; actual
component/backward counts are separate from the archived sampler's three-member
equivalent cost counters. The physical push is explicitly synthetic. Testing
the score path at zero alpha does not claim that every historical physics-driver
factory has a zero-strength no-call policy.

Each invocation is CPU-only, one thread, at most 16 synthetic forwards,
60 CPU/wall seconds and 4 MiB output; no retry, real calculator call, training,
GPU allocation or trained checkpoint forward. Expected work is nine synthetic
forwards for the score checks and four for physical-update checks. Outputs must
be new paths, and exact clean research source revisions are verified before use.

```bash
ssh n2 '
  cd "$XTB_GUIDANCE_CHECK_CODE"
  test "$(git rev-parse HEAD)" = "$XTB_GUIDANCE_CHECK_SHA"
  test -z "$(git status --porcelain)"
  source ~/miniconda3/etc/profile.d/conda.sh
  conda activate xtbflow
  export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
  timeout 60 python scripts/check_narrowed_guidance_entry.py --entry score \
    --research-source "$XTB_RESEARCH_E84E330" --out "$XTB_NEW_SCORE_RECEIPT"
  timeout 60 python scripts/check_narrowed_guidance_entry.py --entry physical_update \
    --research-source "$XTB_RESEARCH_FFCA40F" --out "$XTB_NEW_PUSH_RECEIPT"
'
```

The review source SHA and observed environment belong in the new receipts.
S5 remains unlaunched; these checks do not supply the missing physical signal.

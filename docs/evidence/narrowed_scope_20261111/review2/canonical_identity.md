# Review #195: bind trained results to their earlier public identities

Clean source `71b41cb7e384d54d6f22a8069ec91d934753c328` passed 14 focused
tests, 390 pytest tests/68 subtests, 56 reusable unittests, bootstrap 413 files
and diff check on node2. [Software receipt](canonical_identity_validation.json)
retains commands, environment, hashes, timing and finite limits.

The [new read-only audit](anchored_reference_audit.json) recomputes each result's
canonical identity, excluding only its self-hash field. For the trained records
it then verifies the three hashes from the byte-pinned public report at
`5f5304bcded99f2711ce21a090392cc2d6db8973`:

| Trained attempt | Calls | Published canonical SHA-256 |
| --- | ---: | --- |
| Joint | 468 | ccf7cbd7892aaf5c37474b80ec63618acd79b9ba4c33fd89bdd49c09c34d8dd1 |
| Independent serial | 638 | fc22caeb4421943d1d85f0c4c73efd3451ac8e93a231a881e15f112600a8bac1 |
| Feedback | 526 | bc60cd9a5a63bf9bc9d55ab660b20e9bc750f465be71fe6b894af298c9266d74 |

All three matched. The original public report's byte SHA-256 is
`f3cc2274aa5b3e1d47a8bb0beb4c32f04dbf99948f5b4f4cb6455b773aba8ff0`;
the verbatim anchor and its original source identity are retained. Tests reject
same-verdict replacements with changed cost or evidence even after recomputing
their internal self-hashes. Missing, duplicate or unexpected trained records
are rejected. `unchanged_during_audit` and `published_anchor_verified` now make
the two different checks explicit.

The earlier audit and original physical records are not rewritten. The rule
chain continues to rely on its separately preserved original 2772c7b cache
auditor; this check is not a new certification. Trained verdicts remain joint
unresolved, serial connected and feedback unresolved. No tampering is inferred
from the earlier acceptance gap.

The new audit took 0.103991 wall seconds / 0.101255 CPU seconds, with zero new
model, training, GPU or calculator work. It uses the same three retained service
directories and generation root as the earlier audit, with a new output path.
Its fixed limits are 600 wall/CPU seconds, 256 MiB input, 16 MiB output, one
concurrent process and no retry. Never run the modified certification service
to refresh an old cache.

```bash
ssh n2 '
  cd "$XTB_REVIEW195_CODE"
  test "$(git rev-parse HEAD)" = 71b41cb7e384d54d6f22a8069ec91d934753c328
  test -z "$(git status --porcelain)"
  source ~/miniconda3/etc/profile.d/conda.sh
  conda activate xtbflow
  export PYTHONPATH=src:vendor/mechai_reusable CUDA_VISIBLE_DEVICES=
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
  timeout 600 python scripts/audit_narrowed_reference.py \
    --service live_rule "$XTB_RULE_SERVICE" \
    --service trained_initial "$XTB_TRAINED_INITIAL_SERVICE" \
    --service trained_feedback "$XTB_TRAINED_FEEDBACK_SERVICE" \
    --generation-root "$XTB_GENERATION_ROOT" --out "$XTB_NEW_ANCHORED_AUDIT"
'
```

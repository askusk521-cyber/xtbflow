# S0 availability, 2026-11-11

On node2, clean source `83e5ba65387d91d6a66605d6e495aa2f78a829b5` verified
the joint/event/geometry EMA checkpoints, training status files, input packet,
saved generation, call journal and ledger. See [unmodified receipt](availability.json)
for SHA-256, bytes, metadata, measured CPU/wall time and limits. No forwards,
training, GPU allocation or calculator calls occurred. Bootstrap passed 334 files.

The packet contains the declared 16-atom, mapped, charge-0/multiplicity-1
`ID63623` input and 64 reactant automorphisms; its inference packet contains only
the six allowed fields. Hash availability does not establish independent input
preparation, source environment, cross-dataset non-overlap or efficacy.

Replay on n2 uses a new empty output and the exact clean source. Set
`XTB_NARROWED_CODE` to that source, `XTB_MODEL_ROOT` to the retained V1a model
directory, `XTB_GENERATION_ROOT` to job 4100's retained generation and
`XTB_AUDIT_OUT` to a new output filename. No private path is part of the contract.

```bash
ssh n2 '
  cd "$XTB_NARROWED_CODE"
  test "$(git rev-parse HEAD)" = 83e5ba65387d91d6a66605d6e495aa2f78a829b5
  test -z "$(git status --porcelain)"
  source ~/miniconda3/etc/profile.d/conda.sh
  conda activate xtbflow
  export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
  timeout 180 python scripts/audit_narrowed_assets.py \
    --model-root "$XTB_MODEL_ROOT" --generation-root "$XTB_GENERATION_ROOT" \
    --out "$XTB_AUDIT_OUT"
  python scripts/validate_bootstrap.py
'
```

The script enforces 180 CPU seconds, 256 MiB input and 16 MiB output, no retry;
the outer timeout enforces wall time. Initial read-only reconnaissance confirmed
the old geoinfo rows lack saved intermediate coordinate tensors. That absence
is recorded explicitly in S0's reconstruction policy. No failed chemistry cost
was reset or historical result overwritten.

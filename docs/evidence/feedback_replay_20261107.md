# Feedback decision replay — 2026-11-07, CPU job 4103

**Passed:** four policy episodes (two original reference attempts × fixed and
evidence-heuristic policies) completed and resumed with identical decisions,
states and durable artifact bytes. All 38 inputs (1,241,058 bytes) remained
unchanged. This is real-feedback replay, not a live adaptive discovery trial.

Source `0f030fe7a7dcb69c3422c8494b75a06c4053abf5`, branch
`codex/w139-feedback-loop-20261107`, based on PR #149 at
`5acca873494d3470fc5cea5c1d4d92eaf8e3b461`. The stack continues #144 → #145 →
#146 → #147 → #148 → #149 → this work. Host `node2`, Conda `xtbflow`,
Python 3.10.19, PyTorch 2.10.0+cu128, NumPy 2.2.6 and RDKit 2024.3.3.
No GPU was requested or used. Slurm requested one CPU/2 GiB/five minutes and
allocated two logical CPUs; job 4103 completed in two seconds, exit `0:0`.
Runner wall was 0.247070 s and process CPU 0.143992 s; these do not include
all interpreter/scheduler/setup overhead. New calculator calls: **zero**.

The [public report](feedback_replay_20261107.json) retains policy/config identities,
actions, historical costs and original input/trace hashes. Atomic-number arrays
and mapped atom IDs were omitted from its initial-candidate views; the complete
private source report and replay states remain on n2 and are pinned in the
[artifact envelope](feedback_replay_20261107_artifacts.json). This projection
is a summary, not a substitute executable replay input. Raw coordinates are not
published.

| Original job-4096 feedback | Fixed policy | Evidence heuristic | Historical charged primitives |
| --- | --- | --- | ---: |
| Interrupted IRC, `attempted_unresolved` | Retains failure and lower evidence layers, then stops | Same; a software failure does not justify a chemical conclusion | 265 |
| Complete `valid_alternative` | Retains actual event and original sampler | Selects predeclared independent-serial sampler after the observed/planned event mismatch | 202 |

The alternative event remains `8382f00b0f4144f3b208275d`, distinct from planned
`554a88f8a1149e46f48966fd`. Each policy records exactly one certified channel
for that episode and none for the failed episode. Neither policy obtains a new
physical observation. The 467 historical calls are counted once across the two
original attempts; comparing/replaying two policies does not spend them again.
Sampler selection is recorded with `executed_generation=false`.

No future result enters the initial state. The policy reads only feedback already
returned by the deterministic verifier, plus the frozen candidate order and
remaining historical budget. Every reference import verifies raw artifacts,
request/result, settlement, shared ledger and committed token, and recomputes
scientific feedback. Actual endpoint hashes and raw result locators remain in
the private evidence index. Replaying a cache does not count as another attempt
converging to the same physical channel.

## Identities and bounded invocation

- Policy SHA256: `d56c5673d6e2bbc0964bd8fc1e24d18513576b703f70331cae9383ff28017da0`.
- Replay config SHA256: `c3b5365c1f98145fec88d5ed90f911aa9a844ccefa8dc6b4d6ea128e8d522b88`.
- Complete private report SHA256: `ff647539c66f4bd7bb86c000a8189760918515bfba5aba336270e97b3e0d69ed`.
- Public report SHA256: `15a10b9252d6cd46e97e9877fa16308763eace52df8904235b118763ac807d06`.
- Job script SHA256: `1b4a119fe21caf91a01b6d6f72cf24b8ae8786972972635885262791b0e2e157`.
- Original execution source: `931d81f53a85669e137600c62b343120ff0de3d0`.
- Original panel report SHA256: `0365ceceaae80a5f2bc724dc38b30ff080c17902814c6212a381c9086f5dac9b`.

The preserved `feedback-job.sh` checked the clean exact source, bootstrap,
environment and available disk, then ran this command inside the recorded
five-minute CPU allocation. The script bounded the runner to 120 seconds,
16 MiB input, 16 MiB output, one thread, zero QC and zero GPU. Each policy
episode has a finite 16-action cap and a fully specified historical resource
budget; no envelope cap is raised during replay.

```bash
ssh n2 '
  cd "$HOME/xtbflow-runs/w139-20261107-0f030fe/source"
  source ~/miniconda3/etc/profile.d/conda.sh
  conda activate xtbflow
  test "$(git rev-parse HEAD)" = 0f030fe7a7dcb69c3422c8494b75a06c4053abf5
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
  export PYTHONPATH=src:vendor/mechai_reusable PYTHONDONTWRITEBYTECODE=1
  timeout 180 python scripts/replay_feedback_loop.py \
    --reference-panel "$HOME/xtbflow-runs/w137-20261107-931d81f/panel" \
    --out "$HOME/xtbflow-runs/w139-20261107-0f030fe/replay"
'
```

This is the original invocation; the existing output must not be overwritten.
A new replay needs a new named checkout/output and its own finite Slurm job.
The complete source was separately tested on n2 with
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src:vendor/mechai_reusable timeout 180 python -m pytest -q`:
**350 tests, 68 subtests passed**, one existing torch scalar-conversion warning.
`PYTHONPATH=src:vendor/mechai_reusable python -m unittest discover -s tests/reusable -p 'test_*.py'`
passed 56 tests. Bootstrap verified 371 source files; diff check passed.
The 12 new focused cases cover future-label exclusion, original ledger
verification, unavailable adapters, complete envelopes, cache vs independently
repeated channels, failed costs, corrupted history and interrupted state-write
recovery. These are software tests, not chemical evidence.

W09 remains incomplete: live generation/resampling, physical probes, reference
acquisition, TS refinement and intermediate expansion are unavailable adapters.
There is no real repeated-channel exploration reallocation yet, and sampler
selection after endpoint disagreement has not been followed by a new proposal.
The policy was developed with selected historical feedback, so it is not an
independent or blinded policy evaluation. W10 must compare real policies at
matched complete costs. No superiority, new channel discovery, new training,
family generalization or confirmatory claim follows from this replay.

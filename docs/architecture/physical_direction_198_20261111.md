# Real physical-direction diagnostic — #198

This stage uses the merged #197 `prepare_injection`/`model_values` interface.
For each retained state 10/25/40 it observes the OLD predicted endpoint with
one forward, computes one metered gas-phase GFN2 E/F, then measures the local
endpoint and continuous BE-velocity response at the ordinary post-Euler base.
The base includes the updated event and next clocks. No remaining rollout or
final chemical decoder is evaluated. Every intended state/direction is retained.

The source packet, saved-state receipt/tensors and checkpoint are hash checked
before scientific calls. The clean isolated trained source is pinned at e84e330;
all imported numerical/model files are pinned in the new stage config. The
input-only element vocabulary shim excludes target loaders. Force is negative
native Hartree/bohr gradient divided once by 0.529177210903. The current adapter
and tblite build match the recorded qualified bounded-pairing source/build;
source environment remains unknown, gas is an explicit modeling assumption.

`event_direction` preserves ffca40f's predicted-ΔBE construction, evaluated on
the old predicted endpoint. Its reflected force is not a verified saddle mode.
For each of F/G, one reverse-over-reverse JVP and six central-FD forwards use
flattened unit L2 directions and scales 1e-4, 3e-4, 1e-3 Angstrom. Technical
agreement needs two adjacent scales at relative error <=1e-3; below the recorded
100*float64_eps*(norm(endpoint)+norm(event_velocity)+1)/min(scale) floor, use
that explicit absolute bound. RMS equals L2/sqrt(active atoms), hence /4 for
this 16-atom input. Cached unperturbed next endpoints and derivatives are bound
to raw cache hashes for #199, not reused at changed states.

Limits are 3 E/F, 45 model forwards, 6 JVP/12 reverse passes, 300 wall and
process+child CPU seconds, one CPU thread, zero GPU, 8 GiB and 32 MiB output,
no retries/training/reference work. Failed forwards are durably started before
execution. Each E/F runs in a new process group, has a 60 s limit and is killed
and reaped before timeout settlement; missing receipts/unknown work stop the
stage. SCF failures retain their charged attempt and leave that signal invalid.

The bounded command uses a new clean branch/commit-named n2 worktree and a new
output directory. XTB_SOURCE_SHA is the committed driver SHA in the result;
XTB_RESEARCH_SOURCE is clean e84e330. All private asset paths stay on n2.

```bash
ssh n2 '
  cd "$XTB_CODE"
  test "$(git rev-parse HEAD)" = "$XTB_SOURCE_SHA"
  test -z "$(git status --porcelain)"
  source ~/miniconda3/etc/profile.d/conda.sh
  conda activate xtbflow
  export PYTHONPATH=src:vendor/mechai_reusable CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
  srun --partition=main --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=8G --time=00:05:00 \
    timeout 298 python scripts/diagnose_physical_direction.py \
      --research-source "$XTB_RESEARCH_SOURCE" --model-root "$XTB_MODEL_ROOT" \
      --packet "$XTB_PACKET" --states "$XTB_SAVED_STATES" --out "$XTB_NEW_OUT"
'
```

This new manifest grants only the #198 finite physical work. Old S0 zero-QC
permissions, the 90-forward receipts and all original hashes stay immutable.
Results are exposed local calibration, never channel gain or method superiority.

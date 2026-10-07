# M0 numerical diagnosis — owner-authorized scope

Owner message: “诊断”. No formal data/model/hyperparameter changes, no resumed formal training, no test sampling.

## Evidence
- Read-only sample 7451: train split 0, parent 9abe2286d9cc, 21 atoms; all cached coordinates and BE matrices finite. Largest reactant-to-TS atom displacement 8.802886 Å. This does not establish invalid chemistry or justify exclusion.
- SHA-256 of backbone.py, flow.py, model.py, numerics.py matches local task3 and remote formal source-b8d135d.
- Independent initial-weight CPU forward sweep: initial-forward-7451.json. No nonfinite output in tested sweep, but large near-TS losses for some initializations.
- Independent bounded Slurm diagnostic replay #2547, 1 GPU, 10 minute limit: joint seed 0, unchanged formal configuration and noise settings; no formal checkpoint or run directory overwritten. Remote diagnostic output directory ended up `joint0-` because shell expanded SLURM_JOB_ID before allocation; unique diagnostic parent directory was used and mkdir(exist_ok=False) protected against overwrite.
- Replay failed at exact formal step 7559 and bad sample 7451 only, row 31, t=0.8604222536087036. Elapsed loop 288.853617 seconds. Inputs and all model parameters finite. Checked gradients and parameters on every preceding step: no earlier nonfinite detected.
- `replay-2547/result.json` contains layer and submodule traces. No forward module output was nonfinite. Scalar maxima by six layers approximately 5.93, 35.98, 104.41, 4264, 3.33e7, 1.32e16; vector maxima approximately 5.57, 5.49, 25.41, 666, 1.02e5, 1.53e11.
- Diagnostic pre-failure state and inputs preserved in `replay-2547/diagnostic_state.pt` (diagnostic-only, not candidate checkpoint).
- Independent CPU operator confirmation, `replay-2547/loss_operator_check.json`: float32 b_vel/x_vel outputs finite (maxabs ~9.92e20/~1.25e23); squared residuals overflow for sample 7451; both losses inf. Same state/inputs promoted to float64 produce finite but enormous losses (~3.00e38 event, ~1.45e43 geometry).

## Conclusion and limits
Exact joint seed-0 failure is a finite-forward activation explosion followed by float32 squared-residual overflow in event_loss/geometry_loss, NOT evidence of a zero-distance sqrt NaN or pre-existing nonfinite inputs/parameters. PaiNN scalar/vector update amplification precedes the loss overflow. Float64 does not solve the underlying explosion. All other seed/arm failures remain unlocalized: shared sample association is not proof of identical cause.

## Follow-up localization using saved state (no additional training)
`replay-2547/update_term_localization.json` recomputes the exact layer algebra on CPU. In all six layers, maximum scalar/vector activation for sample 7451 is at zero-based atom 7 (hydrogen). The dominant scalar amplification is `a_sv * inner_to_s((uv * vv).sum(dim=2))`, reaching ~36.3, 101.7, 4274, 3.33e7, 1.32e16 in layers 1–5. Other 63 samples remain moderate: last-layer maximum scalar ~10.90, vector ~7.81. This localizes the explosion to that atom's scalar/vector update rather than a batch-wide failure.

`replay-2547/atom7_geometry.json`: atom 7 is H; nearest other atom in cached TS is 8.509 Å away, versus reactant C–H distance 1.095 Å and product nearest H distance 0.744 Å. This may reflect a dissociative configuration, source/path selection, or preprocessing issue; raw HDF5 provenance must be checked before classifying it. No sample has been modified/excluded.

## Raw official HDF5 provenance check
`raw_h5_provenance.json` maps cache index 7451 to `/train/C6H14O/rxn10025`. All endpoint datasets have shape [1,21,3]; atomic numbers match. Reapplying the implemented centering/proper Kabsch rotation reproduces cached endpoints within 2.02e-7 Å. The raw official transition_state itself has H7 nearest-neighbor distance 8.50926101963622 Å: rotation/caching did not create the isolated H configuration.

`raw_trajectory_check.json`: the 3914-frame positions array contains the named TS exactly at index 3912; that frame energy equals the named TS energy (-8486.325991648939 eV). Global trajectory energy maximum is instead index 7 (-8483.601968137806 eV), but this array is not established as an ordered converged minimum-energy path, so replacing the named TS with an argmax frame is NOT justified. Endpoint atomic numbers all agree. This rules out an accidental first-frame slice of a multi-frame named TS and obvious cache/alignment corruption; it does not validate the source TS chemically.

## Remaining failed runs: exact bounded diagnostic replay
Slurm #2548 (joint_s2), #2549 (geometry_s1), #2550 (geometry_s2), each one GPU, 12-minute allocation limit. Independent script replay_failures.py reads original run_meta settings. No formal artifact modified. Actual loop elapsed 414.73 / 395.47 / 352.42 seconds; allocation wall time is not available from sacct (accounting disabled). Results and diagnostic states retained in sibling replay-<run> folders.

All three reproduced original failure steps exactly: joint_s2 10893 (bad row 45, t=0.8604839444), geometry_s1 11447 (bad row 41, t=0.9130203724), geometry_s2 10095 (bad row 51, t=0.9581016898). For each, sample 7451 is the sole nonfinite geometry-loss row; inputs/parameters remain finite and preceding-step gradient/parameter checks found no nonfinite. Geometry runs have finite forward-module outputs followed by inf loss, matching squared-error overflow mechanism. Joint_s2 differs in final overflow location: trunk s/v remain finite (~1.34e25/~3.42e17), but EventHead `si * sj` overflows to inf before the first pair Linear, which produces NaN. `head_product_check.json` confirms this. Thus all four formal failures share extreme forward amplification associated with sample 7451, but the first overflow operator is not identical.

## Proposed next decision (not implemented)
1. Authorize independent diagnostic replay/localization of other failed seeds to check whether the mechanism is shared.
2. Then approve a protocol/code change only after reviewing exact proposed numerical stabilization for PaiNN update products/vector scale; preserve common settings and parameter-matching obligations across arms.
3. Do not delete sample 7451, silently clamp outputs, treat float64 as a fix, lower only one arm's learning rate, select diagnostic state as a checkpoint, or resume/test without approval.

Formal experiments remain stopped. Goal incomplete. This report is an interim diagnosis, not a gate result or completion audit.

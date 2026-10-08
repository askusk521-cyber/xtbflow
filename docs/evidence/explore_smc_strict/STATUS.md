# Strict-order replication: STOP at CPU smoke reproduction gate

User explicitly selected option2, authorizing new strict execution while preserving prior outputs. Original branches and datasets remain untouched.

## Current evidence
- New branch explore/smc-strict-replication, pushed frozen execution commit d878762e28b5cb225b3765d13d3389538de85310. New scripts only plus execution configuration and inventory; original scientific definitions unchanged.
- Clean source-d878762 used. CPU-only smoke,2parents32proposals seed0,one worker,nice10,CUDA hidden. Launched with load2.27. No formal replication calculations or Slurm jobs started.
- CPU smoke stopped at C1 gate. File strict-smoke-d878762/c1_manifest.json: n64, event_status_mismatches0, raw_max_error0.0015258261937560746 kcal/mol, threshold1e-6, gate_passed=false,dirty=false.
- Log strict-smoke.log, process timing strict-smoke-time.txt. PID918391 finished. No automatic downstream jobs.
- Only reproduction fields inspected; no scientific H/utility outcomes examined.

## Read-only saved-state diagnostic
- Compared64 saved CPU trajectories with old GPU seed0 trajectories using query_id/proposal keys. Query payloads and generator model hashes match exactly.
- At k35 maximum x_hat difference8.106231689453125e-06 Angstrom; at k45 maximum5.918741226196289e-05 Angstrom. End x difference2.565234899520874e-05 Angstrom. Continuous states already differ before xTB evaluation.
- Evidence strict-saved-state-diagnostic.json; script diagnose_saved_states.py. No new network/xTB calls, no H/utility inspection, no input modifications. Initial noise states not saved and not compared. Platform and batch shape vary together; this is evidence of upstream numerical differences, NOT proof of their cause.
- Downstream still stopped. Do not use these small coordinate differences as authority to relax the1e-6 energy gate.

## Interpretation and mandatory stop
This is the first actual failed numerical reproduction gate in strict replication. Handoff explicitly requires stop rather than bypass on any failed reproduction check. Do not silently increase tolerance, rewrite manifest, or proceed with formal stages.
Potential explanation, not established: CPU rollout with2-parent batch differs slightly from original GPU/batch256 floating-point path; xTB amplifies geometry differences. Endpoint events/status still exactly match. Need distinguish expected numerical platform differences from implementation bug.

## Options and costs for owner
1. Authorize a smoke-specific gate definition: CPU smoke verifies plumbing and same-device repeatability, while formal GPU G1/C1 retains original pg1 exact event/status and1e-6 raw tolerance. No scientific parameter changes; re-freeze execution protocol and repeat CPU smoke. Estimated few CPU minutes then original formal runtime. This does relax scope of an existing gate for smoke and requires explicit permission under handoff.
2. Authorize matching GPU/batch256 reproduction smoke, preserving full numerical threshold; requires Slurm and padding/layout matching, differs from required CPU2-parent smoke. Extra implementation plus small GPU job.
3. Diagnose CPU/GPU/batch effects read-only on copies before choosing; any new calculations must remain bounded and not inspect outcome metrics. Do not presume diagnosis is proven.

New implementation now persists C1 initial states and G2 consumes them directly. Size-independent analysis plumbing covers actual smoke populations and all24 comparisons, formal analysis additionally checks against original full analysis. These paths after C1 have NOT been reached by smoke; cannot claim successful full-chain smoke.

Prior smc-1 and PR116/117 remain valid as previously reported; do not change their results because this new CPU smoke gate failed. Goal remains incomplete. Report this concrete blocker, not the superseded ambiguous-authorization blocker.

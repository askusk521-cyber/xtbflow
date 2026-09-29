# Claim–evidence matrix

Status: live decision document for issues #24, #28, #55, #58, #59 and downstream manuscript work. Evidence states are `implemented`, `development_signal`, `confirmatory_supported`, `failed`, or `not_tested`.

| Claim ID | Candidate claim | Present state | Required controls and statistical unit | Primary evidence required | Downgrade rule |
|---|---|---|---|---|---|
| C-METHOD-1 | The event flow preserves the declared global electron bookkeeping constraint by construction. | implemented at contract/test level | Direct invariant tests over sampled trajectories and decoded candidates; unit is a candidate trajectory | Zero continuous conservation violations plus disclosed discrete rejection/repair rate | If only a penalty or post-hoc repair enforces balance, describe it as regularization/decoding, not constructive conservation. |
| C-METHOD-2 | Dynamic event↔geometry coupling improves intended-TS proposal quality beyond serial event→geometry generation. | not_tested on real event/TS supervision | Matched independent, serial and joint models; unit is held-out parent reaction/family | Reference-validated intended TS per parent and per fixed calculator budget, with paired uncertainty | If joint does not beat serial, remove coupling from the headline and use the smallest effective architecture. |
| C-METHOD-3 | A generated sign-invariant reaction projector improves saddle-aware refinement. | implemented interface; not_tested scientifically | Equal-budget ordinary descent, bare-potential saddle guidance and calibrated-potential saddle guidance; unit is parent reaction | Accepted intended TS per calculator call, mode agreement, failures and wall time | If saddle guidance does not beat ordinary descent on the same potential, retain only as an optional optimizer. |
| C-PHYSICS-1 | The calibrated semiempirical potential improves held-out E/F error over bare GFN2 in its admitted domain. | development_signal | Multi-seed, parent-grouped GFN2/direct/Delta comparison; unit is independent parent molecule/reaction | Energy and force errors, tails, calibration, failure rate and training cost | If the gain is seed- or split-dependent, report a limited development result and do not use “DFT accuracy”. |
| C-DISCOVERY-1 | Reactant-side XTBFlow discovers reference-validated intended TSs without correct product or reference TS input. | not_tested | Frozen input firewall, strong rule baseline, serial baseline and equal downstream validation; unit is unseen parent reaction | Stationary point, exactly one relevant negative mode, bidirectional endpoint connectivity and full cost | If product-, TS-, mode- or target-derived solvent information leaks into inference, relabel the task and rerun. |
| C-CHEM-1 | The selected prebiotic system yields a new, discriminating computational prediction about competing selectivity. | not_tested | Prespecified main system plus independent transfer line; unit is parent chemistry/condition, not conformer | Layered reference evidence, alternatives, uncertainty and a falsifiable predicted direction | If evidence only reproduces known outcomes or one feasible TS, limit the result to retrospective consistency or feasibility. |

## Evidence already available

- Architecture and input-boundary contracts establish intended interfaces, not superiority.
- Synthetic joint-flow save/restore/sampling evidence establishes execution only.
- The 256-configuration SPICE2 result is a one-seed development signal for E/F calibration and is not event/TS evidence.
- GFN2 and CP2K qualification artifacts apply only to their explicit protocol and scope; they do not validate a mechanism.

## Minimum records attached to every claim

Each result used in a claim must identify the code commit, data/split hash, model/checkpoint hash, protocol IDs, allowed inference fields, candidate and calculator budgets, parent-level outcomes, failed attempts, and the exact figure/table consuming the result.

## Development and confirmation use

Development evidence may select architecture, thresholds, candidate chemistry, physical protocol or narrative. Once it influences any such choice, it cannot be relabelled as confirmation. Confirmation evidence is generated only after the freeze package in [`configs/validation/confirmation_v0.1.yaml`](../../configs/validation/confirmation_v0.1.yaml) is complete and execution is explicitly enabled.

A confirmation batch becomes development evidence immediately if any result is inspected before the planned aggregate release, or if the model, application, threshold, metric, budget or validation protocol is changed in response. A new confirmation batch and new split hash are then required.

## Decision gates

1. **Gate G1 — conservation:** C-METHOD-1 must pass exact software invariants and report discrete validity. Failure blocks conservation language but not an unconstrained baseline study.
2. **Gate G2 — coupling:** C-METHOD-2 must show a parent-level gain over the matched serial control after identical physical processing. Failure removes “joint coupling” from title-level claims.
3. **Gate G3 — guidance:** C-METHOD-3 must improve accepted intended-TS yield per call over ordinary descent. Failure removes saddle guidance from novelty claims.
4. **Gate G4 — discovery:** C-DISCOVERY-1 requires the frozen input firewall and reference connectivity evidence. Geometry RMSD alone cannot pass this gate.
5. **Gate G5 — chemistry:** C-CHEM-1 requires a discriminating prediction with an independent computational check. A literature-compatible result alone is insufficient.

## Allowed manuscript language before the gates pass

Allowed: “we implement”, “we define”, “we test the hypothesis”, “development evidence suggests”, and exact bounded statements tied to an artifact.

Not allowed: “we demonstrate superior discovery”, “general mechanism prediction”, “DFT-level”, “kinetic preference”, “validated prebiotic pathway”, or any priority claim unsupported by the novelty audit.

## Negative results

Negative or indistinguishable results remain in the matrix. The response is to simplify the architecture, narrow the domain or change the claim—not to delete families, move thresholds, increase budgets selectively, or recycle a viewed confirmation set.

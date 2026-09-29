# Application selection record

Record ID: `xtbflow-application-selection/v0.1`
Status: **no application selected**
Decision use: exploratory-development
Last updated: 2026-09-29

## Candidate families under review

| Family | Seed records | Current evidence | Current decision |
|---|---|---|---|
| Aqueous thioester-mediated RNA aminoacylation | `origins_acyl_001`, `origins_acyl_002` | Primary-source identity anchors only | Retain in quarantine pending exact state and input preparation |
| Sulfide-mediated aminonitrile ligation | `origins_aminonitrile_001`, `origins_aminonitrile_002` | Primary-source family/scope anchors; exact reaction instances still unresolved | Retain in quarantine; SI extraction is mandatory before structure work |
| Aminooxazole/ribonucleotide-precursor chemistry | `origins_aminooxazole_001`, `origins_aminooxazole_002` | Primary-source identity anchors only | Retain in quarantine pending stereochemistry, phosphate/environment and input preparation |

The registry is [`configs/science/origin_candidates_v0.1.yaml`](../../configs/science/origin_candidates_v0.1.yaml). None of these rows is a model input or result.

## Prespecified selection requirements

A main application requires at least one admitted parent with a nontrivial competition question, qualified low-cost attempts, feasible reference validation, complete failure/cost reporting and a reason the method is useful beyond a strong rule. A transfer line must test a distinct event pattern or mechanism family rather than a substituent change.

The decision will consider scientific discrimination, input integrity, computational tractability, model relevance, reference evidence, transfer value and cost/failure transparency. No family is predeclared the winner.

## Decisions made so far

1. Keep the three #27 families; do not expand to a reaction encyclopedia.
2. Start with two literature anchors per family, matching the six-system first delivery requested in the issue review.
3. Do not call the anchors runnable inputs: all six lack at least one exact structure/microstate/environment/hash requirement.
4. Keep #60 as a later stress-test expansion after the generic benchmark protocol is frozen.
5. Treat every screening result as development evidence; final confirmation uses new unseen records under #28.

## Next discriminating work

For each seed, extract the exact source/SI reaction instance, structures, conditions and uncertainties. Then issue #18 generates reactant-visible conformers/encounter geometries and saves hashes before any search. Candidates that cannot obtain defensible charge, multiplicity or atom-pool contracts remain quarantined while others proceed.

After calculator qualification, run bounded strong-rule/traditional pilots first. Model comparisons begin only when the registered serial and joint implementations can consume the same frozen inputs. No model score will be fabricated to fill the current blank.

## Selection outcome fields to complete later

- selected main system and record/version;
- selected transfer line and distinctness argument;
- frozen evidence snapshot and budget;
- rejected/paused systems and reasons;
- known selection bias from development screening;
- next independent confirmation batch;
- claims allowed and claims explicitly not supported.

## Decision history

| Date | Decision | Evidence used | Confirmation impact |
|---|---|---|---|
| 2026-09-29 | Registered six literature anchors; selected none | Primary papers and issue #27 scope | All records remain exploratory/quarantined; no confirmation batch affected |

# Prebiotic candidate screening protocol

Protocol ID: `xtbflow-origins-screening/v0.1`
Issues: #27 and #60
Status: exploratory protocol; no chemistry family has been selected.

## Scope and current seed set

The initial registry contains two literature anchors in each of three families: aqueous thioester-mediated RNA aminoacylation, sulfide-mediated aminonitrile ligation, and aminooxazole/ribonucleotide-precursor chemistry. These six rows are not runnable inputs. All are quarantined until exact structures, conditions, microstates, atom pools and hashes are prepared.

The registry deliberately does not add the broader #60 stress-test families yet. Phosphoryl transfer, generic carbonyl chemistry and additional C–C/C–N reactions enter only after the generic benchmark protocol is frozen and a bounded expansion decision is recorded.

## Admission ladder

### Stage L0 — literature anchor

A primary paper and exact source locator identify a real experiment or computational system. Chemical names may be present, but no model or physical calculation may consume the row.

### Stage L1 — identity and state contract

Record exact structures, stereochemistry, tautomers/protonation states, net charge, multiplicity, all explicit atoms, buffer/catalyst/environment species, source conditions and uncertainty. Unsupported neutral-singlet defaults are forbidden.

### Stage L2 — reactant-visible input

Issue #18 independently generates conformers, encounter geometries and complete water/environment pools without using the target product, TS, mode or post-hoc active-water assignment. Save generation rules, random seeds, coordinates and SHA-256 hashes before search.

### Stage L3 — qualified physical pilot

Run only qualified calculators and bounded strong-rule/traditional initialization. Record every attempt, failure, calculator call and protocol ID. Semiempirical convergence is not reference validation.

### Stage L4 — exploratory model comparison

When the registered models exist, compare strong-rule, serial and joint methods with the same visible inputs and physical budget. Results may select the application and therefore remain development evidence.

### Stage L5 — application selection

Select at most one main system and one transfer line using the frozen criteria below. All viewed systems and outcomes remain development records; confirmation requires a new unseen batch under #28.

## Selection dimensions

Each admitted parent receives an evidence record, not a single opaque score:

1. **Scientific discrimination:** a concrete competition question with outcomes that distinguish explanations.
2. **Input integrity:** source-supported states and reactant-visible preparation without target leakage.
3. **Computational tractability:** qualified semiempirical/reference methods, finite atom count and bounded failure rate.
4. **Method relevance:** opportunity for event validity or joint event/geometry reasoning beyond a trivial rule.
5. **Reference evidence:** ability to test stationarity, relevant mode and endpoint connectivity.
6. **Transfer value:** a distinct family or event pattern capable of testing a narrowed generalization claim.
7. **Cost and failure transparency:** complete calculator, retry, adjudication and wall-time records.

No absolute-accuracy score automatically selects a family. A system solved by a simple rule, outside the physical domain, missing state evidence, or too expensive can be rejected even if one successful TS is found.

## Required baselines and budgets

- strong-rule event proposal plus traditional TS initialization;
- serial event→geometry model when available;
- joint event↔geometry model when available;
- identical downstream physical and reference-validation protocols;
- fixed candidate and calculator-call ceilings shared across families or explicitly documented equal-cost adaptations.

Reference spot checks include random/strong-rule safeguards, model disagreements and semiempirical failures. Selecting only easy semiempirical successes is prohibited.

## Attempt record

Every attempt writes a JSON record under `reports/origins_screening/` with candidate and input hashes, code/protocol identities, visible input fields, proposal source, budgets, calculator calls, status, evidence tier, failures and decision use. The directory README defines the template.

## Evidence tiers

- `identity_only`: primary-source anchor, not computational input;
- `input_frozen`: hashed reactant-visible structures and state contract;
- `semiempirical_attempted`: bounded qualified low-cost attempt;
- `reference_stationary`: converged reference stationary point;
- `reference_mode`: exactly one relevant negative mode;
- `reference_connected`: bidirectional endpoint connectivity under the frozen protocol;
- `comparative_confirmatory`: unseen frozen batch under #28.

## Selection rule

The selection record must name the retained main system, retained transfer line, rejected/paused systems, evidence used, known biases and the next most discriminating bounded experiment. “Insufficient evidence to select” is an acceptable outcome.

Application selection cannot change the generic benchmark result. A chemistry family that makes the model look favorable is not evidence of algorithmic superiority; generic held-out evidence remains separate.

## Scientific boundaries

The study models elementary reaction proposals and computational selectivity questions. Finite water clusters do not represent a complete aqueous environment or wet–dry cycle. A found saddle does not establish a dominant pathway, rate, product distribution or historical occurrence. New wet experiments are not part of v0.1.

## Current state

As of 2026-09-29, six L0 literature anchors are registered and all remain quarantined. No L1/L2 input, physical attempt, model score, main-system selection or confirmation result exists. The next deliverable is exact source/SI extraction followed by six independently generated and hashed reactant-side inputs.

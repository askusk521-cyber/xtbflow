# Conditional paper outline for XTBFlow v0.1

This outline is a decision scaffold, not a commitment to a positive result or publication venue. Title, abstract and conclusions must follow the gates in [`claim_evidence_matrix.md`](claim_evidence_matrix.md).

## Working title if G2 and G4 pass

**Conservation-preserving joint flows for reactant-side reaction-event and transition-state proposal generation**

Add “with budgeted saddle-aware refinement” only if G3 passes. If G2 fails, use a narrower title describing the surviving component rather than joint coupling.

## Abstract logic

1. Problem: TS discovery is costly, while endpoint-conditioned generators assume information unavailable in open reaction discovery.
2. Method hypothesis: jointly evolve a conservation-constrained electronic event state and an equivariant 3D proposal from reactant-visible inputs.
3. Physical interface: optionally predict a sign-invariant reaction projector and refine with an independently calibrated, strictly metered potential.
4. Evaluation: compare unconstrained, serial and joint controls under matched data, proposal and calculator budgets.
5. Result: insert only frozen parent-level metrics and reference-validated connectivity outcomes.
6. Boundary: state the admitted element/microstate domain and distinguish computational evidence from kinetics or historical prebiotic occurrence.

## Main sections

### 1. Introduction

- Explain the information gap between endpoint-conditioned TS generation and reactant-side discovery.
- Separate reaction-event validity, 3D TS initialization and physical validation.
- State H1–H3 as falsifiable questions rather than presumed contributions.
- Summarize prior-work boundaries using [`novelty_matrix.md`](novelty_matrix.md).

### 2. Problem and information contract

- Define reactant-visible input, supervision-only labels and downstream validation views.
- Define the fixed atom pool, explicit hydrogens, charge/multiplicity and environment pool.
- State the CHNOS, closed-shell v0.1 scope and all quarantine rules.
- Prohibit product, reference TS, reference mode and target-derived active solvent at inference.

### 3. Method

- Event state and constructive conservation: `A b = c`, projected/noise parameterization and legal discrete decoding.
- Equivariant geometry state and sign-invariant projector `P = u u^T`.
- Registered independent, serial and bidirectionally coupled models.
- Independent physical energy `U_GFN2 + Delta U`; no event-dependent energy.
- Metered ordinary and saddle-aware refinement with trust region, clipping and failure policy.

### 4. Experimental design

- Real event/TS development data with parent/family grouping and an input firewall.
- H1 comparison: unconstrained/penalty versus constructive conservation.
- H2 comparison: independent versus serial versus joint coupling.
- H3 comparison: ordinary versus saddle-aware refinement on bare and calibrated potentials.
- Metrics led by accepted intended TS per parent and per calculator call; RMSD is diagnostic.
- Multi-seed model variation and parent-level bootstrap or paired intervals.

### 5. Generic benchmark results

- Report validity, intended-event recovery, refinement success, mode/connectivity acceptance and cost.
- Include failures, abstentions, SCF/search failures and cases where postprocessing erases model differences.
- Keep development model selection separate from the frozen confirmation batch.

### 6. Prebiotic stress test

- Use the frozen candidate registry and screening protocol after the generic benchmark protocol is fixed.
- Analyze one selected main system and one transfer line; retain non-selected families and reasons.
- Separate stationary-point evidence, endpoint connectivity, free-energy approximations and kinetic interpretation.
- Frame the result as a computational stress test and testable chemical prediction.

### 7. Discussion and limitations

- State which gate passed or failed and simplify the claimed contribution accordingly.
- Discuss information visibility, data coverage, element/microstate scope, solvent representation and potential-model error.
- Do not infer physical time, reaction probability, dominant flux or early-Earth history from the generative coordinate.

### 8. Reproducibility

- Publish code/data/protocol identities, attempt-level failures, calculator ledger and split hashes.
- Release only the permitted public confirmation index; do not expose sealed labels before the planned release.

## Provisional figures and tables

| Item | Required content | Release condition |
|---|---|---|
| Fig. 1 | Information-contract and architecture diagram with serial/joint controls | Interface implementation complete |
| Fig. 2 | H1/H2 matched-budget validity and intended-TS results | #58 real-data gate complete |
| Fig. 3 | H3 accepted intended TS per calculator call and failure breakdown | #59 gate complete |
| Fig. 4 | Selected prebiotic competition network with evidence tiers | #27 selection and #28 confirmation complete |
| Table 1 | Prior-work input/output/validation matrix | Incremental literature audit complete |
| Table 2 | Dataset and split provenance by independent parent/family | Real benchmark admitted |
| Table 3 | Full compute budget, failures and reference evidence | Frozen runs complete |

## Prewritten outcome branches

- **G2 passes, G3 passes:** joint co-flow is the headline; projector-guidance is a secondary contribution.
- **G2 passes, G3 fails:** joint co-flow is the headline; physical refinement is standard downstream processing.
- **G2 fails, G3 passes:** do not claim joint architecture value; evaluate whether the projector/refinement interface alone survives a fair baseline.
- **Both fail:** publish only a narrower dataset, audit or negative methods result if independently useful; do not preserve the original title by rhetoric.

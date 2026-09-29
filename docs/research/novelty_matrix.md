# XTBFlow v0.1 novelty and prior-work matrix

Status: working research audit for issues #24 and #55. Verified through 2026-09-29 against the primary papers or official records listed below. This document records claim boundaries; it is not a legal patent search and does not establish priority.

## Proposed paper question

Can a reactant-side model jointly evolve a conservation-constrained reaction event and an SE(3)-equivariant three-dimensional transition-state proposal, and does that coupling improve reference-validated intended-TS yield under the same physical-compute budget as serial and unconstrained controls?

## Prior-work comparison

| Work | Inference information | Generated object | Conservation / geometry | Physical refinement and validation | Consequence for XTBFlow claims |
|---|---|---|---|---|---|
| TSDiff, arXiv:2304.12233 | 2D reaction graphs | TS geometries | Diffusion in 3D; no XTBFlow event-state conservation contract | Geometry and barrier comparisons on TS benchmarks | XTBFlow is not the first graph-to-TS generative model. |
| OA-ReactDiff, arXiv:2304.06174 | Known reactant and product objects | TS geometry by inpainting | Object-aware SE(3) equivariant diffusion over R/TS/P | Confidence selection and quantum-chemical evaluation | Equivariance and joint multi-object modeling are not standalone novelty. |
| FlowER, arXiv:2502.12979 | Reactant-side reaction representation and conditions | Electron redistribution / products / mechanisms | Electron-flow parameterization enforces exact global balance | Product and mechanism prediction; not a 3D TS generator | Conservation-by-construction alone is already prior art for reaction events. |
| React-OT, DOI 10.1038/s42256-025-01010-0 | Paired reactant and product geometries | Deterministic TS geometry | Object-aware SE(3) optimal-transport-like flow | GFN2-xTB pretraining/input preparation and DFT workflow integration | Flow-like TS generation and low-cost quantum support are established for endpoint-conditioned tasks. |
| MolGEN, arXiv:2507.10530 | Task-dependent conditioning; TS generation uses reaction endpoints, product generation is open ended | TS, products, and reaction-network candidates | Conditional flow matching; product and TS modules have different information contracts | Saddle optimization / IRC analyses in the reported network study | XTBFlow is not the first flow-matching framework spanning products and TSs. |
| TS-DFM, DOI 10.1038/s41467-026-74101-0 | Reactant and product endpoints | TS in distance-geometry space | Conditional flow matching over interatomic distances | CI-NEB acceleration and benchmark evaluation | Distance-aware flow matching is prior art; Cartesian co-flow is not novel merely by representation. |
| TransTS, arXiv:2608.14076 | Atom-mapped reactant-product pairs | Transformation-aware TS geometry | Equivariant geometry conditioned on explicit endpoint transformation | IID/OOD saddle refinement and intended-path recovery | Explicit event information improving TS generation is prior art when the product is known. |
| DeePEST-OS / DORTS, DOI 10.1038/s41467-026-72945-0 | Candidate geometries for reactive PES search | Reactive energy/force model and optimized TSs | GFN2-xTB plus learned correction on reactive data | TS search and cross-dataset validation | Delta learning on a semiempirical baseline is an enabling component, not headline novelty. |
| Classical dimer / gentlest-ascent methods | Geometry plus a local unstable direction estimate | First-order saddle search trajectory | Reverses or modifies force along a mode | Stationarity, one negative mode, and endpoint checks | The reflected-force idea is classical; only a tested learned interface could be distinctive. |

Primary-source identifiers and version notes are maintained in [`docs/sources.md`](../sources.md). Rows with a preprint identifier must be rechecked before submission for version or publication changes.

## Falsifiable novelty hypotheses

### H1 — constructive conservation has operational value

The projected/null-space event flow must improve strict decode validity or intended-event recovery relative to an otherwise matched unconstrained or penalty-only event model. Report rejection rates as outcomes, not hidden preprocessing.

Required comparison: unconstrained event flow, penalty-constrained event flow if implemented, and conservation-by-construction flow with matched data, capacity, candidate count, decoder, and postprocessing.

### H2 — dynamic event–geometry coupling has value beyond serial factorization

The bidirectional joint model must outperform a registered event-then-geometry baseline on held-out parent reactions after both receive the same candidate count, physical refinement, and final reference-validation budget. A shared encoder or two output heads alone does not satisfy this hypothesis.

Required comparison: independent branches, serial event→geometry, and joint event↔geometry. Turning coupling off must reproduce the registered serial control.

### H3 — the predicted reaction projector improves saddle refinement

Using the generated sign-invariant projector `P = u u^T` with the calibrated physical force must improve accepted intended-TS yield per calculator call relative to ordinary descent and an equal-budget classical initialization. Reference modes may score the result but may not construct the inference-time projector.

Required comparison: no guidance, ordinary force descent, bare-GFN2 saddle guidance, calibrated-potential descent, and calibrated-potential saddle guidance.

## Explicit non-claims

XTBFlow v0.1 does not claim to invent flow matching, SE(3)-equivariant message passing, electron conservation, Delta learning, semiempirical pretraining, force reflection, or transition-state optimization. It does not claim physical reaction dynamics, calibrated reaction probabilities, kinetic dominance, or historical occurrence in prebiotic Earth environments.

The current synthetic joint-flow smoke proves software execution only. The SPICE2 E/F result supports development of a calibrated potential only; it is not event or TS supervision. Until issues #58 and #59 pass their decision gates, the combined architecture remains a testable hypothesis.

## Current novelty hierarchy

1. **Headline candidate:** conservation-by-construction plus dynamically coupled event/geometry generation under a fixed physical-compute budget.
2. **Conditional secondary contribution:** generated sign-invariant reaction projector connected to equal-budget saddle-aware refinement.
3. **Supporting component:** GFN2 plus Delta learning, calculator accounting, and provenance controls.

The manuscript must demote or remove any level whose prespecified comparison fails. No wording such as “first”, “breakthrough”, or “general” is permitted without a separately documented search and matching evidence.

## Method novelty statement — draft, conditional on H1–H3

XTBFlow tests a conservation-preserving co-flow in which a projected electronic event state and an SE(3)-equivariant three-dimensional transition-state proposal exchange information throughout generation from reactant-visible inputs. The proposal also exposes a sign-invariant reaction projector to an independent, metered calibrated semiempirical potential. The contribution is not the individual use of flow matching, equivariance, Delta learning or reflected-force saddle search; it is the measurable advantage, if any, of this information-restricted coupling over matched unconstrained, serial and ordinary-refinement controls.

This paragraph is hypothesis language. It may move into the manuscript contribution list only after the corresponding real-data and equal-budget gates pass.

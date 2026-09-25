# mechai research handoff

Date: 2026-09-26.

This document transfers the useful, reusable evidence from mechai into the
xtbflow project. It is a curated handoff, not a full source or data migration.
Large data, weights, complete run directories, private host paths and raw
archives remain in the original project and are referenced only by their
source-relative evidence paths.

## Scientific task

The intended research contribution is a new model that proposes conserved
elementary reaction events together with three-dimensional transition-state
initial guesses from an independent reactant microstate and solvent context.
The model should solve a class of reaction-discovery problems under a fixed
physical-computation budget, then be used to study why neighboring oxygen
retention, methylation and deletion change Ala-SEt/nucleoside aminoacylation in
water.

The chemistry question is deliberately open. Acid/base populations, local
electronic and hydrogen-bond effects, conformational preorganization, solvent
and proton-transfer organization, donor hydrolysis, and product migration are
separate explanations. A single water chain, one low-energy cluster, or one
transition-state coordinate cannot establish the dominant solution mechanism.

## Conclusions that survived the current audits

1. **The project has a working engineering base, not a demonstrated AI
   chemistry gain.** UniTS-Lib and the official Formula-OOS baseline were
   checked; the first geometry residual comparisons did not show useful
   improvement.
2. **The 41-source-TS event ranking route is closed.** On 41 records from 19
   groups, the inputs and labels are conditioned by already searched source TS
   geometries. A strengthened rule outperformed the current learned ranking,
   showing a geometry shortcut rather than general reactant-only discovery.
3. **The smallest shared event--geometry prototype is reusable software only.**
   It has 1,409 parameters, a shared scalar compatibility score, invariant
   coordinate feedback, explicit masks and registered control arms. It is
   untrained and its gradients are not physical forces.
4. **Sampler cost wiring is reproducible.** Three seeds completed the same
   six-arm, 1,000-step pilot under the declared cap; the relative total-time
   variation was about 0.9%. This establishes cost and callback accounting, not
   geometry or chemical efficacy.
5. **The public geometry data do not yet support the intended joint task.**
   Transition1x has 10,073 R/TS/P rows but lacks product-free event, charge,
   multiplicity and solvent labels; 1,038 of its 1,073 published complement
   rows repeat training-side reactant identities, leaving a 34-row diagnostic
   without family certification.
6. **FlowER is an auxiliary graph/state resource, not a training trigger.**
   Its two releases passed bounded format, identity, network and BE audits, but
   the data contain product-selection effects, unknown multiplicities and no
   three-dimensional water environment. It is not evidence for aqueous
   mechanism discovery.
7. **The Ala-SEt/nucleoside validation protocol is ready as a contract, not as
   a result.** Five mechanism classes, perturbation groups and G0--G3 gates are
   written down; all physical gates remain closed and no target-system path or
   independent DFT conclusion exists.

## Reusable implementation

The `vendor/mechai_reusable/` snapshot contains:

- exact atom/electron conservation for explicit-H CHNOS endpoint states;
- event proposals that combine heavy-atom edits, electron edits and explicit
  proton transfers without inventing atoms or silently guessing charge;
- bounded water-relay enumeration with whole-tie budget handling and a clear
  distinction between heuristic ordering and physical feasibility;
- a sample manifest validator that keeps unknown, ambiguous, unavailable and
  blocked evidence explicit;
- the event--geometry compatibility score, terminal decoder and corresponding
  CPU tests.

The JSON schema and three illustrative task-mode manifests are copied under
`schemas/` and `configs/reusable/`. They are examples and contracts, not real
training records.

## What this handoff does not establish

- no flow-matching model has been trained here;
- no xTBloom, CP2K or other quantum-chemistry runtime has been installed or
  validated in the current environment;
- no new DFT/xTB calculation was created by this transfer;
- no independent reactant seed manifest or family-certified test split exists;
- no aqueous rate, barrier, flux, yield or unique mechanism has been inferred.

The next valid method experiment must first pin a calculator, generate same-
geometry semiempirical/reference energy-force pairs, freeze a source/family
split and compare the joint model against equally budgeted rules, serial
models and common physical post-processing.

## Evidence map

The machine-readable public-data limits are in
`configs/public_data_manifest.yaml`. Detailed source evidence remains in the
original mechai project under the paths listed in
`docs/MECHAI_REUSE_MANIFEST.json` and in the inherited reports named by the
asset index. The original working tree was dirty at source commit
`df20cc446f1d655ccb428fb3eeea1a7abe66357b`; the copied files are therefore a
content-addressed snapshot, not a claim that the old repository was clean.

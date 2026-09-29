# Exploration–confirmation isolation contract

Protocol ID: `xtbflow-exploration-confirmation/v0.1-draft`
Issue: #28
Status: draft; no confirmation batch is frozen or authorized by this document.

## Purpose

This contract allows exploratory architecture and application selection while preventing viewed or decision-influencing results from being relabelled as independent confirmation. It governs records, splits, access, freeze metadata, downgrade events and allowed claims.

## Record classes

### `exploratory-development`

May be used to choose data, architecture, hyperparameters, thresholds, candidate chemistry, physical protocols, budgets, figures or narrative. All failed attempts and negative families remain recorded. Once a record influences any decision, it remains development evidence permanently.

### `confirmatory-sealed`

Created only after the freeze package is complete. Labels, reference TSs, modes, active solvent assignments and validation outcomes are unavailable to proposal/training/model-selection paths. A sealed batch is not described as third-party validation unless a genuinely independent custodian and access process exist.

### `confirmatory-released`

A sealed batch becomes released at the prespecified aggregate release point. It may support only the frozen claims and analyses. Any subsequent tuning uses it as development evidence.

### `quarantine`

Identity, microstate, grouping, provenance, units, license, structure or access evidence is incomplete. Quarantined records enter neither development metrics nor confirmation.

## Group firewall

All frames, conformers, protonation/tautomer states, solvent/water arrangements, fidelities, augmentations and endpoint/TS records derived from one parent chemistry remain in one split group. Grouping precedence is:

1. `parent_reaction_id`;
2. `mechanism_family_id` for cross-family claims;
3. `microstate_family_id`;
4. `environment_family_id`;
5. structure-derived hashes used only to detect duplicates, never to fragment a parent group.

A claimed unseen-family test must contain an actually unseen event pattern or mechanism family, not only a new substituent or conformer.

## Inference and supervision firewall

The confirmatory proposal path may read only frozen reactant-visible fields. It may not read the correct product, reference TS, reference mode/projector, target-derived reactive atoms or waters, reference connectivity, barrier, label-derived stopping signal, or a manually chosen conformer based on confirmatory outcomes.

Training and development may use product/TS labels only through a separate supervision view. Validation software receives proposal outputs after sampling and writes results to an attempt ledger that the proposal process cannot query.

## Freeze package

Before `execution_authorized` can become true, one immutable manifest must specify:

- code commit and clean-tree state;
- model architecture/config and checkpoint hashes;
- calibrated-potential identity and admitted domain;
- application-selection record and chosen main/transfer systems;
- public input-index hash and sealed-label custody description;
- allowed inference fields and input-firewall test result;
- candidate counts, sampling seeds and duplicate policy;
- ordinary/guided refinement protocols and exact calculator-call accounting;
- reference method, convergence/mode/connectivity rules and adjudication policy;
- primary/secondary metrics, statistical unit and interval method;
- finite budgets, stop conditions and aggregate release rule.

Null or `TBD` freeze fields mean the batch is not frozen. A document timestamp alone is not a freeze.

## Access and decision log

Every access to sealed metadata or outputs records actor/process identity, timestamp, batch ID, fields accessed, reason and resulting decision. Automated validators may check hashes and schema without exposing labels. Same-owner storage and a changed agent prompt do not constitute independent custody.

## Downgrade rule

The entire affected batch becomes `exploratory-development` if any result is viewed before the prespecified release, a frozen choice is changed in response, unplanned per-case adjudication informs the model, or the batch is rerun selectively. The downgrade is append-only and records the trigger. A new confirmation claim requires a new unseen batch and split hash.

## Matched-budget execution

Strong-rule, serial and joint methods receive the same declared proposal count, eligible physical postprocessing, calculator-call ceiling, retry policy and final reference criteria. Shared-candidate-pool and open-proposal evaluations are reported separately. Failed SCF/searches, rejected candidates and adjudication calls remain charged and visible.

## Prespecified evaluation hierarchy

Primary method metric: unique reference-accepted intended TSs per independent parent reaction, accompanied by accepted intended TSs per total calculator call.

Reference acceptance requires a converged stationary point, exactly one relevant negative mode under the frozen threshold, and bidirectional endpoint connectivity under the frozen path protocol. Geometry RMSD, event similarity, semiempirical convergence and one negative eigenvalue without connectivity are diagnostic or intermediate evidence only.

Report parent/family-level paired estimates and uncertainty. Conformers, frames, candidates, random seeds and calculator retries are not independent chemical samples. Report abstention, invalid decode, duplicate, SCF, optimization, frequency, connectivity and budget-exhaustion failures separately.

## Public index and sealed material

[`data/manifests/confirmation_public_index.json`](../../data/manifests/confirmation_public_index.json) contains only permitted batch metadata, split/group hashes and release state. It must never contain correct products, TS coordinates, modes, active-water labels, barriers or validation outcomes while sealed.

Sealed labels may live outside the repository only when their custodian, checksum and access procedure are documented. A path that no execution environment can actually access is not a valid custody mechanism.

## Claim handling

Confirmation supports only the claims and metrics named before execution. Exploratory application screening may select one main system and one transfer line, but those screening results remain development evidence. Cross-family or broad chemistry language requires actual frozen transfer evidence.

New wet experiments are outside v0.1. Agreement with published experiments is retrospective consistency unless an independent prospective protocol says otherwise. A validated TS does not by itself establish dominant kinetics, solution free energy or historical occurrence.

## Current readiness

As of 2026-09-29 the contract and machine-readable draft exist, but the freeze package is incomplete, no confirmation records are indexed, and `execution_authorized` is false. This work therefore starts no calculation and produces no confirmatory scientific result.

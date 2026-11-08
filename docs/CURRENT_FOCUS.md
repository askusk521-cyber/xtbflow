# Current Research Focus — 2026-11-08

This file is the current handoff pointer for issue #129. It records the active
scientific question and the exact boundary of the evidence; it is not a claim
that the next panel or comparison has already run.

## Main question

Can a feedback-driven proposal complete the path

```text
reactant-only input → measured physical feedback → bounded proposal/action
→ common TS/mode/bidirectional-IRC certification → endpoint and cost settlement
```

under a fixed budget, and then add a strict certified channel? The first real
path has completed. The current observed increment is **zero**: 4101 retained
one strict channel, 4105 produced one new candidate, and 4106 completed its
bounded certification attempt with `attempted_unresolved` / `PROTOCOL_FAILURE`.
That unresolved candidate is not a chemical negative.

## Frozen interpretation

- 4101: two reference candidates, 1 strict channel, 1106 calculator calls.
- 4105: one feedback-driven candidate, 50 model forwards, 0 new calculator calls.
- 4106: 526 calculator primitives, 2112.462 wall seconds, 0.6603476
  CPU-core-hours, 0.5865953 GPU-hours, raw `PROTOCOL_FAILURE`, strict increment 0.
- The 4106 attempt is retained in the failure denominator and ledger. Any
  diagnostic retry must use a new attempt and independent budget and is excluded
  from same-panel tuning.
- No Agent-efficiency, J/P interaction, development-comparison, or independent-
  confirmation claim is supported by this run.

## Panel and protocol status

```text
panel_status: pending_freeze
panel_manifest_sha256: null
comparison_protocol_sha256: pending
```

The next candidate pool is the audited #146 development-test split: 45 records
in 9 parent/family groups from one Diels–Alder archive. Apply only the declared
CHNO/reactant-only/charge-multiplicity/mapping/identity/quarantine filters,
publish the eligible pool and manifest hash, deterministically choose five by
`manifest_sha256 + parent_id`, and exclude `ID63623` used by 4101. Fewer than
five eligible parents becomes a `pilot`. No method result may change this pool.

The first comparison is W08's 2×2: J versus strong serial/backtracking, crossed
with terminal/no-in-generation P versus in-generation P. Fixed, heuristic, and
feedback-Agent policy comparison follows only after this generation comparison
and a real closed loop.

## Active remote work

The handoff below reflects the live PR state at the time of this commit. Merged
PRs are retained as provenance; open drafts still require their own review gate.

- [PR #151](https://github.com/askusk521-cyber/xtbflow/pull/151): dual-clock
  software prototype, open draft, head `5544a4a`.
- [PR #152](https://github.com/askusk521-cyber/xtbflow/pull/152): bounded
  feedback-generation actions, open draft, head `ae47ed3`.
- [PR #153](https://github.com/askusk521-cyber/xtbflow/pull/153): narrow stable
  scientific and environment contract, open draft, head `e94b466`.
- [PR #154](https://github.com/askusk521-cyber/xtbflow/pull/154): terminal
  evidence projection, open draft, head `5f5304b`.
- [PR #155](https://github.com/askusk521-cyber/xtbflow/pull/155): this current
  handoff-pointer update, open draft; its head is the commit containing this list.
- [PR #156](https://github.com/askusk521-cyber/xtbflow/pull/156): joint-flow and
  target-free generation parent, merged at `7a95431` into its custom base.
- [PR #157](https://github.com/askusk521-cyber/xtbflow/pull/157): claim-evidence
  integrity, merged at `fe476a4` into `main`.
- [PR #158](https://github.com/askusk521-cyber/xtbflow/pull/158): run-status
  retry accounting, open draft, head `35fe823`, base `main`.
- [PR #159](https://github.com/askusk521-cyber/xtbflow/pull/159): model/projector,
  merged at `fb7a42a` after #156.
- [PR #160](https://github.com/askusk521-cyber/xtbflow/pull/160): generation/scoring,
  merged at `8653706` after #159.
- [PR #161](https://github.com/askusk521-cyber/xtbflow/pull/161): data schema/audit,
  merged at `fe93331` after #160.
- [PR #162](https://github.com/askusk521-cyber/xtbflow/pull/162): docs/evidence,
  open draft, current head `63ad6be`, base #161 `fe93331`.
- [PR #163](https://github.com/askusk521-cyber/xtbflow/pull/163): scheduler evidence,
  open draft, current head `a9af5f6`, base `main`.
- [PR #164](https://github.com/askusk521-cyber/xtbflow/pull/164): reaction-network
  contract, open draft, head `52a34dd`, base `main`.
- [PR #165](https://github.com/askusk521-cyber/xtbflow/pull/165): campaign/input
  contract, open draft, head `1c55f58`, base `main`.

The current release gates are the open PRs and their exact-head CI/evidence;
merged PRs above are historical dependencies rather than pending work.

## n2 status

There are no queued or running n2 jobs (`squeue -u "$USER"` is empty). Jobs
4101, 4105, and 4106 have terminal public reports and retained process/ledger
records. No rerun is required for this acceptance round. The next n2 command
must be created from the frozen panel and protocol manifest, in a clean
commit-named execution directory, with the environment and resource ceilings
recorded in its evidence file.

## Recent evidence

- 4101 reference report SHA-256:
  `746e9089b1333ac598d01d4ac107e5321b8e790346ba2ebb98bcac811f73f36f`
- 4105 generation report SHA-256:
  `be6ad07aa623df731a33d6ee7668ce8542cd0302d6793befcdcca767eb20d051`
- 4106 certification report SHA-256:
  `9b773ee98c3dcc154f6c45b2a2c98c777ef4cb15cfb0f823f6982500a755dd8a`
- Public terminal projection:
  `docs/evidence/live_feedback_generation_reference_terminal_20261108.json`
- Latest terminal projection commit: `5f5304bcded99f2711ce21a090392cc2d6db8973`.

## Next delivery

1. Freeze and publish the eligible-parent pool and panel manifest hash.
2. Freeze the W08 budget vector, method-arm configs, certification protocol, and
   stopping/retry rules.
3. Run the matched 2×2 development comparison and publish all arms' ledgers,
   strict-channel dedup records, unresolved units, and mechanism diagnostics.
4. Decide whether a specific bottleneck justifies a later module; do not add W04,
   W05, or W12 before W08 exposes that bottleneck.

## Claim limits and pending decisions

The current result is a real closed-loop software/chemistry execution with zero
new strict channels from the feedback proposal. It is not an efficiency result,
not a J/P interaction result, not an independent confirmation, and not a
cross-source generalization result. The unresolved failure stage remains
`unknown` in the public projection. No scientific decision is pending from the
current acceptance; the next decision is the pre-registered W08 comparison
after its panel and vector budget are frozen.

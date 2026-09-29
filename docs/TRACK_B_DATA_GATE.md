# Track-B reaction/event/TS data gate

Track B is the real-data gate for the joint event--geometry model. It is
separate from Track A energy/force calibration. SPICE energy/force records may
support a calibrated potential, but they must not be used as reaction-event or
transition-geometry supervision.

## Admission boundary

A non-quarantined Track-B record must reconstruct all of the following:

1. A reactant-visible graph and three-dimensional input geometry.
2. Explicit charge, multiplicity, microstate and atom mapping.
3. A product graph supervision label with frozen bytes and provenance.
4. Product coordinates with explicit units and atom correspondence, or an explicit
   `unavailable` mask that carries no placeholder geometry.
5. An event label with frozen representation and byte-level provenance.
6. A transition geometry with explicit coordinate units.
7. A pinned reference protocol locator/hash, software version and source revision.
8. Parent-reaction, reaction-family and split-group identities.
9. A completed overlap audit.

The contract is implemented in `xtbflow.data.track_b.TrackBRecord` and mirrored
by `schemas/track_b_reaction_ts.schema.json`.

## Deployment firewall

`TrackBRecord.reactant_view()` excludes product labels, event labels, transition
geometries and reference protocols. Inputs are rejected from admission when the reactant
geometry, environment or active species were selected from the reference
result. Confirmatory records require an independently generated reactant input;
development records may also use a declared reactant endpoint when its origin
is documented and is not reconstructed from the transition structure.

## Leakage policy

The audit checks exact input fingerprints, source record identities, exact
source-record payload hashes, parent reactions and split groups. Reaction
families are held out by default; the auditor only relaxes that check with an explicit command-line flag. Quarantined
records never participate in split statistics.

Run the current empty gate manifest with:

```bash
PYTHONPATH=src python scripts/audit_track_b_manifest.py \
  data/manifests/track_b_reaction_ts.v0.1.jsonl \
  --repository-base-commit abee6248daaf0c81cde2e676b0b139f4b5d26d1e \
  --output docs/evidence/track_b_manifest_audit_20260929.json
```

A zero-record result is intentional until a source satisfies the complete
contract. Missing units, electronic state or provenance must remain unknown;
they must not be inferred from common practice.

## Independent seeds and attempts

`IndependentReactantSeed` freezes the exact reactant microstate, geometry,
electronic state, environment, generation-protocol locator/hash and split
group and split role before downstream search. `audit_seed_leakage()` checks
parent, family, split-group and exact-input isolation before any search starts.
The pilot remains disabled in `configs/seeds/pilot.yaml`; enabling it requires
real records in `data/manifests/independent_seeds.v1.jsonl`.

`SearchAttempt` records the proposal separately from the observed event. Every
update appends a monotonically increasing version to
`data/manifests/search_attempts.v1.jsonl`; previous versions are immutable.
Calculator protocols carry method, software, version, units and a protocol
hash; any executed calls require a hashed raw log. This prevents an
unsuccessful intended channel from being rewritten as a negative example when a different channel or a numerical failure occurred.

## Source audit on 2026-09-29

The source comparison registry is
`configs/data/track_b_source_candidates_v1.json`; the reproducible audit is
`docs/evidence/track_b_source_audit_20260929.json`. Rebuild it with
`scripts/audit_track_b_sources.py` and explicit local asset paths; the report
stores only configured locators, basenames, sizes and hashes. Seven candidate
sources were compared and none is currently admitted:

- The local Kingfisher export has real searched events, but its input geometry
  was extracted from the archived transition structure and its water selection
  is not independent of that search.
- The local UniTS-Lib asset has useful structural fields, but coordinate units
  are not embedded in the inspected raw object array, the event conversion is
  not frozen under this contract, and some records exceed the current CHNOS
  scope.
- SPICE2 remains Track-A E/F calibration data, while USPTO-derived edits and
  FlowER graph states lack the required three-dimensional TS supervision.
- No pinned local Transition1x asset was available for a byte-level audit;
  the local RGD1 archive is missing its referenced per-record geometry assets.

These assets may remain diagnostic. They cannot be silently promoted to blind
reactant-only training or confirmatory evidence.

## Gate for issue #58

The first development corpus may be small, but every admitted row must carry a
product graph label and an explicit product-geometry availability mask. It must
contain at least enough held-out parents to run the serial-versus-joint comparison without sharing a
parent, exact input, source row or source-record payload across splits.
Training is blocked until the manifest reports at least one non-quarantined development split and the audit
has no leakage error. A schema or smoke test alone is not scientific evidence.

## Zero-state audit commands

```bash
PYTHONPATH=src python scripts/audit_independent_seeds.py \
  data/manifests/independent_seeds.v1.jsonl \
  --repository-base-commit abee6248daaf0c81cde2e676b0b139f4b5d26d1e \
  --output docs/evidence/independent_seed_manifest_audit_20260929.json

PYTHONPATH=src python scripts/audit_search_attempts.py \
  data/manifests/search_attempts.v1.jsonl \
  --repository-base-commit abee6248daaf0c81cde2e676b0b139f4b5d26d1e \
  --output docs/evidence/search_attempt_manifest_audit_20260929.json
```

Both reports currently contain zero records and explicitly block scientific
claims. This is preferable to filling the manifests with examples or inferred
metadata.

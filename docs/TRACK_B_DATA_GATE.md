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

### Literature candidate to independent-seed readiness

Issue #27's six-row literature registry is audited without copying it into this
branch. The exact source is commit `2baa5ce9dad0b8271abec7321e8a256acebe5d5e` with registry SHA-256
`e990807c009dc53137551a66e878cdbddbd61d2e82479ae6e62c97a7bb0cb30e`; the report is
`docs/evidence/origin_seed_readiness_audit_20260929.json`.

All six rows satisfy the literature-anchor fields, but none satisfies the
identity/state freeze or the `IndependentReactantSeed` contract. The audit
therefore reports 6 literature anchors, 0 identity/state-ready candidates, 0
independent-seed-ready candidates and 0 validated seed records. Manifest
population remains disabled and search is not authorized. The provisional
neutral-singlet values on the two aminooxazole anchors are explicitly ignored
because their status is not `confirmed`.

Every row is still blocked by quarantine admission, an unfrozen structure,
unconfirmed charge/spin, declared missing requirements, no license record and
no explicit independent-seed payload. Rebuild the bridge audit with the exact
source-registry bytes:

```bash
PYTHONPATH=src python scripts/audit_origin_seed_readiness.py \
  --registry "$ORIGIN_CANDIDATE_REGISTRY" \
  --registry-locator \
    "github:askusk521-cyber/xtbflow@2baa5ce9dad0b8271abec7321e8a256acebe5d5e:configs/science/origin_candidates_v0.1.yaml" \
  --registry-source-commit 2baa5ce9dad0b8271abec7321e8a256acebe5d5e \
  --repository-base-commit d2379270bf5b59a1523a55db63ab56d3418a382a \
  --output docs/evidence/origin_seed_readiness_audit_20260929.json
```

This audit is a one-way governance bridge only. It does not create geometries,
atom mappings, microstates, solvent snapshots or search attempts, and it cannot
promote provisional values into the independent seed manifest.

## Source audit on 2026-09-29

The source comparison registry is
`configs/data/track_b_source_candidates_v1.json`; the reproducible aggregate
audit is `docs/evidence/track_b_source_audit_20260929.json`. Seven candidate
sources were compared and none is currently admitted:

- SPICE2 remains Track-A E/F calibration data, while USPTO-derived edits and
  FlowER graph states lack the required three-dimensional TS supervision.
- The local Kingfisher export has real searched events, but its input geometry
  was extracted from the archived transition structure and its water selection
  is not independent of that search.
- The local UniTS-Lib asset has useful structural fields, but coordinate units
  are not embedded in the inspected raw object array, the event conversion is
  not frozen under this contract, and some records exceed the current CHNOS
  scope.
- The exact public Transition1x endpoint asset is now byte- and structure-
  audited. It contains aligned reactant, product and transition-state endpoint
  coordinates, but it lacks explicit formal charge, multiplicity, trusted
  Track-B event labels, certified reaction-family holdouts and an independently
  certified chemical atom mapping. Its pinned Zenodo record does not state an
  explicit redistribution license, so raw bytes are not copied into this
  repository.
- The local RGD1 archive is still missing its README-referenced per-record
  geometry assets.

Three cached assets were observed without copying their raw bytes into this
repository:

- Transition1x `train_rpsb_all.pkl`: 55,458,032 bytes, SHA-256
  `36078a96aaf476f762dd4f1cf63a3f598e59b9191e7c1b819c5b007793078f65`,
  MD5 `701a457634cce7a6cae5318e8cd18082`;
- `UniTS_Lib.npy`: 1,439,687,943 bytes, SHA-256
  `6a8fff071330603600276e200e65feef3ab5e4d285f81091a3eb0cc13bdb70e2`;
- Kingfisher `events.jsonl`: 429,234 bytes, 41 non-empty rows, SHA-256
  `ee277408552e4b10149b350902796100ee4ade7ba30e8b836dfdc557ee4da5ae`.

All observed sizes and hashes match the candidate registry. Rebuild the
aggregate source audit with explicit local paths rather than embedding
machine-specific locations:

```bash
PYTHONPATH=src python scripts/audit_track_b_sources.py \
  --config configs/data/track_b_source_candidates_v1.json \
  --asset transition1x_preprocessed="$TRANSITION1X_TRAIN" \
  --asset units_lib="$UNITS_LIB" \
  --asset kingfisher_ch2o_events_v2="$KINGFISHER_EVENTS" \
  --audit-date 2026-09-29 \
  --repository-base-commit 0116d45747ca474ecf4877d59752646e5fdd363e \
  --output docs/evidence/track_b_source_audit_20260929.json
```

All three observed assets remain diagnostic. Matching bytes do not resolve
missing independent reactant inputs, electronic state, product/event
conversion, family diversity, environment independence, atom mapping or
licensing requirements. None is admitted for blind reactant-only joint
training or confirmatory evidence.

### Transition1x endpoint audit

The specialized report is
`docs/evidence/transition1x_track_b_audit_20260929.json`. It verifies exact
size, SHA-256 and MD5 before using a NumPy-only restricted unpickler; no archive
or project code is executed. The full 10,073 records were checked. Atom counts
range from 4 to 23, all reactant/product/TS coordinate shapes and atomic-number
rows match, and the shared `rxn`, `formula` and `num_atoms` fields align. The
published `use_ind` list contains 9,000 indices with a 1,073-record complement.

Rebuild it with:

```bash
PYTHONPATH=src python scripts/audit_transition1x_track_b.py \
  --asset "$TRANSITION1X_TRAIN" \
  --audit-date 2026-09-29 \
  --repository-base-commit 0116d45747ca474ecf4877d59752646e5fdd363e \
  --output docs/evidence/transition1x_track_b_audit_20260929.json
```

This supports an endpoint-conditioned geometry diagnostic only. The field named
`charges` stores atomic numbers, not formal charge. The source has no explicit
multiplicity or trusted event label, and row-preserving endpoint correspondence
is not a certified chemical atom map. Coordinates must not be converted into a
Track-B event label without a separately frozen and validated graph-perception
and event-representation contract.

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

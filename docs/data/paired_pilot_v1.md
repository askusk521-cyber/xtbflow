# paired_pilot_v1

`paired_pilot_v1` is the first real event–TS pilot admitted through the unified
Track-B contract. It uses the pinned public Diels–Alder reaction-space archive
(Figshare article 29118509, version 5, archive SHA256
`4df57289edecdfc518da6aab147b704745a9f42901893671aab05a3f72bd73ed`) and the
audited adapter in `src/xtbflow/data/dft_da.py`.

The source contains 1,580 CSV rows. The adapter admitted 201 rows after checking
the mapped SMILES, XYZ row order, reconstructed endpoint connectivity, neutral
singlet state evidence from both monomer logs, CHNO scope, and non-empty mapped
bond edits. These records cover 41 parent groups and are split by deterministic
family identity into train, validation and test. The family identity rule is
`canonical_reactant_components_without_atom_maps_v1`: canonical reactant
components are compared after removing arbitrary atom-map numbers, while the
original mapped SMILES remains the calculation identity. The exact funnel and
blocker counts are in `data/manifests/paired_pilot_v1/admission_report.json`.
The frozen split contains 147 train, 9 validation and 45 test records; family
and parent groups remain whole across those splits. Its frozen assignment seed
is `dft-da-track-b-v1-family-invariant-1299`.

Each manifest row is a `TrackBRecord`; no third record type is introduced. The
manifest stores source locators and hashes. Raw archive bytes remain in the
external cache and are not committed. `scripts/build_paired_pilot.py` rebuilds
the manifest and audit artifacts from that cache.

The loader exposes two explicit views. `reactant_input` contains atomic numbers,
reactant coordinates, charge, multiplicity and masks. `training_supervision`
contains mapped bond-edit labels and TS coordinates. Product coordinates,
reference protocol metadata and other validation-oracle fields are excluded
from every returned batch. `input_view_fingerprint.json` records the per-row
and aggregate firewall fingerprints.

Each row now records the evidence boundary in `reference_protocol`: the
published TS-tools geometry level, the fact that per-record energy labels are
not supplied, the monomer-log state source, and the unresolved TS identity
mapping. Those qualifiers keep the rows usable for a development pilot while
preventing them from being described as independently revalidated DFT TS
labels.

This pilot is development evidence only. It does not claim confirmatory
performance, independent-reactant generalization, or equivalence to the
Reaction-QM electronic-structure level. Reaction-QM remains a separately
audited source: its coordinate map-order rule is source-code verified, but
endpoint local-map correspondence, elementary-step pairing and other gates are
not yet sufficient for admission.

`scripts/run_dft_da_blind_rollout.py` is the bounded reactant-only inference
entry point for issue #92. It records actual candidate attempts, hashes,
duplicates, failures, flow evaluations and parent-macro/bootstrap scores. It
does not call xTB or CP2K; the current pilot therefore has zero physical
validation calls and cannot establish a scientific joint-flow advantage.

# Legacy asset index and P0 handoff

Date: 2026-09-25.

This index records what the execution environment could actually inspect from
the existing `mechai` project. The old project remains outside this repository;
its dirty working tree, raw data, weights and full runs were not copied or
modified. Public-facing files use repository-relative evidence names and do not
publish private host paths.

## Accessible assets

| Asset | Evidence reused | Current role | Boundary |
| --- | --- | --- | --- |
| Event bookkeeping and bounded relay proposals | `src/mechai/data/events.py`, `src/mechai/data/proposals.py`, related CPU tests | Candidate implementation reference | Conservation is necessary bookkeeping, not a TS or mechanism certificate |
| UniTS-Lib / Formula-OOS | `configs/data/units_assets.json`, `docs/STATUS.md`, `reports/E1/`, `reports/E2/` | Official geometry baseline and adapter reference | Source-TS/endpoint conditioning and limited pilot coverage remain; no event discovery claim |
| Transition1x | `data/manifests/transition1x_public.json`, `reports/public/transition1x-v1/` | Endpoint geometry diagnostic | Missing event/electronic/solvent labels; original complement is not family-independent |
| FlowER v2 | `data/manifests/flower_v2_public.json`, `docs/FLOWER_STATE_CONTRACT_RESULTS.md` | Auxiliary graph-state and conservation audit | No automatic training; no full 3D solvent or multiplicity contract |
| Kingfisher CH2O | `data/manifests/kingfisher_ch2o.json`, `docs/EVENT_CORRESPONDENCE_RESULTS.md` | Searched-TS / solvent diagnostic | Source geometry and selected waters are a known shortcut |
| USPTO-15K | `data/manifests/uspto_core_v1.json` | Reactant-side net-edit diagnostic | Not a complete electron-flow or aqueous-mechanism dataset |
| RGD1 | `data/manifests/rgd1_public.json`, `docs/RGD1_DATA_GATE.md` | Metadata-only audit | Required mapped-SMILES and HDF5 assets are absent |

The machine-readable public subset is [public_data_manifest.yaml](../configs/public_data_manifest.yaml).

## Runtime and connection state

- The local mechai checkout was readable and had the expected current branch and
  dirty working tree. It was not imported into xtbflow.
- The documented n2 contract remains Slurm `main` with one `pro6000` GPU per
  GPU job, standard 8 CPU cores and 48 GB. A live SSH probe timed out during
  this handoff, so remote filesystem bytes, queue state and installed software
  are unverified.
- Local executable lookup found no `xtb`, `xtbloom`, `tblite`, CP2K or Slurm
  commands. No calculator smoke test, training, DFT, xTB or QC job was run.
- xTBloom source identity was checked through the public GitHub API at commit
  `2cbdf1db8661ccbd5cb7d3d4bfc868a848cbbff3`; the repository reports GPL-3.0.
  This is source metadata, not an installed-runtime verification.
- CP2K remains the planned first fresh-reference backend, but its executable,
  version, method and license protocol are not configured here.

## Missing P0 items

1. A reachable execution host or an updated host-local configuration.
2. An installed and pinned xTBloom build with energy/force unit and sign tests.
3. A pinned CP2K reference protocol and a separate reviewer for physical
   validation.
4. A real public-data split manifest and an independently generated reactant
   seed manifest; neither is fabricated by this handoff.
5. Measured calculator costs before setting the next stage's finite cap.

The absence of these items is an evidence gap, not a scientific negative.

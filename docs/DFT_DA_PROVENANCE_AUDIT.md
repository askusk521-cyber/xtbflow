# Diels--Alder geometry and label provenance audit

The Diels--Alder v5 archive has two different provenance layers.  Its
`dataset_xtb_final.csv` table contains `DG_act`/`DrG` values from the published
M06-2X/def2-TZVP refinement and `DG_act_xtb`/`DrG_xtb` values from the GFN2-xTB
reaction profiles.  The XYZ files in that archive do not contain a method tag.
The source paper describes the reaction-profile geometries as xTB geometries;
the separately published [DFT geometry archive](https://figshare.com/articles/dataset/DFT_dataset_M06-2X_def2-TZVP_/28768301)
contains the corresponding `_dft.xyz` files and provides a useful file-level
comparison, but a coordinate difference by itself is not a proof of an
optimization method.

Run the standard-library audit on n2 after checking out the commit that
contains this script.  The raw archives stay in the n2 cache and are never
written to the repository.

```bash
cd ~/xtbflow/xtbflow
git rev-parse HEAD
source ~/miniconda3/etc/profile.d/conda.sh
conda activate xtbflow
python scripts/audit_dft_da_provenance.py \
  --cache-root ~/.cache/xtbflow/dft-da-full-29118509-v1 \
  --dft-cache-root ~/.cache/xtbflow/dft-da-dft-source-28768301-v1 \
  --repository-base-commit "$(git rev-parse HEAD)" \
  --audit-date 2026-09-30 \
  --output ~/xtbflow-runs/dft-da-provenance-20260930.json \
  --allow-hard-failures
```

The audit is bounded by the finite source tables and files (one process, no
calculator calls, no network access, and less than ten minutes wall time).  It
records SHA-256 and size for every source XYZ/log, checks mapped element order,
records Gaussian/xTB state and termination evidence from both monomer logs,
compares every available counterpart in the independent DFT archive, and
checks the DFT energy columns against its 1,582-row CSV.  It emits archive-
relative locators only, so private cache paths cannot enter the evidence.  The
report also hashes the source geometry protocol descriptor separately from the
raw archive hash.

The file-level gate deliberately remains conservative:

* Main-archive XYZ method metadata is `unknown`; the source-level xTB
  declaration is retained as a development claim.  The monomer logs provide
  complete GFN2-xTB external evidence for the rows where both logs are present
  and clean, but do not prove the level of the aggregate reactant/product/TS
  files.
* The TS atom sequence agrees with the mapped reactant element sequence, but
  no TS XYZ contains explicit atom-map IDs or a source sidecar.  TS mapping is
  therefore `order_consistent_but_unmapped` and is quarantined for any claim
  that requires verified atom correspondence.
* The reactant is a source-declared endpoint.  The archive does not document an
  independently sampled reactant microstate or preparation protocol, so an
  independent-reactant claim remains `unknown`.
* Matching DFT energy values and distinct coordinates from the independent
  archive do not upgrade the main XYZ files to DFT geometries.  No TS
  frequency, IRC, optimization-convergence, or physical-validation claim is
  made by this audit.

The resulting JSON is evidence for provenance and claim limits, not a model
benchmark.  A Track-B manifest can be passed with `--manifest`; any record
whose protocol field appears to promote the geometry to DFT, or whose TS
evidence is only `derived_under_contract`, is reported as a review flag rather
than accepted as file-level proof.

The bounded n2 run used for this revision is summarized in
[`docs/evidence/dft_da_provenance_audit_20260930.json`](evidence/dft_da_provenance_audit_20260930.json).
It covered all 1,580 rows, found 1,580/1,580 DFT-energy CSV matches, found
1,580/1,580 TS element-order matches but 0 explicit TS maps, and recommends
quarantine for DFT-geometry and TS-map claims.  The summary records the source
archive hashes, reference archive hashes, host, environment, code identity,
timing, and unverified items without retaining raw bytes or private cache
paths.

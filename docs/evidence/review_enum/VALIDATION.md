# T1 verification and resources

## Scientific checks

- 386 training parents: all 2547 in-domain unique catalogue events were regenerated; missing=0. Four checkpoint files are hashed in the manifest, each complete and passed.
- Exact frozen training feasibility counts select b3f3, mean 34867.4 <=50000. Domain choice was committed before formal evaluation.
- Full-precision B0 completed-candidate recall reproduces PR107 to <1e-9 at n=1,2,4,8,16. `b0-reproduction.json` is archived by remote hash.
- All 342 formal screen parents and 62 dev parents completed exact enumeration. The screen set is formal/freeze.json, not the larger reserve manifest.
- Both frozen scoring ensembles produced finite scores for every screen event; all 342 score files are present.
- Every screen parent and each k passed the independent combinatorial hypergeometric check; small-count exhaustive-subset tests also pass.
- Primary AUC excludes k=16 (B0 seed-cell censoring >5%); seven parents have no uncensored seed at one retained point and are explicitly excluded. The paired estimate uses 335 parents in 66 formula clusters. Missing seeds are not coded as failures.
- Exact symmetry/canonicalization optimizations are covered by small-domain equality tests and real-parent count checks. The authoritative canonical_event implementation still constructs every channel ID.

## Resource accounting

`resources.json` includes measured CPU user+system time for completed runs and explicitly labeled conservative wall-time bounds for stopped attempts. The total conservative CPU allocation is 13.156347 core-hours, including a one-core-hour preparation allowance, below 50. These are not falsely presented as exact measured timings for interrupted processes.

GPU Slurm jobs: 2700 (10-minute limit, timed out), 2707 (20-minute limit, timed out), 2708 (completed in 524.01 seconds), 2710 (completed in 59.69 seconds). Allowing an extra minute of scheduler termination grace for each timed-out job gives a conservative GPU allocation of 2503.70 seconds = 0.695473 GPU-hours, below 2. CPU associated with scoring is also below the one-core-hour preparation allowance. Earlier CPU Slurm jobs:2691 feasibility timeout,2692 tests. Subsequent CPU-direct runs were authorized by the owner. No other session's job was canceled, no GPU ran outside Slurm, and no DFT ran.

## Replay

Analysis source62ecad0 was a clean pushed clone. Per-parent acquisitions resumed across listed source commits without changing numerical definitions; manifests name each source. Required input and raw output hashes are recorded in SHA256SUMS.remote. Results, report and plot were generated from complete raw files, not partial snapshots.

Full test trio at source-e6535e6 passed. Final evidence-commit full tests, exact result replay, and hash verification are recorded in the final PR update. The report has no claim of chemical certification; all findings remain exploratory.

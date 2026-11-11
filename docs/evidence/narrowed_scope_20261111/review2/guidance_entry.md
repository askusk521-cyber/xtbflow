# Review #194: real guidance entry and update-order checks

At `fbe0ad1b4d34ed1bea675fa1150735a4e5661d35`, both separate exact-source
entry checks passed on node2, in addition to 274 pytest tests/68 subtests,
56 reusable unittests, bootstrap 346 files and diff check. The later schema-only
commit versions the clarified numerical-control output; the entry driver itself
is unchanged. Final source `fb024e039ae116b5856886e3ae9981c44d56edfb` again
passed 274/68, 56 reusable tests and bootstrap 346 files; see the
[final-schema receipt](final_schema_validation.json). Publication receipts
replace private checkout prefixes in warning text with a variable and retain
the original receipt/log hashes; no numerical field changes.

The [score-entry receipt](score_entry.json) installs nonempty recording and
failing guides in e84e330's actual sampler. At alpha zero, each uses two
synthetic network forwards, zero guide evaluations/backwards, zero additional
charged work, and bitwise identical full trajectories/counters to no guide.
Positive-alpha controls execute two guide evaluations/two aggregate backwards
and reach the failing guide. The sampler's six three-member equivalent forward
and backward units are distinguished from those actual fixture counters.
Nine synthetic network forwards include the positive failing control.

The [physical-update receipt](physical_entry.json) uses ffca40f's actual
`recorded_rollout` callback/update with a recording synthetic network. One
bounded push changes geometry, leaves the first-step event equal to baseline,
is consumed by the next network call and changes its event. This is established
by actual execution order, not by constructing a replacement object that keeps
b unchanged. Padding, centering and conservation pass. The paired paths consume
four synthetic forwards and no physical backend call.

These 13 synthetic forwards are software checks, not additional trained-model
diagnostics or independent chemistry. The old 90 started/89 completed trained
forwards and FD/JVP results remain unchanged. Their two earlier booleans are
limited unguided/state-replacement controls; the old receipts are preserved.
This checks zero strength in the installed score path and lag in the physical
push path. It does not qualify every historical physics-driver factory or
establish that real physical guidance improves a chemical decision.

Exact source-file hashes, resource caps, real wall/CPU timing and claim limits
are in both receipts. Reproduction and the 60-second/16-synthetic-forward/
4-MiB-per-invocation bounds are in
[the entry contract](../../../research/GUIDANCE_ENTRY_REVIEW_20261111.md).
No new QC, GPU use, checkpoint load or research training occurred.

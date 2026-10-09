# Exact results restoration

The complete results.json exceeds the handoff 1 MB committed-file limit. Its original bytes remain on n2 at the path in packaging.json. The deterministic gzip archive restores those bytes exactly:

```bash
gzip -dc results.json.gz > /your/audit/directory/results.json
sha256sum /your/audit/directory/results.json
```

Compare the digest with raw_results_sha256 in packaging.json. All report references to results.json refer to this restored full file, not a reduced summary. Original analysis ran from clean pushed source-8b007e1 and replayed byte-identically before packaging. Mandatory T1 and DFT stop points are unrelated and remain unresolved; this package does not complete the whole handoff.

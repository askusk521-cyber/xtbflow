# V1a development disposition: HOLD_POWER

The supplied v2.1 protocol was advanced through training, real five-arm development
streams, score calibration, drift/replay controls, and complete-gate power planning.
All ten generators completed 60,000 steps and all six score members completed
20,000 steps. The formal screen and all quantum calculations remain unstarted.

The final plan uses actual three-training-seed development averages from 62 parents
in 21 formula groups. The primary difference SD is 0.21708 and its formula ICC is
0.15948. At the full reserved inventory of 344 parents / 72 groups, target joint
power is 81.485% under the estimated Gaussian nuisance and 81.925% under the
empirical-skew working model. The conservative planning guard gives 72.785% and
73.635%, below 80%. The guard increases SD by 10% and nonnegative ICC by 0.05;
these are disclosed analyst choices made before the first power scan, not literal
numeric requirements in the guide. Target effects and formal gate thresholds are
unchanged. The hold therefore reflects sensitivity to nuisance uncertainty, not a
formal rejection of H1 and not proof that 80% power is impossible.

Actual GPU controls show exact alpha-zero equality and exact full B2 event replay.
The maximum observed batch-size comparison error is 2.15e-6. All planned pulse
pairs are retained; matched-endpoint score calibration does not label unmatched
candidates with invented energies. The original raw HDF5 was not located in this
run; cached-data hashes and inherited raw-source provenance remain an explicit
limitation.

Final scientific implementation `5aa6f96` passed 341 pytest tests, 68 subtests,
56 reusable unittest tests, and the tracked-file inventory check. Post-test changes
are delivery documentation and aggregate evidence. Formal production wrappers,
rarity sampling, formal analysis and V1b source export were not entered and are
not claimed complete after the pre-screen hold. No PR was published or merged.

The complete Chinese report is in `DEVELOPMENT_RESULT_ZH.md`. Authoritative run
artifacts are listed in `EXECUTION.md`; the aggregate decision is stored under
`docs/evidence/v1a/development_decision.json`.

## 中文说明

工具和环境已经通过实际训练、生成与开发集验证，三个训练种子的模型全部完成。
在最多 344 个预留母体上，功效点估计略高于 80%，但加入合理的方差余量后约为 73%，因此本轮按保守规划停在正式采样前。
这个 `HOLD_POWER` 表示统计把握不够稳健，不表示机制假设已经被否定；正式集和 V1b 都没有启动。

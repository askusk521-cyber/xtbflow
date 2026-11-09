# Live feedback → existing independent reference service

This extends `scripts/certify_generated_candidates.py`, based on the existing
#137 service with its storage-monitor and worker-startup fixes. No new TS,
Hessian, IRC or endpoint optimizer is implemented. The service runs in its own
clean checkout and `xtbflow-qc` environment. The live producer stays at its
separately pinned source; no unreviewed branch merge is required.

`--live-feedback` selects the immutable artifact input, with
`--generation-config`, `--reactant-context`, `--config` and `--out` supplied
explicitly. The reference configuration freezes the live manifest/context byte
hashes, clean live source, input configuration, parent, seed, method and budget.
Both frozen arms are passed in order, with identical per-candidate caps; none is
selected using a reference outcome.

The handoff checks snapshot hashes, matched starting geometry, source/electronic
state, atom inventory and settled upstream ledgers. Only `snapshot_after.geometry`
crosses into the reference service. Proxy-refined geometries, energies, modes,
endpoints and legacy `CERTIFIED` booleans do not become candidate inputs. The
four-element live event vector is retained in snapshot provenance as guidance
scores; it has no mapped chemical decoder. Accordingly `planned_product_be` is
explicitly null under candidate contract v2, never a fabricated no-edit graph.

`build_reactant_context` reuses the source adapter's reactant-only SMILES/XYZ
graph audit. It checks source hashes, explicit charge/multiplicity and stable
atom-map order, builds the reactant BE matrix and enumerates at most 2048
reactant automorphisms. The source's product/TS/event fields are not used. An
unknown map, license, electronic state, graph or environment blocks the handoff.
The source is a declared reactant endpoint from a development archive, not an
independently prepared blind reactant or family-held-out confirmation input.

The independent protocol remains `xtbflow-reference-certification/1`: RKS
wB97X/6-31G*, geomeTRIC OptTS, final gradient and full Hessian, mass-weighted
internal modes, both IRC directions, endpoint optimization/frequencies and
mapped endpoint identity. Source/backend hashes and package/build versions are
part of the result/cache identity. Scalar xTB force checks cannot bypass this
chain. Existing numerical thresholds and channel equivalence are unchanged.

Upstream xTB feedback/proxy costs are retained separately from reference
SCF/gradient/Hessian primitive charges. Every reference candidate remains in
the denominator, including protocol failure, timeout, unresolved results and
budget exhaustion. A cached second call must return the identical record,
zero new primitives and zero new channels. A complete reference chain without
a proposal may identify an observed channel, but prediction-match fields stay
null. Same-endpoint results add no channel.

Run only on n2 in a bounded Slurm GPU allocation, using the exact command and
configuration recorded with the execution evidence. The reference budget is
separate from the earlier proxy budget. All raw coordinates, Hessians, paths,
request/result files and primitive journals remain in the private run directory.
Public evidence contains hashes, outcomes, cost totals and claim limits.

## 中文说明

本次接入复用已有独立参考认证服务，不重新编写过渡态或反应路径算法。输入会先核对
源码、配置、原子映射和账本，再把两条 arm 反馈后的几何分别送入相同协议和预算。
四维指导分数没有化学产物图含义，所以预测产物保持 null，不能偷偷用反应物图补齐。
真实计算失败也保留费用和分母；只有完整参考链及真实端点检查通过才记录认证通道。

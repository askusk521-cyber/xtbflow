# W11 campaign manifest

The W11 manifest freezes a common reactant-side input pool before any model or
calculator is run. It is a preparation and audit contract for the fixed-water
demonstration in issue #141; it is not a chemistry result.

## Inputs and identity

`atom_pool` lists every component in the campaign. Each entry records a role,
explicit-hydrogen flag, composition, charge, multiplicity, input locator,
SHA-256 and license record. Water components must have explicit hydrogens.
`candidates` bind independently prepared reactant geometries to parent and
family IDs and declare every water role as `participating`, `spectator` or
`unresolved`. A candidate with status `quarantine` may keep charge,
multiplicity or geometry hash as an explicit unresolved token; the validator
never defaults these fields.

The validator rejects product, transition-state, observed-event and sealed
label fields. Candidate `input_origin` must be `independent_reactant`, so a
reference geometry cannot be silently reused as a deployment input. The
manifest input fingerprint covers source, atom pool and candidates. Every
comparison arm must carry that same fingerprint and the complete candidate ID
set, which keeps rule, serial, joint and optional agent comparisons on common
inputs.

`reference_protocol` records the protocol identity and units, while `budget`
sets finite calculator-call, wall-time, CPU-core-hour and GPU-hour caps.
`deduplication` records the keys and whether de-duplication is shared across
strategies. No quantum-chemistry call is performed by the loader or validator.

## States and failure handling

`development_screening` is for candidate preparation and screening. `frozen`
requires an ISO-8601 timestamp, complete electronic state and geometry hashes,
non-quarantined candidates, and at least one participating water. `quarantine`
is an explicit holding state for incomplete or unverified inputs. A malformed
manifest fails validation; an unresolved candidate remains a visible
quarantine record and is never promoted implicitly.

The bounded CLI is:

```bash
PYTHONPATH=src python scripts/validate_campaign_manifest.py \
  configs/campaigns/w11_campaign_fixture.json
```

The checked-in fixture is synthetic and only proves contract behavior. It
does not select a real chemical system, establish a mechanism, certify a
transition state, or support a method comparison.

## 中文说明

W11 清单在运行模型和计算器之前冻结反应物输入池。每个组分都要写明角色、显式氢、组成、电荷、自旋、输入文件哈希和许可证，水分子的参与或旁观身份也要单独记录。候选输入只能来自独立准备的反应物，清单会拒绝产物、过渡态、实际事件和封存标签等目标信息。

清单可以处于开发筛选、冻结或隔离状态。隔离记录可以保留明确的 `unknown` 或 `unresolved`，但验证器不会替它们猜默认值；冻结状态必须给出完整的电子态、几何哈希和水角色。规则、串联、联合及可选 Agent 比较必须共享同一输入指纹和候选集合，预算字段限制调用次数、时间和 GPU/CPU 资源。

仓库中的示例是合成测试夹具，只验证软件合同，不代表真实化学选择、过渡态认证或方法优越性。正式筛选仍需在独立数据和固定协议下进行，并把失败和未解析输入保留在隔离状态。

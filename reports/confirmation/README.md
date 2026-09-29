# Confirmation reports

This directory is reserved for issue #28 confirmation outputs. It is intentionally empty of results.

A run may write here only when `configs/validation/confirmation_v0.1.yaml` has:

- `status: frozen`;
- `execution_authorized: true`;
- no null freeze-manifest fields;
- a non-empty public input index and split hash;
- immutable code, checkpoint, physical and reference protocol identities;
- finite proposal, calculator and reference budgets.

Every report must link its freeze-manifest hash, retain all failures and costs, and follow the aggregate release rule. Early access or post-result changes trigger the downgrade rule in `docs/validation/exploration_confirmation_contract.md`.

As of 2026-09-29 no batch is frozen, no compute is authorized and no result exists.

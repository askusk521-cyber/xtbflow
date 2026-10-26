# Workflow B software acceptance — 2026-10-02

This report is bound to `main` commit `a0c959358b988f220f66a3641c53bb9d997069da` and the machine-readable JSON beside it.

The validator passed the frozen six-arm software matrix, including the current-state unidirectional ablation and the conservation projection ablation. All learned controls shared the measured model state (`23,387` trainable parameters, identical state hash) and a zero calculator-call budget. The conservation residual for projected controls was at most `1.67e-16` on the fixture.

Validation also passed:

- focused workflow/joint/conservation tests: `22 passed`;
- reusable standard-library suite: `56 passed`.

This remains software evidence. No event/TS training, held-out comparison, calculator call, or chemistry qualification was performed. Issue #58 remains the blocker for scientific execution, and SPICE energy/force records were not used as event/TS supervision.

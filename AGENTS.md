# Repository Guidelines

## Project Structure & Module Organization

- `src/xtbflow/` contains the installable Python package: calculators, models,
  data contracts, runtime budgeting, training, sampling, and validation.
- `tests/` holds pytest contract tests; `tests/reusable/` contains the
  standard-library/PyTorch reusable suite.
- `configs/` stores protocols, resource limits, manifests, and model settings.
  `docs/` stores architecture, provenance, handoff, and machine-readable
  evidence. `scripts/` contains bounded audit and diagnostic entry points.
- `vendor/mechai_reusable/` is an imported snapshot; do not edit it in place.
  Treat `archive/` as historical material.

## Build, Test, and Development Commands

```bash
python -m pip install -e ".[test]"       # editable package and test tools
PYTHONPATH=src:vendor/mechai_reusable pytest -q
PYTHONPATH=src:vendor/mechai_reusable \
  python -m unittest discover -s tests/reusable -p 'test_*.py'
python scripts/validate_bootstrap.py      # verify tracked-file hashes
```

Run focused tests with a path, for example `pytest -q tests/test_calculator_contract.py`.
Update `bootstrap_inventory.json` with `python scripts/validate_bootstrap.py --refresh`
when an intentional tracked-file change is ready to commit.

## Coding Style & Naming Conventions

Use Python 3.10+, four-space indentation, type annotations, and small explicit
functions. Use `snake_case` for modules, functions, and variables; `PascalCase`
for classes; and descriptive protocol IDs. Keep calculator units, charge, and
multiplicity explicit at API boundaries. No formatter is enforced, so run
`git diff --check` and keep imports and error handling clear.

## Testing Guidelines

Name tests `test_*.py` and test functions `test_*`. Add contract tests for
boundary conversions, failure classification, budgets, and provenance. Run
both suites above plus focused tests for changed modules; do not claim numeric
or chemical qualification from mocks or smoke tests alone.

## Commit & Pull Request Guidelines

Use concise imperative subjects with the repository’s prefixes: `feat:`,
`fix:`, `test:`, `docs:`, or `chore:`. A pull request should explain the
behavioral change, link the relevant issue, list validation commands, and call
out evidence limits. Never commit credentials, private data, model weights, or
unbounded job configurations; record public provenance and bounded resources.

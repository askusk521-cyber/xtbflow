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

## GitHub Communication Language Requirement

For every issue, pull request, review comment, discussion, release note, or other project update posted on GitHub, keep the existing English explanation for AI and tooling, and add a clearly labeled `中文说明` section with at least three complete Chinese sentences for human readers. The Chinese explanation must state what changed or what is wrong, why it matters, and what action or expected result is needed. Use everyday Chinese that an undergraduate student can understand; explain unavoidable technical terms briefly and keep the English and Chinese sections factually consistent. Update both sections whenever the scope, status, or expected result changes.

中文说明必须让本科生也能直接看懂。请用至少三句完整的中文，说明改了什么或哪里有问题、为什么重要、接下来需要做什么或应该看到什么结果。遇到必须保留的技术名词，要顺手解释它的意思；英文和中文内容必须表达同样的事实。

## Repository Management: Human Merge Approval

All PR and branch merges require explicit approval from the repository owner
or a human reviewer explicitly delegated by the owner, unless that human
clearly states that approval is not required for the specified scope. This
applies to every target branch and merge method, including merge commits,
squash merges, rebase merges, and fast-forward integration.

Routine permission to develop, commit, push to working branches, test, or
manage PRs is not permission to merge. Passing CI, an AI review, silence, and
general instructions such as "continue" or "handle everything" do not count
as human approval or an explicit waiver. Do not enable auto-merge, invoke a
merge API/command, or use direct pushes/cherry-picks to bypass this gate.

Before merging, record the human approval or explicit waiver and its scope
in the PR, together with the current head SHA. New commits require renewed
approval unless an explicit human waiver covers the changed scope. Without
clear authorization, leave the PR open and report that approval is pending.
See [repository management](docs/REPOSITORY_MANAGEMENT.md) for the full policy.

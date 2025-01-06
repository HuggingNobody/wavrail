# Contributing to wavrail

Thank you for your interest in wavrail. This project welcomes
contributions that improve code quality, documentation, and tests.

## Development environment

```bash
uv venv
uv pip install --python .venv/bin/python -e ".[dev]"
make format
make test
```

## Style

- Code is formatted with `ruff format` and linted with `ruff check`.
- Type hints are required for new public functions and dataclasses.
- Tests live under `tests/` mirroring the package layout.
- Public APIs must include a short docstring describing units and
  invariants.

## Commit hygiene

- Each commit must leave the tree in a buildable state.
- Conventional commit subjects are preferred (`feat:`, `fix:`,
  `docs:`, `test:`, `refactor:`, `chore:`).
- Do not combine unrelated changes into a single commit.

## Reporting bugs

Open a GitHub issue with a minimal reproduction script and the
output of `wavrail --version`. Bug reports must not include
private data.

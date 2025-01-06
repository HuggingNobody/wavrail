SHELL := /bin/sh
.SHELLFLAGS := -ec
PY    := .venv/bin/python
RUFF  := .venv/bin/ruff
PIP   := uv pip --python .venv/bin/python

.PHONY: build test test-all format format-check lint typecheck release clean help

help:
	@echo "wavrail Makefile targets:"
	@echo "  build         Build a wheel without isolation (.venv/bin/python -m build --wheel --no-isolation)."
	@echo "  test          Run pytest with the fast marker filter (-m 'not slow')."
	@echo "  test-all      Run the entire pytest suite (slow and torch-extra included)."
	@echo "  format        Apply ruff format and ruff check --fix to src/tests/examples/scripts."
	@echo "  format-check  Verify ruff format --check and ruff check exit 0."
	@echo "  lint          Run ruff check on src/tests/examples/scripts."
	@echo "  typecheck     Run mypy against the package."
	@echo "  release       Build the wheel and verify it installs cleanly."
	@echo "  clean         Remove build artifacts."

build:
	$(PY) -m build --wheel --no-isolation

test:
	$(PY) -m pytest -q -m "not slow"

test-all:
	$(PY) -m pytest -q

format:
	$(RUFF) format src tests examples scripts
	$(RUFF) check --fix src tests examples scripts

format-check:
	$(RUFF) format --check src tests examples scripts
	$(RUFF) check src tests examples scripts

lint:
	$(RUFF) check src tests examples scripts

typecheck:
	$(PY) -m mypy src/wavrail

release: build
	$(PY) scripts/check_package.py

clean:
	rm -rf build dist .pytest_cache .ruff_cache .mypy_cache
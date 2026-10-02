# Shortcuts for common tasks. Each target is one uv command, listed in CONTRIBUTING.md.
# Windows without make: run the uv command shown for the target.

.PHONY: help setup check fast unit filecheck test lint fmt types example new-op new-pass clean

help:
	@echo "setup      install dependencies into .venv"
	@echo "check      everything CI runs (format, lint, types, unit, filecheck)"
	@echo "fast       check without pyright"
	@echo "unit       pytest only"
	@echo "filecheck  lit/FileCheck tests only"
	@echo "fmt        auto-format and auto-fix lint"
	@echo "example    run df-opt on examples/matmul_2x2.mlir"
	@echo "new-op     NAME=map TIER=a   scaffold a new op"
	@echo "new-pass   NAME=df-place     scaffold a new pass"

setup:
	uv sync

check:
	uv run python scripts/check.py

fast:
	uv run python scripts/check.py --fast

unit:
	uv run pytest -q

filecheck:
	uv run lit -v tests/filecheck

test: unit filecheck

lint:
	uv run ruff format --check .
	uv run ruff check .

fmt:
	uv run ruff format .
	uv run ruff check --fix .

types:
	uv run pyright

example:
	uv run df-opt examples/matmul_2x2.mlir

new-op:
	uv run python scripts/scaffold.py op $(NAME) --tier $(TIER)

new-pass:
	uv run python scripts/scaffold.py pass $(NAME)

clean:
	rm -rf .pytest_cache .ruff_cache .lit_test_times.txt
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

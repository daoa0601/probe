.PHONY: sync check test format doctor plan

sync:
	uv sync --all-extras

check:
	uv run ruff check .
	uv run ruff format --check .
	uv run basedpyright
	uv run pytest --cov

test:
	uv run pytest

format:
	uv run ruff check --fix .
	uv run ruff format .

doctor:
	uv run probe doctor

plan:
	uv run probe sweep --dry-run configs/gemma4-26b-a4b-baseline.toml

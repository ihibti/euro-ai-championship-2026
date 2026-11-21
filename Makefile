.PHONY: setup lint format test smoke check-env papers clean

setup:
	uv sync --extra dev

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .
	uv run ruff check --fix .

test:
	uv run pytest

smoke:
	uv run pytest -m smoke
	uv run python scripts/run_experiment.py --config configs/example.toml --no-env-probe

check-env:
	uv run python scripts/check_env.py

papers:
	bash scripts/fetch_papers.sh

clean:
	rm -rf outputs .pytest_cache .ruff_cache

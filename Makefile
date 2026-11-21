.PHONY: setup lint format test smoke check-env papers clean data score board check submit

TASKS = watermark_localization, model_equality or unlearning
require = $(if $($(1)),,$(error make $@ needs $(1)=...; see Commands in AGENTS.md. Tasks: $(TASKS)))
TORCH_FOR_TASK = $(if $(filter model_equality,$(TASK)),,--extra torch)
TORCH_FOR_ANSWER = $(if $(filter %.pt %.pth,$(ANSWER)),--extra torch,)

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

data:
	$(call require,TASK)
	uv run --extra data python scripts/data.py --task $(TASK)

score:
	$(call require,TASK)
	$(call require,ANSWER)
	uv run $(TORCH_FOR_ANSWER) python scripts/score.py --task $(TASK) --answer "$(ANSWER)" $(if $(NOTES),--notes "$(NOTES)",) $(if $(TRUTH),--truth "$(TRUTH)",)

board:
	$(call require,TASK)
	uv run python scripts/board.py --task $(TASK)

check:
	$(call require,TASK)
	$(call require,FILE)
	uv run $(TORCH_FOR_TASK) python scripts/submit.py --task $(TASK) --file "$(FILE)" --check

submit:
	$(call require,TASK)
	$(call require,FILE)
	uv run --extra api $(TORCH_FOR_TASK) python scripts/submit.py --task $(TASK) --file "$(FILE)"

clean:
	rm -rf outputs .pytest_cache .ruff_cache

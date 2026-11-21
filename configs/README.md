# configs/

TOML files consumed by `scripts/run_experiment.py`. Required keys: `experiment.name`, `experiment.task`; optional `experiment.seed` (default 0), `experiment.deterministic`, `experiment.output_dir`. Task parameters go under `[task]`.

Override from the CLI: `--set task.separation=2.0 --set experiment.seed=7` (values parsed as TOML literals).

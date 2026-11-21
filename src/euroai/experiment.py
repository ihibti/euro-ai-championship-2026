"""Minimal, task-agnostic experiment runner.

A task is a function `(task_config, rng) -> metrics dict`. The runner seeds
everything, creates a timestamped run directory, writes `manifest.json`
(config, seed, git state, environment) before the task runs and
`metrics.json` after it. Register real tasks in `TASKS` once the challenge is
known; `synthetic_detection` exists only to exercise the pipeline.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from euroai.config import get_path
from euroai.env import detect_environment
from euroai.metrics import best_threshold_accuracy, roc_auc, tpr_at_fpr
from euroai.provenance import build_manifest, git_state, make_run_dir, write_json
from euroai.seeding import derive_seed, make_rng, seed_everything
from euroai.stats import bootstrap_ci

Task = Callable[[Mapping[str, Any], np.random.Generator], Mapping[str, Any]]


@dataclass(frozen=True)
class RunResult:
    run_dir: Path
    metrics: dict[str, Any]


def synthetic_detection(cfg: Mapping[str, Any], rng: np.random.Generator) -> dict[str, Any]:
    """Gaussian scores for negatives N(0,1) and positives N(separation,1)."""
    n_pos = int(cfg.get("n_positive", 500))
    n_neg = int(cfg.get("n_negative", 500))
    separation = float(cfg.get("separation", 1.0))
    target_fpr = float(cfg.get("target_fpr", 0.01))
    scores = np.concatenate([rng.normal(separation, 1.0, n_pos), rng.normal(0.0, 1.0, n_neg)])
    labels = np.concatenate([np.ones(n_pos, dtype=int), np.zeros(n_neg, dtype=int)])
    ci = bootstrap_ci(
        roc_auc,
        labels,
        scores,
        n_resamples=int(cfg.get("bootstrap_resamples", 500)),
        rng=rng,
        strata=labels,
    )
    return {
        "auc": ci.estimate,
        "auc_ci95": [ci.low, ci.high],
        "auc_theoretical": 0.5 * (1.0 + math.erf(separation / 2.0)),
        f"tpr_at_fpr_{target_fpr:g}": tpr_at_fpr(labels, scores, target_fpr),
        "best_threshold_accuracy": best_threshold_accuracy(labels, scores),
        "n_positive": n_pos,
        "n_negative": n_neg,
    }


TASKS: dict[str, Task] = {"synthetic_detection": synthetic_detection}


def run_experiment(
    config: Mapping[str, Any],
    *,
    tasks: Mapping[str, Task] = TASKS,
    output_root: str | Path | None = None,
    now: datetime | None = None,
    probe_environment: bool = True,
) -> RunResult:
    name = str(get_path(config, "experiment.name"))
    task_name = str(get_path(config, "experiment.task"))
    seed = int(get_path(config, "experiment.seed", 0))
    deterministic = bool(get_path(config, "experiment.deterministic", False))
    if task_name not in tasks:
        raise KeyError(f"unknown task {task_name!r}; known: {sorted(tasks)}")

    started_at = now or datetime.now(timezone.utc)
    seed_report = seed_everything(seed, deterministic=deterministic)
    root = output_root or get_path(config, "experiment.output_dir", "outputs")
    run_dir = make_run_dir(root, name, started_at)
    manifest = build_manifest(
        config=config,
        seed=seed,
        environment=detect_environment(probe=probe_environment).to_dict(),
        git=git_state(Path.cwd()),
        started_at=started_at,
        extra={"task": task_name, "seed_report": seed_report.to_dict()},
    )
    write_json(run_dir / "manifest.json", manifest)

    rng = make_rng(derive_seed(seed, "task", task_name))
    metrics = dict(tasks[task_name](config.get("task", {}), rng))
    write_json(run_dir / "metrics.json", metrics)
    return RunResult(run_dir=run_dir, metrics=metrics)

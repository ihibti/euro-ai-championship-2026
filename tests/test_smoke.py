"""End-to-end checks that the scaffold works on a CPU-only machine."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from euroai.config import load_config
from euroai.experiment import run_experiment

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.smoke


def test_run_experiment_writes_manifest_and_metrics(tmp_path):
    config = load_config(ROOT / "configs" / "example.toml")
    result = run_experiment(config, output_root=tmp_path, probe_environment=False)
    manifest = json.loads((result.run_dir / "manifest.json").read_text())
    metrics = json.loads((result.run_dir / "metrics.json").read_text())
    assert manifest["seed"] == 1234
    assert manifest["task"] == "synthetic_detection"
    assert abs(metrics["auc"] - metrics["auc_theoretical"]) < 0.05
    assert metrics["auc_ci95"][0] <= metrics["auc"] <= metrics["auc_ci95"][1]


def test_runs_are_reproducible(tmp_path):
    config = load_config(ROOT / "configs" / "example.toml")
    first = run_experiment(config, output_root=tmp_path, probe_environment=False)
    second = run_experiment(config, output_root=tmp_path, probe_environment=False)
    assert first.run_dir != second.run_dir
    assert first.metrics == second.metrics


def test_cli_scripts(tmp_path):
    run = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_experiment.py"),
            "--config",
            str(ROOT / "configs" / "example.toml"),
            "--set",
            "task.separation=2.0",
            "--output-root",
            str(tmp_path),
            "--no-env-probe",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(run.stdout)["metrics"]["auc"] > 0.85

    env = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check_env.py"), "--no-probe"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "python_version" in json.loads(env.stdout)

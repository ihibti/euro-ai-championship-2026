# euro-ai-championship-2026

Team workspace for the European AI Championship 2026 hackathon (CISPA EU AI Championship 26–27, JSC project `training2666`).

**Status:** this is scaffolding only. **We have not chosen a project.** The challenge, data and rules are still TBD (see below). Assumptions and open questions live in [NOTES.md](NOTES.md).

## Quickstart (local, CPU only)

You need [uv](https://docs.astral.sh/uv/). It installs Python 3.12 into the project `.venv` and never touches global packages.

```
make setup        # uv sync --extra dev     (numpy + pytest + ruff only)
make test         # uv run pytest
make lint         # ruff check + ruff format --check
make smoke        # end-to-end synthetic run, writes outputs/<timestamp>_<name>/
make check-env    # JSON report: python, packages, GPU, Slurm, loaded modules
make papers       # (re)download reference PDFs and verify SHA-256
```

Without `make` (these lines also work in nushell):

```
uv sync --extra dev
uv run pytest
uv run python scripts/run_experiment.py --config configs/example.toml --set task.separation=2.0
python3 scripts/check_env.py --no-probe
```

`scripts/check_env.py` uses only the standard library and needs no install, so you can run it on a fresh machine before any venv exists.

**pip fallback** (for example with a module-provided Python on JURECA): `python -m venv .venv`, then `.venv/bin/pip install -e ".[dev]"`.

### Optional dependency groups

These are declared in `pyproject.toml` and pinned in `uv.lock`. **None are installed by default.**

| Extra | Contents | Use |
|---|---|---|
| `dev` | pytest, ruff | always |
| `torch` | torch | GPU work |
| `vision` | torch, torchvision, pillow | image tasks, e.g. watermarking |
| `hf` | torch, transformers, datasets, accelerate, safetensors | LLM/NLP tasks, e.g. model auditing |
| `sci` | scipy, scikit-learn, pandas | classical baselines, MIA attack models |
| `notebook` | jupyterlab, ipykernel, matplotlib | local notebooks |

Install extras with `uv sync --extra dev --extra torch` (add more `--extra` flags as needed). Install only what the chosen task needs.

## Layout

```
src/euroai/         task-independent utilities (numpy only)
  seeding.py        seed_everything, derive_seed, make_rng, torch_worker_init
  env.py            read-only environment detection (stdlib, py3.8+)
  config.py         TOML configs + dotted --set overrides
  provenance.py     run dirs, manifest (config/seed/git/env), sha256_file
  experiment.py     run_experiment(config) + task registry (synthetic_detection demo)
  metrics.py        roc_auc, roc_curve, tpr_at_fpr, best_threshold_accuracy,
                    bit_accuracy, binomial_tail_pvalue, psnr
  stats.py          permutation tests, unbiased MMD², MMD permutation test, bootstrap_ci
  kernels.py        prompt-masked Hamming kernel (as in Gao et al. 2025 code)
  data.py           seeded / stratified disjoint index splits
  submission.py     validated id,value CSV writer (format TBD)
  tracking.py       offline JSONL metric logger
  templating.py     <PLACEHOLDER> rendering that refuses unfilled templates
scripts/            run_experiment.py, check_env.py, render_template.py, fetch_papers.sh
configs/            example.toml (pipeline smoke config)
tests/              unit tests + smoke tests (pytest -m smoke)
notebooks/          00_quickstart.ipynb (outputs cleared; commit without outputs)
papers/             reference-paper briefs (PDFs gitignored, fetch with `make papers`)
jsc/                JURECA connection guide, ssh config, sbatch + env templates
data/               gitignored; see data/README.md
docs_and_plans/     guides/ (references), work/{todo,doing,done}/ (plans)
```

### Adding a real experiment

1. Write a task function `(task_cfg, rng) -> dict` and register it in `euroai.experiment.TASKS`. Better: put the logic in its own module and register only a thin wrapper.
2. Copy `configs/example.toml` to a new config and set `experiment.task` to the new task.
3. Run it with `scripts/run_experiment.py --config ...`. Each run writes `manifest.json`, `metrics.json` and any extra files to `outputs/`.
4. Keep the logic pure where possible (data in, numbers out) and test it with synthetic inputs.

## Challenge — TBD

| Item | Status |
|---|---|
| Task / problem statement | **TBD** |
| Dataset (source, licence, size, personal data?) | **TBD** |
| Evaluation metric and scorer | **TBD** |
| Submission format and endpoint | **TBD**. Paris slides showed `id,label` with probability in [0,1], posted to a leaderboard server. Unconfirmed for this event. |
| Rules: external data, pretrained models, team size, compute limits | **TBD** |
| Deadlines | **TBD** |
| Judging criteria | **TBD** |

## Reference papers

These three papers were provided by the organizers. We do not assume any of them defines the challenge. Briefs are in [papers/README.md](papers/README.md).

1. Souček et al., *Pixel Seal: Adversarial-only training for invisible image and video watermarking*, [arXiv:2512.16874](https://arxiv.org/abs/2512.16874)
2. Gao, Liang, Guestrin, *Model Equality Testing: Which Model Is This API Serving?*, [arXiv:2410.20247](https://arxiv.org/abs/2410.20247)
3. Kurmanji et al., *Towards Unbounded Machine Unlearning*, [arXiv:2302.09880](https://arxiv.org/abs/2302.09880)

[docs_and_plans/guides/baseline-options.md](docs_and_plans/guides/baseline-options.md) lists possible baselines for each type of challenge.

## Compute (JSC JURECA-DC)

The full, source-referenced guide is [jsc/README.md](jsc/README.md). Short version:

- **Login:** `ssh <judoor-user>@jureca.fz-juelich.de`.
  - You need an Ed25519 key **with a passphrase**, uploaded in JuDoor with a `from="…"` IP restriction. It becomes active within 15 minutes.
  - MFA is mandatory, so SSH asks for a TOTP code. Use `jsc/ssh_config.template`; its ControlMaster settings avoid typing the TOTP for every connection.
- **Project:** `jutil env activate -p training2666` exports `$PROJECT` and `$SCRATCH`. Work in `$PROJECT/$USER`. Caches and large data go to `$SCRATCH`, which is purged 90 days after last access.
- **GPU node:** 4× A100 40 GB and 128 cores. Allocation is exclusive and charged per node-hour.
  - Partitions: `dc-gpu` (≤24 h, no internet) and `dc-gpu-devel` (≤2 h, has internet).
  - Download models and data on login nodes, then run offline.
- **Jobs:** render `jsc/jureca_gpu.sbatch.template` with `scripts/render_template.py`, then run `mkdir -p logs` and `sbatch`. The reservation name and budget account must be confirmed first.
- **Jupyter:** https://jupyter.jsc.fz-juelich.de, logging in with **JSC Account**. This has not been tested.
- **Data constraint:** stay within the approved project. **Do not process personal data.** JSC requires a GDPR assessment plus a signed addendum for any personal or pseudonymised data.

## Reproducibility conventions

- Every run records its config, seed, git commit and dirty flag, and environment in `manifest.json`.
- Use `derive_seed(base, "purpose", ...)` for child seeds (splits, workers, bootstrap) instead of reusing one seed everywhere.
- Record each dataset's SHA-256 (`euroai.provenance.sha256_file`) in NOTES.md when it arrives.
- `uv.lock` pins every dependency, including the optional extras.
- Commit notebooks without outputs.

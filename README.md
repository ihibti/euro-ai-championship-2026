# euro-ai-championship-2026

Team Begg's workspace for the European AI Championship 2026 hackathon (CISPA EU AI Championship, Paris, 21 and 22 November 2026; compute at JSC, project `training2666`).

## Results

Public leaderboard scores at the close of the event. The final ranking uses the hidden 70% of each task's data.

| Task | Public score | Method in one line | Code and records |
|---|---|---|---|
| Watermark localisation | 0.7444 | one segmentation network per watermark method, trained on synthetic composites made with the organizers' embedders and scored on a held-out fifth of the real images | `tasks/watermark_localization/` |
| Model equality | 1.00 | each endpoint is identified as a named variant of a public base model (system prompt, sampling setting, watermark, adapter); fp8 copies are found with a likelihood-ratio test against a simulated fp8 copy | `tasks/model_equality/` |
| Unlearning | 0.3101 | fine-tune towards a cross-fitted stand-in for the retrained model, rescale the last layer, average the weights of several runs | `tasks/unlearning/` (write-up: `method.md`, log: `night.md`) |

- **Methods, evidence, what failed and limits:** [docs_and_plans/guides/methods-and-results.md](docs_and_plans/guides/methods-and-results.md).
- **How the work was done:** most of it ran overnight, by coding agents working under limits the team lead set beforehand (one agent alone could submit or use the cluster). The rules they followed are in [AGENTS.md](AGENTS.md); the log of the night is [docs_and_plans/work/doing/night-plan.md](docs_and_plans/work/doing/night-plan.md).
- **Each task's statement** is in `tasks/<task>/README.md`. Assumptions and open questions are in [NOTES.md](NOTES.md).

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

Task commands (details in [AGENTS.md](AGENTS.md)):

```
make data   TASK=<task>                          download into data/<task>/ (watermark: also unzips into release/)
make score  TASK=<task> ANSWER=<file> NOTES=...  local score: prints score=<x>, writes tasks/<task>/results/*.json
make board  TASK=<task>                          result records, best first
make check  TASK=<task> FILE=<file>              every submission check, without submitting
make submit TASK=<task> FILE=<file>              submit: a human at a terminal, 30-minute cool-down, typed confirmation
uv run --extra api python scripts/me_query.py --prompts <file> [--endpoints ref|ep|<list>] [--spend]
uv run --extra api python scripts/me_query.py --budget    answers left on every endpoint
```

Secrets (`HACKATHON_API_KEY`, `HF_TOKEN`) come from the environment or from `~/.config/euroai/.env` (mode 600), never from the repo. See `.env.example`.

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
| `api` | requests | `make submit`, `scripts/me_query.py` |
| `data` | huggingface_hub | `make data` |
| `torch` | torch | GPU work |
| `vision` | torch, torchvision, pillow | image tasks, e.g. watermarking |
| `hf` | torch, transformers, datasets, accelerate, safetensors | LLM/NLP tasks, e.g. model auditing |
| `sci` | scipy, scikit-learn, pandas | classical baselines, MIA attack models |
| `notebook` | jupyterlab, ipykernel, matplotlib | local notebooks |

Install extras with `uv sync --extra dev --extra torch` (add more `--extra` flags as needed). Install only what the chosen task needs.

## Layout

```
AGENTS.md           instructions for every agent: map, commands, hard rules, work contract
tasks/<task>/       README.md (full statement), upstream/ (organizer files, unmodified),
                    results/ (make score records), submissions/ (make submit records)
  model_equality/answers/   every endpoint answer collected; never delete
src/euroai/         shared utilities (numpy only; torch, timm, requests imported lazily)
  challenge.py      the three tasks: dataset, server task id, file rules
  records.py        one JSON record per scoring run or submission; the board
  me_score.py       model-equality practice MCC, as the organizers' score_practice.py
  wm_score.py       watermark per-image mIoU, as the organizers' wmloc/utils.py
  validators.py     local submission checks for the three tasks
  credentials.py    secrets from the environment or ~/.config/euroai/.env
  gate.py           the submission gate behind make submit and make check
  me_query.py       budgeted, resumable model-equality querying
  fetch.py          data locations and zip extraction
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
scripts/            score.py, board.py, data.py, submit.py, me_query.py (thin CLIs over src/euroai),
                    run_experiment.py, check_env.py, render_template.py, fetch_papers.sh
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

## Challenge

Three tasks, all scored at http://34.147.229.55/leaderboard_page, where our team name is **Begg**. The hackathon ends at 2026-11-22 12:00 Europe/Paris. Each task's full statement is in `tasks/<task>/README.md`.

| Task | Goal | Metric | Local score |
|---|---|---|---|
| [watermark_localization](tasks/watermark_localization/README.md) | Mark the watermarked cells (128 x 128 grid) of 1,020 test images | mean over images of the two-class mIoU | exact, on the 510 labelled training images |
| [model_equality](tasks/model_equality/README.md) | Give family and model labels for 33 LLM endpoints | 0.1 · MCC_family + 0.9 · MCC_model over 528 pairs | exact, on the 9 free reference endpoints |
| [unlearning](tasks/unlearning/README.md) | Make a ViT-B/16 forget 3,500 training images | forget score × retain score; 0 below 70% validation accuracy | proxy only |

The organizers have not stated rules on external data, pretrained models or paid APIs, nor the submission limits for watermark and unlearning. `make submit` applies the model-equality limit (one per 30 minutes, failures included) to all three.

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
- **Team folder:** `/p/scratch/training2666/hibti1/Begg/`, one subfolder per task with its own `.venv` (see each task's README, On JURECA).
- **Jobs:** render `jsc/jureca_gpu.sbatch.template` with `scripts/render_template.py`, create the task's `output/`, then `sbatch`. It carries the verified event values: account `training2666`, reservation `cispahack` (until 2026-11-22 14:00), `gpu:1`, 8 CPUs.
- **Jupyter:** https://jupyter.jsc.fz-juelich.de, logging in with **JSC Account**. This has not been tested.
- **Data constraint:** stay within the approved project. **Do not process personal data.** JSC requires a GDPR assessment plus a signed addendum for any personal or pseudonymised data.

## Reproducibility conventions

- Every run records its config, seed, git commit and dirty flag, and environment in `manifest.json`.
- Use `derive_seed(base, "purpose", ...)` for child seeds (splits, workers, bootstrap) instead of reusing one seed everywhere.
- Record each dataset's SHA-256 (`euroai.provenance.sha256_file`) in NOTES.md when it arrives.
- `uv.lock` pins every dependency, including the optional extras.
- Commit notebooks without outputs.

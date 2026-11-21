# AGENTS.md

Team workspace for the European AI Championship, Paris. Our team name on the leaderboard is **Begg**. The hackathon ends at **2026-11-22 12:00 Europe/Paris**. There are three tasks, each scored on a shared leaderboard server. This file is the only agent instruction file in the repo.

## Map

| Path | Contents |
|---|---|
| `tasks/<task>/README.md` | Full task statement: metric, rules, budgets, data. Read it before touching the task. |
| `tasks/<task>/upstream/` | Organizer starter files, unmodified. Never edit them. |
| `tasks/<task>/research.md` | Papers, code and ranked approaches for the task. |
| `tasks/<task>/results/` | One JSON per scoring run, written by `make score`. |
| `tasks/<task>/submissions/` | One JSON per submission attempt, written by `make submit`. |
| `tasks/model_equality/answers/` | Every endpoint answer collected so far. Never delete or rewrite. |
| `src/euroai/` | Shared, tested utilities. |
| `jsc/` | JURECA access, event setup, job templates. |
| `docs_and_plans/` | Plans (`work/todo`, `work/doing`, `work/done`) and guides. |

Task names: `watermark_localization`, `model_equality`, `unlearning`.

## Commands

```
make setup                                        install dev dependencies with uv
make test        make lint
make data   TASK=<task>                           download task data into data/<task>/ (HF_TOKEN for gated data)
make score  TASK=<task> ANSWER=<file> NOTES=...   print one line, score=<number>, and write a result record
make board  TASK=<task>                           list result records, best first
make submit TASK=<task> FILE=<path>               the only way to submit: validates, enforces cool-down, asks to confirm
uv run --extra api python scripts/me_query.py     the only way to query model-equality endpoints
```

## Hard rules

1. **Irreversible budgets.**
   - Model equality allows 100 answers per scored endpoint (`ep-01`…`ep-33`) for the whole event.
   - Every task allows one submission per 30 minutes, and a failed submission counts.
   - Scored queries and submissions need a human at a terminal. Never call the server directly, from a loop, or from a cloud session.
2. **Reference endpoints `ref-01`…`ref-09` are free.** Develop methods on them.
3. **Secrets.**
   - `HACKATHON_API_KEY` and `HF_TOKEN` come from the process environment first, then from `~/.config/euroai/.env`. That file sits outside every checkout, holds `KEY=VALUE` lines, and has mode 600; `EUROAI_SECRETS_FILE` overrides its path.
   - Never commit, print, paste or log a secret, and never create a `.env` with real values inside the repo.
   - The organizers' upstream scripts ask you to paste the key into the file. Do not; use our scripts.
   - Cloud agents have no secrets. They work from committed answers and public data.
4. **Do not tune to the public leaderboard.** It covers about 30% of the data. Judge methods locally.
5. **JURECA.**
   - Stay inside the team folder `/p/scratch/training2666/<lead>/<team>/`.
   - Submit jobs only through the job template.
   - Never change ACLs or shared `.venv`s, and never rerun the setup scripts.
   - Compute nodes have no internet, so download on login nodes.
6. **Laptop agents reach JURECA only through `ssh jureca-mux ...`, and only after `ssh -O check jureca` succeeds.** If no connection is open, stop and ask a human. Never retry SSH in a loop: JSC firewall-blocks addresses that open too many connections.
7. **No personal data**, anywhere.

## Work contract

- Work starts from an issue that follows `.github/ISSUE_TEMPLATE/task.md`. Assign yourself before starting.
- Use one branch and one PR per issue. Never push to `main`.
- A PR quotes the command and the number before and after, and lists what was tried, including what failed.
- Each task has one owner. The owner merges PRs that touch only `tasks/<task>/`, once the quoted command reproduces. Changes outside task directories go through the lead.
- Run `make score` so the record lands in `results/`, and commit records with the PR. Commit and push a submission record as soon as it is written, so teammates see the cool-down.
- Research outputs go in `tasks/<task>/research.md`.

## Code rules

- Shared code in `src/euroai/` and every scorer get tests. Experiment code needs none.
- Write pure functions with thin scripts, and inject I/O (network, clock, TTY) so tests never touch the network.
- No inline comments.
- Commit messages are plain, with no AI attribution and no Co-Authored-By lines.
- Before committing: `make lint && make test`.

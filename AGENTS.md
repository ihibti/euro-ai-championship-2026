# AGENTS.md

Team workspace for the European AI Championship, Paris. Our team name on the leaderboard is **Begg**. The hackathon ends at **2026-11-22 12:00 Europe/Paris**. There are three tasks, each scored on a shared leaderboard server. This file is the only agent instruction file in the repo.

## Start here

1. Find out who you work for: `gh api user --jq .login` prints your human's GitHub login.
2. If their laptop is not set up, take them through `docs_and_plans/guides/teammate-onboarding.md`, one step at a time: tools, the scoreboard token, their JURECA key and the team folder. Its checks show which steps are left. It runs on their laptop, never in a cloud session.
3. If they own a task, read its playbook and work from it. If the playbook is not there yet, read the task README and pull `main` again in a few minutes. If they own no task, take an open issue for the task they name.

| Task | Owner | Playbook |
|---|---|---|
| `model_equality` | `ramzerk` | `tasks/model_equality/PLAYBOOK.md` |
| `watermark_localization` | `ihibti` (the lead) | none yet; work from the task README and the open issues |
| `unlearning` | `kkrgp` | `tasks/unlearning/PLAYBOOK.md` |

## How the ranking works

Read from the leaderboard page on 2026-11-21:

- A team is ranked only when it has a score on **all three tasks**. Get a first valid submission in on every task early, even a modest one.
- Per task, only the team's best submission counts. The score is rescaled across teams: the best team gets 1, the lowest 0. The overall score is the average of the three rescaled scores.
- A task where every team is weak is therefore worth as much as the others. Beating the current best there counts as a full point for that task.
- The board shows public scores on about 30% of the data. The final ranking uses the other 70%.

## Map

| Path | Contents |
|---|---|
| `tasks/<task>/README.md` | Full task statement: metric, rules, budgets, data. Read it before touching the task. |
| `tasks/<task>/PLAYBOOK.md` | How the owner's agent plays the task: mindset, working loop, first moves. |
| `tasks/<task>/upstream/` | Organizer starter files, unmodified. Never edit them. |
| `tasks/<task>/research.md` | Papers, code and ranked approaches for the task. |
| `tasks/<task>/results/` | One JSON per scoring run, written by `make score`. |
| `tasks/<task>/submissions/` | One JSON per submission attempt, written by `make submit`. |
| `tasks/model_equality/answers/` | Every endpoint answer collected so far. Never delete or rewrite. |
| `src/euroai/` | Shared, tested utilities. |
| `jsc/` | JURECA access, event setup, job templates. |
| `.github/ISSUE_TEMPLATE/` | `task.md` for a work brief, `gpu-request.md` to ask the lead for a GPU run. |
| `docs_and_plans/` | Plans (`work/todo`, `work/doing`, `work/done`) and guides. |
| `docs_and_plans/guides/teammate-onboarding.md` | Step-by-step setup of a teammate's laptop, token and JURECA access. |

Task names: `watermark_localization`, `model_equality`, `unlearning`.

## Commands

```
make setup                                        install dev dependencies with uv
make test        make lint
make data   TASK=<task>                           download task data into data/<task>/ (HF_TOKEN for gated data)
make score  TASK=<task> ANSWER=<file> NOTES=...   print one line, score=<number>, and write a result record
make board  TASK=<task>                           list result records, best first
make check  TASK=<task> FILE=<path>               every submission check, without submitting; agents may run it
make submit TASK=<task> FILE=<path>               the only way to submit: validates, enforces cool-down, asks to confirm
uv run --extra api python scripts/me_query.py     the only way to query model-equality endpoints
uv run --extra api python scripts/me_query.py --budget   answers left on every endpoint; asks nothing
```

`make score` covers model_equality (MCC on `ref-01`…`ref-09`) and watermark_localization (mIoU on the train split; ANSWER is a `.npz` or `.pt` with `image_names` and `masks`). Unlearning has no exact local score.

## Hard rules

1. **Irreversible budgets.**
   - Model equality allows 100 answers per scored endpoint (`ep-01`…`ep-33`) for the whole event.
   - Treat every task as one submission per 30 minutes, and a failed submission counts, until the organizers say otherwise. Model equality states this limit officially.
   - Scored queries and submissions need a human at a terminal. Never call the server directly, from a loop, or from a cloud session.
2. **Reference endpoints `ref-01`…`ref-09` are free.** Develop methods on them.
3. **Secrets.**
   - `HACKATHON_API_KEY` and `HF_TOKEN` come from the process environment first, then from `~/.config/euroai/.env`. That file sits outside every checkout, holds `KEY=VALUE` lines, and has mode 600; `EUROAI_SECRETS_FILE` overrides its path.
   - Never commit, print, paste or log a secret, and never create a `.env` with real values inside the repo.
   - The organizers' upstream scripts ask you to paste the key into the file. Do not; use our scripts.
   - Cloud agents have no secrets. They work from committed answers and public data.
   - The organizers disqualify a team whose scoreboard token is shared outside the team or committed to a public repository, and they revoke the token. It travels only by private message from the lead, and is typed only into the secrets file: never into a chat with an agent, a cloud session, an issue or a file in the repo.
   - Run `git config core.hooksPath .githooks` once per clone. The pre-commit hook and `make lint` then refuse content that holds a secret from your machine, a Hugging Face token or a private key.
4. **Do not tune to the public leaderboard.** It covers about 30% of the data. Judge methods locally.
5. **JURECA.**
   - Stay inside the team folder `/p/scratch/training2666/hibti1/Begg/`. Each task's README has an On JURECA section.
   - **One job at a time for the whole team.** The organizers' rule, copied word for word on 2026-11-21 at 19:50:
     > The cispahack reservation is full (30/30 nodes) and several teams are holding multiple nodes, so other teams can't get in. Nodes here are exclusive: every job gets a whole node with 4 GPUs. Teams are allowed to use only node at a time. So two 1-GPU jobs = two nodes = a violation.
     >
     > The rule: your whole team (everyone in /p/scratch/training2666/<lead>/<team>/) may run at most one job at a time, counting running and queued jobs.
     >
     > Check your team before submitting (not squeue -u $USER — it can't see teammates):
     > ```
     > T=$(pwd | grep -oE '/p/scratch/training2666/[^/]+/[^/]+')
     > squeue -r --reservation=cispahack -h -o "%u|%T|%Z" | grep "$T"
     > ```
     > If anything shows up, your team already has its node — don't submit another.
     >
     > Need more than 1 GPU? Use one job with #SBATCH --gres=gpu:4 and parallelise inside it — never two jobs.
     > More work to do? Chain it: sbatch --dependency=afterany:<jobid> next.sh. Don't stack a queue. Arrays: --array=0-N%1.
     > Always submit from inside your team folder.
     >
     > If your team is on >1 node right now, please scancel down to one job immediately.
   - **Only the lead's account holds that job.** Everyone else, human or agent, never runs `sbatch`, `srun`, `salloc` or `scancel`. To get a GPU run, push your code and open an issue with `.github/ISSUE_TEMPLATE/gpu-request.md`. The lead's agent runs it on a free GPU of the team's node and answers on the issue.
   - The lead's agent submits only through the job template and from inside the team folder. It keeps one job, with at most one more chained behind it with `--dependency=afterany`. The check above prints usernames: never paste its output into an issue, a commit or a file.
   - Never change ACLs or shared `.venv`s, and never rerun the setup scripts.
   - Compute nodes have no internet, so download on login nodes.
6. **Laptop agents reach JURECA only through `ssh jureca-mux ...`, and only after `ssh -O check jureca` succeeds.** If no connection is open, stop and ask a human. Never retry SSH in a loop: JSC firewall-blocks addresses that open too many connections.
7. **No personal data**, anywhere. That includes JSC usernames; the lead's appears only because the team folder path contains it.

## Work contract

- Work starts from an issue that follows `.github/ISSUE_TEMPLATE/task.md`. Assign yourself before starting.
- A GPU run is asked for with `.github/ISSUE_TEMPLATE/gpu-request.md` (hard rule 5). Push the code first: the lead's agent runs the commit you name, and nothing else.
- Use one branch and one PR per issue. Never push to `main`.
- A PR quotes the command and the number before and after, and lists what was tried, including what failed.
- Each task has one owner (table under Start here). The owner merges PRs that touch only `tasks/<task>/`, once the quoted command reproduces. Changes outside task directories go through the lead.
- Run `make score` so the record lands in `results/`, and commit records with the PR. Commit and push a submission record as soon as it is written, so teammates see the cool-down.
- Research outputs go in `tasks/<task>/research.md`.

## Code rules

- Shared code in `src/euroai/` and every scorer get tests. Experiment code needs none.
- Write pure functions with thin scripts, and inject I/O (network, clock, TTY) so tests never touch the network.
- No inline comments.
- Commit messages are plain, with no AI attribution and no Co-Authored-By lines.
- Before committing: `make lint && make test`.

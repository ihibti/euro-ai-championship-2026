# Swarm workspace for the 24-hour championship: design

Status: awaiting approval. Written 2026-11-21 about 15:45. Hackathon ends 2026-11-22 12:00 Europe/Paris.

## Intent

Three people, three tasks, about twenty hours. Agents from three vendors, running under three people's accounts, must be able to start cold on any task, measure their own progress and hand work back through GitHub. Colleagues' agents work without the lead's local tooling, so everything an agent needs lives in the repo and in the issue text. The setup has to be usable within the hour; anything slower is paid for in task time.

## Goals, non-goals, anti-goals

Goals
- An agent can start from a clone plus one issue, with no other briefing.
- Each task has one command that prints the number to climb.
- Budgets that cannot be recovered are spent only by a person.
- Expiring allowances are used before they expire.

Non-goals
- An orchestration layer, GitHub CI, a custom interface.
- Automatic issue triggers. One may be added later if a ten-minute test passes.

Anti-goals
- Spending the 100-answer endpoint budget or a submission slot by accident.
- The team API key or the Hugging Face token in git, or handed to cloud agents.
- An agent touching another team's files on JURECA.

## Facts the design rests on

Sources: organizer slides (2026-11-21), the public `SprintML/hackathon` dataset on Hugging Face and the starter files of each task.

| | Watermark localization | Model equality | Unlearning |
|---|---|---|---|
| Goal | Label each 4x4 cell of a 512x512 image as watermarked or clean (128x128 grid) | For 33 endpoints, give family and model labels | Return a ViT-B/16 state dict that behaves like a model retrained without 3,500 forget images |
| Score | Mean over images of (IoU clean + IoU watermarked) / 2 | 0.1 x MCC family + 0.9 x MCC model, negative MCC counts as 0 | forget score x retain score, zero under 70 % held-out accuracy |
| Local number | Exact, on the 510 labelled training images | Exact on 9 reference endpoints with published answers and no query limit | Proxy only: no labels, no retain set, no gold model |
| Data | `train.zip`, `test.zip`, starter kit; gated on Hugging Face, manual approval | Starter scripts only, public | `model.pt` 344 MB, `forget_images.npz` 527 MB, public |
| GPU | Yes | No | Yes |
| Irreversible budget | Submission limit unknown | 100 answers per scored endpoint for the whole event, 4 concurrent requests per team, one submission per 30 minutes | Submission limit unknown |

- Submissions and queries go to one HTTP server with a team API key in the `X-API-Key` header.
- Model equality counts quantization, LoRA adapter and base model as different models. Watermark, system prompt and decoding strategy do not change the model.
- JURECA: reservation `cispahack`, partition `dc-gpu`, account `training2666`, one GPU per job. The team lead runs `hackathon_setup.sh` once, teammates run `teammate.sh`. The shared folder is `/p/scratch/training2666/<lead>/<team>/`.
- Capacity: lead has Claude Max 5x (39 % of the week left, 29 % for Fable, two full resets), Devin Max (allowance resets about 09:00 Sunday) and Antigravity with Gemini Flash and older Claude models only. Two colleagues share one Claude Max and one Devin Max. No overage anywhere.

## Approaches considered

- **A. Repo as harness, issues as queue, agents started by hand.** Chosen. It is the only option that works for colleagues' agents, it is the fastest to build, and nothing in it is vendor-specific.
- **B. A plus label triggers.** Needs API keys stored as repo secrets, reaches only the seats whose keys are stored, and is partly undocumented on these plans. Deferred.
- **C. A coordinator that fans work out.** Covers one machine only and costs more tokens. Orca's existing fan-out may be used ad hoc; nothing is built around it.

The lead delegated the choice ("whatever is easiest").

## Design

### 1. Repo contract

```
AGENTS.md                 the only instruction file, read by Claude Code, Devin and Antigravity
tasks/<task>/README.md    full task statement, metric, rules, data, known limits
tasks/<task>/upstream/    organizer starter files, unmodified
tasks/<task>/results/     one JSON file per run
tasks/<task>/submissions/ one JSON file per submission
src/euroai/               shared utilities, tested
```

Commands, identical on a laptop, in a cloud session and on a JURECA login node:

- `make data TASK=<task>` downloads the task data into `data/<task>/`, which is gitignored.
- `make score TASK=<task>` prints one line, `score=<number>`, and writes a result file.
- `make board TASK=<task>` prints the results sorted by score.
- `make submit TASK=<task> FILE=<path>` is the only path to the server.

Results and submissions are one file per record, so concurrent PRs never conflict on a shared log.

### 2. Guardrails on budgets that cannot be recovered

- **Scored endpoints.** A single query module wraps the organizers' client. Reference endpoints are free. Scored endpoints need an explicit spend flag and a typed confirmation at a terminal, and the planned spend and remaining budget are printed first. Every answer is appended to a committed file, so analysis never needs a second query.
- **Submissions.** `make submit` asks for a typed confirmation, refuses during a cooldown, and records file hash, commit, person and returned score.
- **Secrets.** `HACKATHON_API_KEY` and `HF_TOKEN` live in `.env` on people's machines and on JURECA. Cloud agents do not get them; they work from committed answers and public data.
- **Cluster.** Agents stay inside the team folder, submit jobs only through the job template and never modify ACLs or shared environments.

These guards stop mistakes, not a determined process. `AGENTS.md` states them as rules as well.

### 3. Issue and PR contract

An issue is a cold-start brief with fixed fields: task, goal, files to read first, paths the agent may change, the command and number that define done, what is out of scope, and a time budget. Labels: one per task, plus `infra`, `needs:gpu` and `needs:api-key`. An agent claims an issue by assignment before starting.

A PR names its issue, quotes the scoring command with the number before and after, and lists what was tried, including what failed.

### 4. Ownership and merging

One owner per task. The owner, or an agent acting on their instruction, merges any PR that touches only `tasks/<task>/` once the quoted command has been run. Changes outside task directories go through the lead. There is no CI and no required review.

Tests are required for shared code and for each scorer, because every agent climbs whatever the scorer reports. Experiment code needs none.

### 5. How each seat is started

| Seat | Start | Use it for |
|---|---|---|
| Claude on the lead's laptop | Orca worktree per issue, Opus 5.5 for workers, Fable for the lead session | Anything that touches JURECA or the API key; scorers; merging |
| Claude cloud | `claude --cloud` with the issue text | Self-contained code and analysis on public data |
| Devin cloud | `devin --cloud -p` with a prompt file; `.devin/blueprint.yaml` installs uv and syncs | Same, and the first choice until its allowance resets |
| Antigravity | `agy -p --mode auto`, local only | Mechanical chores; optional |
| Colleagues' agents | Their own tools on a clone | Issues for the task they own, returned as PRs |

`scripts/issue_prompt.py <number>` turns an issue into a prompt for any of these. Its prompt builder is a pure function with a test.

### 6. Budget schedule

- Devin: saturate until the reset near 09:00 Sunday, since the unused part expires; the fresh allowance covers the last three hours.
- Claude: when a limit blocks work and the next five-hour refresh is more than an hour away, spend a reset. Otherwise move that hour's work to Devin.
- Leaderboard: submissions are decided by the task owner.

### 7. JURECA

The lead runs `hackathon_setup.sh` with `HF_TOKEN` exported, because the watermark dataset is gated and the script skips a task whose download fails. The repo is cloned into the team folder. The job template takes the organizers' values. Laptop agents reach the cluster through the open SSH connection.

## Calls made by the builder, open to correction

- One owner per task, and owners merge inside their task directory.
- Cloud agents receive no secrets.
- Local worktrees are Orca-managed under `~/orca/workspaces/`, not under `~/worktrees/`.
- Organizer starter files are vendored unmodified; the organizer slide deck is not committed.
- No AI attribution in commits, stated in `AGENTS.md` so it holds on every machine.
- Antigravity is treated as optional.

## Not settled by this design

- The watermark submission format and its limits, unknown until the gated starter kit is readable.
- The unlearning proxy score, which is the owner's first piece of work.
- Whether colleagues' Devin cloud sessions can see a private repo they do not own. Tested once they are invited; agents on their own machines work regardless.

## Human actions on the critical path

1. Request access to `SprintML/Hackathon_Watermark_Localisation_Task` on Hugging Face and create a read token. Approval is manual.
2. Put the team API key into `.env`.
3. Decide the JURECA team lead, collect the three Jülich usernames and pick the team folder name.
4. Invite the colleagues to the GitHub repo.
5. Connect GitHub in Devin's settings; run `/web-setup` once in Claude Code for cloud sessions.
6. Name the owner of each task.

## Build order

1. `AGENTS.md`, task READMEs with vendored starter files, issue and PR templates, labels, first issues. About 20 minutes; colleagues can start after this.
2. `make data`, `make score`, `make board`, `make submit`, the query guard, with tests. About 40 minutes.
3. Devin blueprint, issue prompt script, job template update.

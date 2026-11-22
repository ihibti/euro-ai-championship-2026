# The swarm doctrine against this event

Written 2026-11-21 at about 20:30 Europe/Paris, for the lead and every agent. The team works by Georgios Konstantopoulos, "Effective Swarm Riding for Engineers" (Paradigm, 2026-11-20): frontier models, cloud agents, short prompts, no plans, no orchestration, no docs, no agent-written unit tests, simulation tests instead, push to master. This page checks each premise against the event and says what to keep and what to change.

## Where the premises fail

| The post assumes | Here | So |
|---|---|---|
| A cheap, faithful local check: benchmarks, simulation tests | Watermark: exact, but the honest part is 111 held-out images (standard error 0.017), and result records carry the train score, which rewards memorisation (c5: 0.91 train, 0.73 held-out). Model equality: exact only on one small model, one family, 36 pairs. Unlearning: none; the server is the check, at most twice an hour, with a human typing. | A swarm is worth what its checks can resolve: a lot on watermark, little on model equality, almost nothing on unlearning. |
| Writing code is the bottleneck | The bottlenecks are serial: one node with 4 GPUs for the whole team (the organizers' rule, 19:50), one SSH connection on the lead's laptop, a human for every submission and scored query, the lead session's quota. | More agents queue on the same serial steps. |
| Mistakes are cheap to undo | Scored answers, submission slots and every look at a public score cannot be undone. | The gates cover the first two. Nothing covers the third: each public score acted on spends some of the hidden split's independence. |
| The repo is the memory | What matters lives on JURECA (`output/`), in running jobs, in sessions and on branches. An organizer's answer on outside data sat only on a branch; a watermark literature file sat uncommitted in a local worktree; `NOTES.md` and `baseline-options.md` still describe the hours before the tasks were known. | An agent starting cold acts on stale or missing facts. |
| More agents give more ideas | Same models, same `AGENTS.md`, the same ranked list in each `research.md`. | Proposals are correlated unless each agent gets a different hypothesis. |
| The goal is a number to climb | Each task is rescaled from the worst team to the best and the three are averaged; a team without a score on every task is not ranked. | A point is worth the most on the task where teams are closest together. Nothing in the repo records the other teams' scores. |
| A short prompt in a good codebase is enough | Each task is built around a trap aimed at the generic approach. Every real gain so far came from a fact about how the data were made. | Breadth walks into the trap; one careful measurement finds the fact. |

## What follows if nothing changes

- Candidates outrun checks. GPU runs queue inside one node, and the submission rate, two per hour per task while someone is awake, binds before compute does.
- The lead and the lead session are the coordinator, the GPU gateway, the submitter and the merger. When quota or sleep runs out, coordination stops just as the morning window opens.
- Overnight runs without a local check optimise the proxy they were given. In unlearning that proxy is circular by its own account (`tasks/unlearning/research.md`, section 5).

## The case against swarming here

- The gains came from a few facts found by one thread with all the context: how the watermark dataset was built, the nine whole classes, the last-layer imprint of the forget set.
- Model equality and unlearning have fixed data. More agents cannot buy information, only run more analyses on the same data, and the analysis that looks best on 36 pairs is the most overfit one.
- Coordination grows with the square of the number of agents; the lead's attention does not grow at all.
- Copies of one model share blind spots, and the post drops cross-review, the cheap way to catch them.
- If the final ranking takes the hidden score of each team's best public submission, risky candidates that score high by luck are a liability.
- "Delete the tests" fails at the scorer, the reward that every agent climbs. Keep our rule: tests for scorers and gates only.

## Failure modes to watch

- A task without any accepted submission at the end: the team is not ranked.
- Which submission counts for the final ranking is not stated: the hidden score of the best public one, the best hidden score over all of them, or the last one. Each calls for a different strategy; ask the organizers.
- A watermark candidate trained on all real images (`--all-real`) looks better on the train score and leaves no honest estimate.
- Choices per kind of edit rest on 4 to 21 held-out images per method.
- One SSH connection: a sleeping laptop or a new hotspot address stops all GPU work and the unlearning submissions until a human types the one-time code.
- Cloud agents cannot help on watermark: its data are gated and cloud sessions get no token. They can work on unlearning, whose files are public and whose last-layer edits need only a CPU, and on model-equality answers once these are committed.

## What to keep

- Gates on everything that cannot be undone; tests on scorers and gates; one owner per task; one record file per run.
- Briefs shaped like `docs_and_plans/work/doing/watermark-rectangle-search.md`: one hypothesis, a budget, a stop rule, a table to bring back.

## What to change

1. Climb honest numbers only. On watermark, compare candidates on the held-out fifth at the test mix (`tasks/watermark_localization/points_map.py`), and prefer one only when its paired held-out gain has an interval above 0.
2. Treat every irreversible action as an experiment. Before each submission or scored spend, write down what it should show and what follows each result, as the unlearning plan does for its first submissions.
3. Give each agent a different hypothesis and a condition that kills it, never "improve the score".
4. Record the leaderboard. Once an hour a human copies the best, the worst and our public score per task into the repo; it shows where a point is cheapest.
5. Write the night's decision rules before anyone sleeps: which candidate goes in after which result, and the last submission per task by 11:00.
6. Put every fact an agent may need on `main` at once: rulings, measurements, dead ends. Correct or delete stale pages.

## Facts this page rests on

- Watermark held-out numbers and the dead ends: `tasks/watermark_localization/analysis.md`.
- The unlearning model's training loss: `tasks/unlearning/research.md`, last section.
- The organizers' GPU rule: `AGENTS.md`, hard rule 5. The ranking rules: `AGENTS.md`, How the ranking works.

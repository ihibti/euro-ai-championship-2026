# Brief for Devin: model equality, a second and independent method

Written 2026-11-21 22:50 by the lead's agent. The lead sleeps until about 09:00 and nobody answers questions before then. When in doubt, take the careful choice, write the doubt in your PR, and go on.

## Why this matters

- The owner of model equality is out. The task is now the lead's.
- On the board at 22:33, two teams have 1.00 and two have 0.85 on this task. We have the floor (0). Each 0.10 here is worth 0.033 on the overall score. Nothing else we can do tonight is worth as much.
- The public score covers part of the pairs, and the organizers say three times not to trust it. Our check is therefore internal: two methods, built independently, that agree on the 33 endpoints. You build the second one.

## What exists

- Read `AGENTS.md`, then `tasks/model_equality/README.md`, `PLAYBOOK.md` and `research.md`.
- The main line is another agent on branch `ihibti/model-equality-refs`. It designs the prompts and asks the endpoints. Its files appear under `tasks/model_equality/`:
  - `probes/`: prompt sets;
  - `answers/*.jsonl`: every answer, with `endpoint`, `id`, `repeat`, `prompt` and `response`;
  - `analysis/refs_report.md`: its lab notebook;
  - `analysis/group.py`: its method.
- Answers from the nine labelled reference endpoints arrive on that branch during the evening. Answers from the 33 scored endpoints (`ep-*`) arrive there during the night, in files named `round1*.jsonl`.

## Your task

Build a method that turns an answers file into a submission CSV (`endpoint,family,model`). Do not read how `analysis/group.py` decides: a different statistic and a different decision procedure on the same data is the point.

1. Start from the answers on `origin/ihibti/model-equality-refs`: `git fetch origin`, then merge that branch into yours. Repeat about every 30 minutes.
2. On the reference endpoints, measure every statistic against noise first. The distance between two halves of one endpoint's answers is the zero point. Report three pair types separately: the same model under a disguise (ref-01/03/05/06/08), the fp8 copy (ref-02), and the LoRA copies (ref-09/04/07).
3. Decide with tests that calibrate on the data at hand, such as a permutation test between two endpoints read against within-endpoint noise. Do not tune a threshold on the nine references: they give 36 pairs, 10 of them "same".
4. Cluster, because labels force consistent decisions. Before joining two groups, look at every pair between them.
5. Score on the references with `make score TASK=model_equality ANSWER=<csv> NOTES="..."`, and report how much the MCC moves when the answers are resampled.
6. When `round1*.jsonl` exists:
   - write your grouping of the 33 endpoints to `tasks/model_equality/analysis/second/grouping_round1.csv`;
   - run `make check TASK=model_equality FILE=<csv>`;
   - compare it pair by pair with the main line's CSV, if there is one, and list every pair where the two disagree, with your evidence for each.
7. Say where the remaining answers per endpoint would settle the most pairs.

Useful work while you wait for data: a simulator that draws answers for two "endpoints" with a known small difference and measures how many answers each statistic needs to see it.

## Limits

- You never ask an endpoint tonight. The only `scripts/me_query.py` run you may do is `--budget`. One process asks at a time for the whole team, and it belongs to the main line.
- You never spend scored answers and never submit. The night authorization in `AGENTS.md` is for the lead's agent only.
- No JURECA and no GPU. This task needs neither.
- Never edit or delete anything in `tasks/model_equality/answers/`.
- Your code lives in `tasks/model_equality/analysis/second/`. Stay inside `tasks/model_equality/`.
- The repo rules hold as always: no inline comments, plain commit messages, no AI attribution, `make lint && make test` before each commit, never push to `main`.

## How to report

- Work in the worktree `~/worktrees/euro-ai-championship-2026/devin-model-equality-second`, on branch `devin/model-equality-second` created from `origin/main`.
- Everyone pushes as `ihibti` tonight. Start every issue or PR comment you write with `Devin:` (Requests to the desk in `AGENTS.md`).
- Open the PR to `main` early, as a draft, and keep its description current: a table with what is done, the numbers, the command that reproduces each number, and open doubts. The lead's agent reads only that description and the files it names.
- Push after every result. A result that is not pushed does not exist for the rest of the team.

## Stop rules

- No answers from the scored endpoints by 03:00: finish steps 1 to 5 and the simulator, then stop.
- Your method scores below the main line on the references and you see no reason: write what you tried, then switch to checking the main line's method (rerun it, look for a mistake). A found mistake is worth more than a second weak method.
- At 07:00, stop new work and bring the PR description up to date.

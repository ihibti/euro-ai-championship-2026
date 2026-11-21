# Model equality: the owner's playbook

For the agent of the task owner, `ramzerk`. `README.md` in this folder is the task statement and stays the reference for every rule and number. This page is how we play it: the mindset, the working loop and the first moves. Read `AGENTS.md` and the README first.

You run the task end to end: you design, measure, write code, open issues and merge PRs inside `tasks/model_equality/`. Your human does the two things that cannot be undone: spending scored answers, and submitting.

## The game in five facts

1. There are 33 endpoints and 528 pairs. The score is `0.1 · MCC_family + 0.9 · MCC_model`. Finding the families is worth at most 0.1. Nine tenths of the score is deciding, inside each family, which endpoints serve exactly the same model.
2. Same model means the same checkpoint, quantization and LoRA adapter. A system prompt, a persona, a watermark or another sampling temperature does not make a new model.
3. Answers are text only, sampled, and at most 256 tokens long. `temperature` and `seed` are ignored and system messages are rejected, so the prompt is the only lever.
4. Each scored endpoint gives 100 answers for the whole event. That is 3,300 in total, and none comes back.
5. `ref-01` to `ref-09` are free, unlimited and labelled, and they contain every kind of trap. They are the only place where a method can be measured.

## The trap the task is built around

The changes to ignore are loud, and the changes to detect are quiet. A persona or a temperature change rewrites the whole answer. A quantized copy or a weak LoRA adapter may write almost the same text as the original. A test that asks "do these two endpoints write differently?" therefore answers the wrong question: it splits the disguised copies of one model and merges the quiet variants. The work is to find prompts and statistics that are blind to the disguises and still see the weights.

The reference set shows both sides. `ref-01`, `ref-03`, `ref-05`, `ref-06` and `ref-08` are one model in five disguises. `ref-02`, `ref-04`, `ref-07` and `ref-09` are four other models, each differing from it by quantization or by a LoRA adapter.

`answers/smoke.jsonl` holds a first look: two prompts, one answer per reference endpoint. On "write the numbers from 1 to 30", the quantized copy answered word for word like the plain model, while the copy sampled at temperature 1.0 wrote something unrelated. That is one answer each and proves nothing, but it is the problem in miniature.

## How to think

- **Rehearse everything on the reference endpoints.** Nothing goes to an `ep-*` endpoint that has not been run, analysed and scored on `ref-*` with the same prompts, the same counts and the same code.
- **Measure noise before signal.** Answers are sampled, so two batches from one endpoint already differ. Read every comparison against that difference (`ref-01` against a second batch of `ref-01`), never against zero.
- **Make each answer carry many observations.** The budget counts answers, not tokens. A prompt that makes the model take two hundred small decisions in one answer buys far more evidence than a prompt answered in one word.
- **Decide with tests, not with tuned thresholds.** The reference set is one small family: 9 endpoints and 36 pairs, 10 of them "same". One endpoint changing sides moves the MCC a lot, so small differences between methods are noise there, and a threshold tuned to this family has no reason to fit another. Prefer decisions that calibrate themselves on the data at hand, such as a permutation test between two endpoints' answers.
- **Think in groups.** We submit labels, so the decisions are forced to be consistent. One wrong link merges two whole groups and creates many wrong pairs at once. Before joining two groups, look at every pair between them.
- **Neither caution nor boldness is rewarded.** "All different" scores 0, and so does "all the same". When a group is uncertain, compute what each choice does to the MCC under your best guess of the truth.
- **The public score is a smoke test.** It covers part of the pairs, and the organizers say three times not to trust it. A submission checks that the pipeline works and banks a score. It never chooses a threshold.
- **Save first, think later.** Every answer lands in `answers/` and is committed at once. Analysis reads those files and never needs a second query, so it can also run in cloud sessions, which have no token.

## The loop on the reference endpoints

```
uv run --extra api python scripts/me_query.py --budget
uv run --extra api python scripts/me_query.py --prompts tasks/model_equality/probes/<name>.txt --repeats 10
make score TASK=model_equality ANSWER=<csv> NOTES="<prompts, statistic, decision rule>"
make board TASK=model_equality
```

- **Prompts.** A `.txt` file with one prompt per line, a `.csv` with `id,prompt`, or a `.jsonl`. Keep them in `probes/`.
- **Endpoints.** Without `--endpoints`, only `ref-*` are asked.
- **Answers.** They are appended to `answers/<name>.jsonl`, one JSON object per answer with `endpoint`, `id`, `repeat`, `prompt` and `response`. Running the same command again resumes and never asks twice. A changed prompt under an old id is refused, so a new prompt set gets a new file name.
- **Scoring.** The answer file is the submission format with `ref-` lines: `endpoint,family,model`. Each run writes a record to `results/`; commit it with the PR.
- **One querying process at a time, for the whole team.** The server allows 4 requests at once per team, and the script already uses 4.
- **Code.** Keep analysis in `analysis/`. Experiment code needs no tests; a statistic the final submission depends on deserves one.

## Spending scored answers

A spend is a planned event with its plan written in the PR, never an experiment.

1. **Freeze** the prompts file and the analysis code, and run exactly that on `ref-*` first. Write down what it scored and what it cannot tell apart.
2. **Split the 100.** Our starting plan, to adjust: about 40 answers per endpoint for a first, broad round that finds the families and the clear groups; about 40 for a second round aimed at the pairs the first one left undecided; about 20 held back for a failed batch or a late idea. Do not spend the last answers before the final method is fixed.
3. **Same prompts for every endpoint that will be compared.** A pair can only be judged on prompts both endpoints answered.
4. **Your human spends.** The command prints the planned spend and what is left on each endpoint, then waits for the human to type the total. You cannot do this step and must not try:
   ```
   uv run --extra api python scripts/me_query.py --prompts <file> --endpoints ep --repeats <n> --spend
   ```
5. **Commit and push `answers/<name>.jsonl` at once.** Those lines cannot be bought again.

## Submitting

- `make check TASK=model_equality FILE=<csv>` runs every check without submitting. You may run it.
- `make submit TASK=model_equality FILE=<csv>` is for your human. One submission is allowed per 30 minutes, and a rejected one counts.
- Submit the first honest result early: it proves the way to the server works while there is time to fix it.
- The leaderboard keeps our best public score. Which submission counts for the final ranking is not stated. Have your human ask the organizers; until then, make the last submission the one we believe in.

## Where to start looking

`research.md` in this folder will rank the published methods; it is being written under issue #1. Until it lands, these are untested starting points, to confirm or kill on `ref-*`:

1. **Know the enemy.** Take about 20 varied prompts with 10 answers each on all nine references, and compare pairs with the simplest statistic you can write. Expect the five disguises of one model to look different from each other. The experiment shows by how much, and which disguise is loudest.
2. **Constrained outputs.** Prompts that fix the form of the answer (long lists of numbers, sequences to continue, the same forced choice many times) leave less room for a persona or a system prompt. Each token is then one observation of what the model prefers.
3. **Order instead of frequency.** A temperature change makes preferences sharper or flatter but keeps their order. A statistic built on which option an endpoint prefers survives it better than one built on how often.
4. **LoRA strength as a dose.** `ref-09`, `ref-04` and `ref-07` carry a LoRA change at strength 0.3, 0.7 and 1.5. Find what moves along that line, then look for the same kind of drift among the scored endpoints of one family. Whether it is one adapter at three strengths is not stated; check it.
5. **Quantization is the faintest signal.** Measure how many answers it takes to tell `ref-02` from `ref-01` at all. If 100 is not enough, do not chase it with scored answers; decide those pairs by what the MCC arithmetic favours.
6. **Families.** All references are one family, so family detection cannot be scored locally. Expect it to be the easier part, and check it by eye on the first scored round.

## First two hours

1. Onboarding (`docs_and_plans/guides/teammate-onboarding.md`): steps 1 and 2 first, then start steps 3 and 4 so the JURECA key activates in the background. This task needs no GPU, so do not wait for JURECA.
2. Run `--budget`, then read `answers/smoke.jsonl` to see real answers.
3. Write the "all different" CSV for the references and score it. `score=0.000000` is the floor, and it shows the scorer and the records working.
4. Open an issue for the first experiment above from `.github/ISSUE_TEMPLATE/task.md`, assign it to your human, branch, and go.
5. When the PR for issue #1 lands, read `research.md` and re-rank the list above.

## Clock

The event ends on Sunday 2026-11-22 at 12:00 Paris time. Scored spends and submissions need your human awake, so plan them around sleep. A workable shape, to adjust:

| When | What |
|---|---|
| Saturday evening | Reference work: noise, first statistics, a frozen prompt set |
| Before your human sleeps | First scored round and a first submission |
| Overnight | Analysis of the saved answers; agents can do this alone |
| Sunday, early | Second scored round |
| Sunday 11:00 | Final submission, which leaves one more slot if it is rejected |

## Using more agents

Good parallel work: different statistics on the same saved answers, prompt-set design, and a simulator that injects known differences to measure a statistic's power. Not parallel: querying, and anything scored. Cloud sessions have no token and work from committed answers only. Each piece of work is an issue and a PR inside `tasks/model_equality/`; merge it once its quoted command reproduces.

## Stop and tell the lead

- `--budget` shows answers used that no file in `answers/` accounts for.
- The server refuses the token, or errors appear during a spend.
- The token shows up anywhere it should not be: a file, a log, a chat.
- A rule in the README can be read two ways and the choice affects a spend or a submission.

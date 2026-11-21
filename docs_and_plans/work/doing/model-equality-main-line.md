# Brief: model equality, main line on the reference endpoints

The desk gives this brief to one agent, the only one that asks the endpoints. It was first run on 2026-11-21 from 22:40 to about 23:30. That run's notes and next steps are in `tasks/model_equality/analysis/refs_report.md` on branch `ihibti/model-equality-refs`, section "Status at handoff". Read them first and continue from them, not from zero.

Worktree `~/worktrees/euro-ai-championship-2026/model-equality-refs`, branch `ihibti/model-equality-refs`, dev tools installed. Run every command from that directory. Read, in order: `AGENTS.md`, `tasks/model_equality/README.md`, `PLAYBOOK.md`, `research.md`. The tools are `scripts/me_query.py`, `src/euroai/me_query.py` and `src/euroai/me_score.py`.

## Situation

- 33 scored endpoints `ep-01`…`ep-33`, 100 answers each for the whole event. Spending them and submitting are the desk's actions, inside the lead's night authorization; you never do either.
- Nine reference endpoints `ref-01`…`ref-09` are free, unlimited and labelled (`tasks/model_equality/upstream/practice_answers.csv`). ref-01/03/05/06/08 are one model in five disguises: plain, system prompt, persona, watermark, temperature 1.0. ref-02 is its fp8 copy. ref-09/04/07 carry a LoRA at strength 0.3, 0.7 and 1.5.
- Score = 0.1 · MCC_family + 0.9 · MCC_model. On the public board two teams have 1.00 and two have 0.85, so a near-perfect grouping is possible within 100 answers per endpoint.
- The event ends on 2026-11-22 at 12:00 Paris time.

## Goal (reference endpoints only)

Find a prompt set and a decision procedure that group endpoints by model identity (same checkpoint, quantization and LoRA adapter) while ignoring the disguises. Package both so they run unchanged on the scored endpoints.

## Hard limits

- Query only with `uv run --extra api python scripts/me_query.py --prompts <file> --repeats <n>`. Without `--endpoints` it asks only `ref-*`. Never pass `--spend`, never name an `ep-*` endpoint, never call the server another way.
- One querying process at a time for the whole team. Check `pgrep -fl me_query.py` before you start one. The server allows 4 requests at once per team, and the script already uses 4.
- Keep the reference answers under about 15,000 in total.
- Never read, print or copy the secrets file or any key.
- Answers go to `tasks/model_equality/answers/<name>.jsonl`. Never delete or rewrite those files. A new prompt set gets a new name. Commit them.
- Stay inside `tasks/model_equality/`.
- Commits: plain messages, no AI attribution. No inline comments in code. Run `make lint && make test` before each commit. Push to `origin ihibti/model-equality-refs`. No pull request, never `main`.
- If the server refuses the key or errors persist, stop and report. Do not retry in a loop.

## Deliverables, in this order. Write as you go, so that a cut-off leaves usable state.

1. `tasks/model_equality/analysis/refs_report.md`: the lab notebook. One short block per experiment: hypothesis, command, table, conclusion.
2. Noise and decoding facts: whether an endpoint is deterministic, answers per minute, typical answer length.
3. The separation experiment. For each probe type, the distance between endpoints against the within-endpoint split-half noise, for three kinds of pair: the same model under a disguise, the fp8 copy, and the LoRA copies (against the plain model and against each other).
4. A frozen first-round prompt file `tasks/model_equality/probes/round1.csv` (`id,prompt`) costing at most 40 answers per endpoint (prompts times repeats). With it, `tasks/model_equality/analysis/group.py`, which reads an answers file and writes `endpoint,family,model` for `ref-*` and `ep-*` alike. Report its reference MCC at exactly those counts (`make score TASK=model_equality ANSWER=<csv> NOTES="..."`, then commit the record) and how far the MCC moves when the answers are resampled. Say plainly what it cannot separate.
5. A recommendation for the remaining 60 answers per endpoint.

## Ideas to test, not conclusions

- The budget counts answers, not tokens: up to 256 generated tokens per answer and 1,024 input tokens per prompt. Make each answer carry many observations.
- The disguises are loud and the weight changes are quiet. Text similarity answers the wrong question. A temperature change keeps the order of preferences, so compare which option is preferred, or rank correlations of per-item rates, rather than raw frequencies.
- A long prompt with a rigid output format leaves a system prompt or a persona little room.
- Probe families to try: forced-choice batteries (one prompt, about 60 numbered near-tie choices, one letter each); edge-of-ability batteries whose per-item errors fingerprint the weights; "random" sequences and their n-gram statistics; verbatim copy or continuation of a long input; plain long generation with a Hamming or n-gram MMD as the baseline to beat.
- LoRA strength 0.3, 0.7 and 1.5 is a dose: find what moves along it. Adapters on the scored endpoints may be about something else, so keep the probes broad.
- Quantization (ref-02) is the faintest signal. Measure how many answers it takes to separate it from ref-01, and say so if that is beyond about 60 to 100.
- Decide with tests that calibrate on the data at hand, such as a permutation test against within-endpoint noise, not with thresholds tuned on nine endpoints (36 pairs, 10 of them "same").
- Labels force transitive decisions: cluster, do not decide pairs one by one.
- Family is worth 0.1 and cannot be scored on the references. Put a few identity and style prompts in round 1; family is the coarse cluster, model the fine one inside it.

## Reply

The report path, the round-1 cost per endpoint, the reference MCC at those counts and its spread, answers per minute, what stays inseparable, and the recommendation for the remaining budget.

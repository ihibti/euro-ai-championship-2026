# Model equality: reference experiments, lab notebook

Started 2026-11-21 22:40 Paris. Only `ref-01`..`ref-09` were queried, always through `scripts/me_query.py` without `--endpoints`. Truth (`upstream/practice_answers.csv`): S = ref-01/03/05/06/08 (plain, system prompt, persona, watermark, temperature 1.0), T1 = ref-02 (fp8), T2/T3/T4 = ref-09/04/07 (LoRA 0.3/0.7/1.5).

## E0. Throughput (measured during the first run)

- Hypothesis: the script delivers well over 60 answers per minute, so the reference budget is time, not answers.
- Command: `uv run --extra api python scripts/me_query.py --prompts tasks/model_equality/probes/cand1.jsonl --repeats 6` (20 prompts, built by `probes/build_cand1.py`).
- Measured: the answer file grew from 4 to 12 lines in 50 s at 22:44, that is about 10 to 12 answers per minute with 4 parallel requests and mostly 256-token answers.
- Conclusion: wrong. The server is the bottleneck. At this rate a 40-answer round on 33 endpoints (1,320 answers) takes about 2 hours and the full 3,300 about 5 hours. The reference work is limited by wall time to a few hundred answers per experiment, not by the 15,000 cap.
- Arrival times logged for 90 s at 22:47: one answer every 5 to 6 s, whether the answer was 15 or 300 characters. Latency does not follow answer length, so short probes do not buy speed.

## E1. First look at 20 probe types (cand1, repeat 0 only)

- Hypothesis: itemised batteries (40 forced choices, 40 ratings, 30 sums, 30 trivia questions) give tens of observations per answer.
- Command: the E0 run, stopped by hand at 104 answers (repeat 0 of the first 12 prompts complete, `story` partial), to free the server for round 1. Resume with `--repeats 1` to finish repeat 0.
- What the reference model does (`first_look.py`):
  - It is a very weak model. It copies the format example instead of answering (`'1. 42'` gives forty 42s on `assoc40` for 02/03/05/09; `'1. 7'` gives forty 7s on `rate40`), echoes the problems back (`arith30` on 05 and 09), writes `0 1 2 3 ...` for random digits and `H T H T` for coin flips. Batteries carry far fewer observations than planned, and many are shared attractors that say nothing about the weights.
  - Answers stop at about 35 numbered lines: `'12. A\n'` costs about 7 tokens, so a battery fits about 30 items in 256 tokens.
  - The persona (05) can leave the task entirely (on `count30` it wrote a list of air pollution topics). The system prompt (03) changes format (words instead of letters on `choice40`, a preamble on `coins`). Temperature 1.0 (08) breaks the format most often.
  - `copy` (a 589-character passage) is reproduced exactly by every endpoint except LoRA 1.5 (07), which drifts after 225 characters. Uninformative below strong LoRA; dropped.
- Identical answers between endpoints, repeat 0:

| prompt | groups of byte-identical answers |
|---|---|
| count30 | {01, 03, 06} without a final period; {02, 04, 09} with `30.`; 07 adds a sentence; 05 and 08 off format |
| backwards20 | {01, 06} |
| assoc40 | {02, 05, 09}, all forty 42s (attractor) |
| arith30 | {05, 09}, problems echoed (attractor) |
| copy | everyone but 07 and 08 (perfect copy) |
| the other 7 | none |

- Conclusion (revised by round 1, repeat 0: the same `count30` prompt gave `ref-03` a preamble and `ref-09` a numbered list, so decoding is sampled, not greedy): decoding on constrained prompts is often but not always close to greedy, so exact agreement between two endpoints is frequent and meaningful when the output is a constrained sequence, and meaningless when it is a degenerate attractor. The final-period decision on `count30` separates plain/system prompt/watermark from fp8 and both weaker LoRAs on one answer each: a near-tie at the end of a fixed sequence is exactly the kind of decision a small weight change flips. Round 1 was built on this: five constrained sequences, two batteries cut to 30 items, two free-text prompts and one identity prompt.

## E2. Answer length

- Command: length summary over the 172 answers saved by 23:02 (cand1 and round 1).
- Median 241 characters (10% below 58, 10% above 816, longest 1,269). Free text reaches the 256-token cap at about 1,100 to 1,200 characters; a numbered list `'12. A'` reaches it at about 230 characters (about 7 tokens per line, 30 to 35 lines).
- Round-1 medians: `seq_*` 77 to 165 characters, `bat_trivia` 385, `bat_choice` 170, `txt_story` 867.
- The run log shows no retries and no rate-limit waits: each request simply takes about 24 s on the server with 4 in flight.

## Statistic used from here on

- Each answer becomes a sparse unit vector of observations, chosen by the prompt id's prefix (`group.py`):
  - `seq_*` (constrained sequence): the whole normalised answer, its first 24 and last 12 characters, and its word bigrams. Two answers share these only when they agree exactly at the start, at the end or everywhere.
  - `bat_*` (numbered battery): one observation per item, `item=first word of the answer`.
  - `txt_*` and `idn_*`: word counts.
- Kernel K between two sets of answers to one prompt: the mean cosine between their vectors; within one endpoint only distinct pairs count, so K_xx is unbiased.
- Similarity of endpoints x and y: K_xy / sqrt(K_xx K_yy), with K averaged over prompts first. It is about 1 when the two endpoints answer like two batches of one endpoint, whatever that endpoint's own spread (a hotter endpoint has a smaller K_xx and is not penalised for it). Distance = 1 - similarity.
- Within-endpoint noise: the same distance between an endpoint's even and odd repeats (needs at least 2 answers per half, so 4 repeats).

## E3. count30 tally across files (same prompt text in smoke, cand1 and round 1; 4 answers per endpoint)

| endpoint | canonical `1, ..., 30` | with final `30.` | other |
|---|---|---|---|
| 01 plain | 3 | 1 | 0 |
| 02 fp8 | 3 | 1 | 0 |
| 03 system prompt | 1 | 1 | 2 |
| 04 LoRA 0.7 | 0 | 4 | 0 |
| 05 persona | 1 | 0 | 3 |
| 06 watermark | 4 | 0 | 0 |
| 07 LoRA 1.5 | 0 | 0 | 4 |
| 08 temperature 1.0 | 1 | 0 | 3 |
| 09 LoRA 0.3 | 0 | 3 | 1 |

- Conclusion (4 answers, indicative only): the final period rises with LoRA strength (0.3 and 0.7 prefer it, 1.5 leaves the format), so it is a dose signal. fp8 answers exactly like plain here. The period difference seen on repeat 0 of cand1 for ref-02 was chance.

## Status at handoff (2026-11-21 23:12)

Stopped by the coordinator before items 3 to 5 of the brief. No separation table, no reference MCC and no `results/` record exist yet.

**Measured**
- Throughput: 8.0 answers per minute on round 1 (142 answers in 1,062 s), 9.3 on cand1, no retries or 429s. One request takes about 24 s on the server with 4 in flight; length does not matter. Scored plan at this rate: 40 answers x 33 endpoints = 1,320 answers = 2.2 to 2.75 h; all 3,300 = 5.5 to 7 h. The scored round 1 has to start tonight for a second round to fit before 11:00.
- Answer length: median 241 characters; free text hits the 256-token cap near 1,100 to 1,200 characters; a numbered battery near 30 to 35 lines.
- Determinism: not measured with 6 repeats yet. What exists: on `count30` ref-06 gave 4 identical answers out of 4 and ref-01/02 3 of 4, while 03/05/07/08 gave 4 distinct answers. Decoding is sampled, often near-greedy on constrained prompts.
- The reference model is very weak (E1): batteries collapse into copied examples and echoed questions, persona and temperature 1.0 often leave the task, copy is perfect for all but LoRA 1.5.

**Frozen**
- `probes/round1.csv` (built by `probes/build_round1.py`): 10 prompts, ids `seq_count30, seq_alphabet, seq_months, seq_days, seq_words40, bat_trivia, bat_choice, txt_story, txt_health, idn_identity`. At `--repeats 4` it costs 40 answers per endpoint. Do not edit it: the reference answers are under these ids.

**Not frozen**
- `analysis/group.py`: statistic and cut described above (factor 3 x RMS self-split distance, average linkage, families as a coarse merge of model clusters at distance 0.5 on `txt`/`idn` features). Never run on real data with 4 repeats; the factor and the family cut are unvalidated guesses. It now refuses to run with fewer than 4 repeats per prompt. Known weakness: an endpoint whose answers share no feature (K_xx near 0, typical of ref-08 on free text) makes the pooled ratio explode because of the `1e-9` floor; check its row before trusting a cut, or use `--mode prompt` (per-prompt normalisation, skips such prompts).
- Its tests are in `analysis/test_group.py`, outside the repo's `testpaths`: run `uv run pytest tasks/model_equality/analysis`.

**Answer files** (all `ref-*`, all committed)
- `answers/smoke.jsonl`: 18 answers, 2 prompts x 9 endpoints (earlier session).
- `answers/cand1.jsonl`: 104 answers, `probes/cand1.jsonl` (20 prompt types), repeat 0 complete for the first 12 prompts (`count30` to `copy`), `story` on 5 endpoints, the 8 domain prompts not asked.
- `answers/round1.jsonl`: 142 answers, `probes/round1.csv`; repeat 0 complete (90), repeat 1 complete for the five `seq_*` prompts and 7 of 9 endpoints of `bat_trivia`.

**Next steps**
1. Finish 4 repeats of round 1 on the references (218 more answers, about 27 min). Only one querying process at a time; to stop it, kill the `.venv/bin/python -u scripts/me_query.py` process by PID (a `pkill` pattern without `-u` misses it).
   `uv run --extra api python scripts/me_query.py --prompts tasks/model_equality/probes/round1.csv --repeats 4`
2. Determinism and cross-endpoint identical-answer rates:
   `uv run python tasks/model_equality/analysis/collisions.py tasks/model_equality/answers/round1.jsonl --matrix`
3. Separation table (items (a), (b), (c) of the brief) per prompt and feature, and the temperature-robust choice table:
   `uv run python tasks/model_equality/analysis/explore.py tasks/model_equality/answers/round1.jsonl --repeats 0-3`
   `uv run python tasks/model_equality/analysis/explore.py tasks/model_equality/answers/round1.jsonl --choice`
4. Grouping at exactly 40 answers, printed distance matrix and self-split noise, then the scored record:
   `uv run python tasks/model_equality/analysis/group.py --prefix ref- --answers tasks/model_equality/answers/round1.jsonl --below 4 --out tasks/model_equality/analysis/round1_refs.csv`
   `make score TASK=model_equality ANSWER=tasks/model_equality/analysis/round1_refs.csv NOTES="round1 x4, pooled kernel seq+bat+txt, avg linkage cut 3x rms self-split"`
5. Ablations at the same count (do not pick a winner on 9 endpoints by MCC alone; prefer the one with a reason):
   `uv run python tasks/model_equality/analysis/evaluate.py --answers tasks/model_equality/answers/round1.jsonl --per-prompt 4 --draws 0` with `--kinds seq`, `--kinds seq,bat`, `--mode prompt`.
6. Collect repeats 4 to 7 for resampling (360 answers, about 45 min), then the spread of the MCC at 4 repeats per prompt and the MCC at 8:
   `uv run --extra api python scripts/me_query.py --prompts tasks/model_equality/probes/round1.csv --repeats 8`
   `uv run python tasks/model_equality/analysis/evaluate.py --answers tasks/model_equality/answers/round1.jsonl --per-prompt 4 --draws 200`
   `uv run python tasks/model_equality/analysis/evaluate.py --answers tasks/model_equality/answers/round1.jsonl --per-prompt 8 --draws 0`
7. LoRA direction and dose (which words rise from 0.3 to 0.7 to 1.5):
   `uv run python tasks/model_equality/analysis/lora_dose.py tasks/model_equality/answers/round1.jsonl --ids txt_story,txt_health`
8. Optional, for the LoRA domain: finish repeat 0 of cand1 (76 answers, about 9 min), which asks the 8 domain prompts once each:
   `uv run --extra api python scripts/me_query.py --prompts tasks/model_equality/probes/cand1.jsonl --repeats 1`
9. Only once steps 4 to 6 show round 1 is worth it, the human spends the scored round (1,320 answers, 2.2 to 2.75 h, types 1320 to confirm); the command is valid nu:
   `uv run --extra api python scripts/me_query.py --prompts tasks/model_equality/probes/round1.csv --name round1-ep --endpoints ep --repeats 4 --spend`
   then commit `answers/round1-ep.jsonl` at once and run step 4 with `--prefix ep- --answers tasks/model_equality/answers/round1-ep.jsonl`.
10. Recommendation for the remaining 60 is open. The default to test in step 6: if the MCC at 8 repeats is clearly above the MCC at 4, spend 40 more of the same round-1 prompts (validated on the references at 80) and hold 20 back. Merge two look-alike endpoints when nothing separates them: among look-alikes the references hold 5 same-model copies for one fp8 copy.

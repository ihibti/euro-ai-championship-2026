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

## E4. Round 1 at 4 repeats: throughput and determinism (2026-11-21 23:55)

- Hypothesis: constrained sequences are near-greedy, so exact repeats within an endpoint are common and exact agreement between endpoints marks shared weights.
- Commands: `uv run --extra api python scripts/me_query.py --prompts tasks/model_equality/probes/round1.csv --repeats 4` (218 answers, 23:30:31 to 23:54:23), then `uv run python tasks/model_equality/analysis/collisions.py tasks/model_equality/answers/round1.jsonl --matrix`.
- Throughput: 218 answers in 1,432 s, **9.1 answers per minute**, 0 failed, 0 skipped. With E0 and the first part of round 1, the rate is 8.0 to 9.3 per minute. The scored round 1 (1,320 answers) therefore takes 2.4 to 2.75 h.
- `answers/round1.jsonl` now holds 360 answers: 10 prompts x 9 endpoints x repeats 0 to 3, no nulls, no duplicates.
- Within-endpoint identical-pair rate (6 pairs per cell). The two batteries, the two free-text prompts and the identity prompt are 0.00 everywhere, except `bat_choice` on 06 (0.17):

| prompt | 01 | 02 | 03 | 04 L.7 | 05 | 06 | 07 L1.5 | 08 | 09 L.3 |
|---|---|---|---|---|---|---|---|---|---|
| seq_alphabet | 0.00 | 0.00 | 0.00 | 0.00 | 0.17 | 0.17 | 0.00 | 0.00 | 0.00 |
| seq_count30 | 0.17 | 0.50 | 0.17 | 0.50 | 0.00 | 0.50 | 0.00 | 0.00 | 0.17 |
| seq_days | 0.17 | 0.17 | 0.17 | 1.00 | 0.17 | 0.00 | 0.00 | 0.00 | 0.50 |
| seq_months | 0.00 | 0.17 | 0.00 | 1.00 | 0.00 | 0.17 | 0.00 | 0.00 | 0.50 |
| seq_words40 | 0.00 | 0.00 | 0.00 | 0.17 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

- Between-endpoint identical rate, averaged over the 10 prompts: the highest are 04-09 (0.19), 03-04 (0.11) and 02-06 (0.10). Plain 01 is at 0.03 with itself and 0.02 to 0.07 with everyone else. Nothing ever matches 07.
- Conclusion: no endpoint is deterministic. The plain model repeats itself in at most 1 pair out of 6 on a sequence: it answers `seq_alphabet` with symbols, the alphabet or a refusal, and `seq_days` starting on Sunday or Monday. The one sharp endpoint is LoRA 0.7: it gives the same answer 4 times out of 4 on days and months, and LoRA 0.3 does in half the pairs. The adapter sharpens these sequences, and 1.5 leaves the format altogether, so this is a dose signal on 04 and 09. At 4 answers, exact agreement is too rare to separate anything else.

## E5. Separation table at 4 repeats (2026-11-21 23:56)

- Hypothesis: on some probe type, the distance between disguised copies of one model (a) stays at the split-half noise, while fp8 (b) and the LoRA copies (c) rise above it.
- Commands: `uv run python tasks/model_equality/analysis/explore.py tasks/model_equality/answers/round1.jsonl --repeats 0-3` and `... --choice`.
- Every distance is computed between halves of 2 answers (even against odd repeats). The exact-match features (`full`, `head`, `tail`) are undefined (nan) or 0/1 on nearly every cell at this count, and so is the composite `seq` feature, so they are left out. The smooth features (`noise` = one endpoint's halves against each other; S = 01/03/05/06/08, Q = 02, L = 09/04/07):

| prompt | feature | noise | a:S-S | b:Q-S | b:Q-01 | c:L.3-S | c:L.7-S | c:L1.5-S | c:L-L |
|---|---|---|---|---|---|---|---|---|---|
| bat_choice | bat | 0.03 | 0.09 | 0.08 | 0.10 | 0.08 | 0.05 | 0.09 | 0.10 |
| bat_choice | words | 0.00 | 0.02 | 0.02 | 0.04 | 0.02 | 0.01 | 0.01 | 0.00 |
| bat_trivia | bat | 0.03 | 0.04 | 0.06 | 0.04 | 0.05 | 0.11 | 0.10 | 0.08 |
| bat_trivia | words | 0.02 | 0.01 | 0.04 | 0.06 | 0.05 | 0.05 | 0.04 | 0.06 |
| bat_trivia | char3 | 0.01 | 0.01 | 0.03 | 0.04 | 0.03 | 0.04 | 0.02 | 0.07 |
| idn_identity | words | 0.08 | 0.14 | 0.14 | 0.06 | 0.15 | 0.06 | 0.20 | 0.19 |
| seq_count30 | char3 | 0.19 | 0.26 | 0.18 | 0.00 | 0.09 | 0.18 | 0.08 | 0.00 |
| seq_days | words | 0.01 | 0.00 | 0.02 | 0.03 | 0.00 | 0.00 | 0.22 | 0.19 |
| seq_months | words | 0.08 | 0.30 | 0.25 | 0.95 | 0.25 | 0.25 | 0.27 | 0.35 |
| seq_words40 | words | 0.06 | 0.13 | 0.00 | 0.00 | 0.05 | 0.02 | 0.04 | 0.01 |
| txt_health | words | 0.03 | 0.02 | 0.04 | 0.01 | 0.00 | 0.01 | 0.07 | 0.03 |
| txt_story | words | 0.01 | 0.01 | 0.01 | 0.02 | 0.01 | 0.03 | 0.03 | 0.03 |

- `bat_choice`, Spearman of per-item P(A): the split-half diagonal is 0.04, 0.31, 0.00, 0.45, -0.02, 1.00, -0.05, 0.27 and 0.06 for 01 to 09. Two answers per half do not estimate a per-item rate, so the temperature-robust rank statistic cannot be used at this count.
- Conclusion: one prompt does what the hypothesis asks. On `bat_trivia`, with words or char3 features, the disguised pairs sit at 0.01 and fp8 and every LoRA at 0.02 to 0.07, three to six times higher, against a split-half noise of 0.01 to 0.02. Knowledge errors carry the weights, and the rigid 30-line format mutes the disguises. Elsewhere the disguises are as loud as the weight changes or louder: the `seq_*` prompts are dominated by format changes (S-S 0.13 to 0.38 on words), and the free-text prompts sit at the noise for every pair. `seq_days` separates only LoRA 1.5. One answer of 4 is a weak base for any of this: every cell compares 2 answers with 2.

## E6. Grouping at exactly 40 answers per endpoint (2026-11-21 23:58)

- Hypothesis: `group.py` (pooled kernel over the 9 `seq`/`bat`/`txt` prompts, average linkage cut at 3 x RMS self-split distance) recovers S, T1, T2, T3 and T4 from repeats 0 to 3.
- Commands:
  `uv run python tasks/model_equality/analysis/group.py --prefix ref- --answers tasks/model_equality/answers/round1.jsonl --below 4 --out tasks/model_equality/analysis/round1_refs.csv`
  `make score TASK=model_equality ANSWER=tasks/model_equality/analysis/round1_refs.csv NOTES="..."`
  and the same with `--mode prompt` and `--out tasks/model_equality/analysis/round1_refs_prompt.csv`. `group.py` gained `--mode`, `--kinds` and a per-endpoint printout of its own kernel K_xx for this check.
- Pooled distance matrix (1 - similarity), self-split distance, and K_xx (mean over the 9 prompts, minimum, number of prompts at exactly 0):

| | 01 | 02 | 03 | 04 | 05 | 06 | 07 | 08 | 09 | self-split | K_xx mean | min | at 0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 01 | 0 | 0.06 | -0.05 | 0.15 | -0.08 | 0.09 | 0.13 | 0.06 | 0.09 | 0.07 | 0.30 | 0.00 | 1 |
| 02 | 0.06 | 0 | 0.03 | 0.25 | 0.10 | 0.01 | 0.12 | 0.10 | 0.14 | -0.09 | 0.38 | 0.00 | 0 |
| 03 | -0.05 | 0.03 | 0 | 0.04 | -0.03 | 0.11 | 0.06 | 0.07 | -0.00 | -0.13 | 0.33 | 0.00 | 0 |
| 04 | 0.15 | 0.25 | 0.04 | 0 | 0.17 | 0.29 | 0.27 | 0.34 | -0.01 | -0.06 | 0.59 | 0.03 | 0 |
| 05 | -0.08 | 0.10 | -0.03 | 0.17 | 0 | 0.11 | 0.10 | 0.13 | 0.09 | -0.21 | 0.26 | 0.01 | 0 |
| 06 | 0.09 | 0.01 | 0.11 | 0.29 | 0.11 | 0 | 0.19 | 0.12 | 0.18 | -0.05 | 0.41 | 0.04 | 0 |
| 07 | 0.13 | 0.12 | 0.06 | 0.27 | 0.10 | 0.19 | 0 | 0.08 | 0.15 | 0.06 | 0.29 | 0.00 | 0 |
| 08 | 0.06 | 0.10 | 0.07 | 0.34 | 0.13 | 0.12 | 0.08 | 0 | 0.23 | 0.01 | 0.22 | 0.00 | 1 |
| 09 | 0.09 | 0.14 | -0.00 | -0.01 | 0.09 | 0.18 | 0.15 | 0.23 | 0 | -0.09 | 0.39 | 0.03 | 0 |

- Pooled: cut 0.307, **one model for all nine, MCC 0.000** (TP 10, FP 26, FN 0, TN 0). Record `results/20261121T225541Z_ihibti_practice-mcc.json`.
- `--mode prompt`: per-prompt ratios on halves of 2 answers explode (distances down to -3.24, self-split down to -2.65), cut 3.435, **one model, MCC 0.000**. Record `results/20261121T225541Z_ihibti_practice-mcc-2.json`.
- Floor check: no endpoint has a pooled K_xx near 0. The lowest is ref-08 at 0.22, so the `1e-9` floor never acts in pooled mode. Only ref-01 and ref-08 have a prompt at exactly 0, `seq_alphabet` in both cases; on `seq_alphabet`, `seq_months` and `seq_words40` most endpoints are at 0.00 to 0.03. The failure is not the floor. The self-split distance compares 2 answers with 2, with one pair for each half's K_xx, so it is noise of about ±0.1 (negative for 6 of 9 endpoints). Its RMS (0.10) times 3 lies above every pair but 04-08. The pooled kernel is also dominated by the two free-text prompts (K_xx 0.4 to 0.87, mostly function words), where every pair sits at the noise.
- Permutation test of "same answer distribution" per pair (`analysis/permtest.py`, pooled kernel, 4 against 4 answers per prompt, 200 permutations; the smallest p-value possible is 0.005):

| | plain 01 | fp8 02 | sys 03 | L.7 04 | persona 05 | wm 06 | L1.5 07 | T1.0 08 | L.3 09 |
|---|---|---|---|---|---|---|---|---|---|
| plain 01 | - | 0.154 | 0.746 | 0.010 | 0.846 | 0.030 | 0.005 | 0.184 | 0.080 |
| fp8 02 | 0.154 | - | 0.338 | 0.005 | 0.080 | 0.318 | 0.010 | 0.025 | 0.065 |
| sys 03 | 0.746 | 0.338 | - | 0.124 | 0.582 | 0.055 | 0.055 | 0.055 | 0.408 |
| L.7 04 | 0.010 | 0.005 | 0.124 | - | 0.005 | 0.005 | 0.005 | 0.005 | 0.458 |
| persona 05 | 0.846 | 0.080 | 0.582 | 0.005 | - | 0.040 | 0.015 | 0.010 | 0.100 |
| wm 06 | 0.030 | 0.318 | 0.055 | 0.005 | 0.040 | - | 0.010 | 0.040 | 0.025 |
| L1.5 07 | 0.005 | 0.010 | 0.055 | 0.005 | 0.015 | 0.010 | - | 0.005 | 0.015 |
| T1.0 08 | 0.184 | 0.025 | 0.055 | 0.005 | 0.010 | 0.040 | 0.005 | - | 0.005 |
| L.3 09 | 0.080 | 0.065 | 0.408 | 0.458 | 0.100 | 0.025 | 0.015 | 0.005 | - |

- 4 of the 10 same-model pairs reject at p < 0.05 (the watermark and temperature copies against others), and 15 of the 26 different-model pairs do.
- Conclusion: at 40 answers per endpoint, round 1 with this statistic does not recover the reference structure. Neither the default cut nor any threshold on these distances can: 03 (system prompt) is closer to 09 (-0.00) and 04 (0.04) than to 06 (0.11).
  - Inseparable at this count: fp8 02 against plain 01 (p 0.15), LoRA 0.3 09 against plain (0.08), and LoRA 0.3 against LoRA 0.7 (0.46).
  - Separated from plain: LoRA 0.7 and LoRA 1.5 (p 0.010 and 0.005).
  - The watermark and temperature copies differ from plain about as much as fp8 does, so a test of "same distribution" cannot be the decision rule for disguised copies either.
  - `group.py` is not fit to run unchanged on `round1-ep.jsonl`: on the references it calls every endpoint of a family one model, which gives MCC_model 0.

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

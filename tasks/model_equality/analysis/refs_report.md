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

## E7. Ablations of group.py at 40 answers per endpoint (2026-11-21 23:59)

- Hypothesis: a subset of the prompt kinds, or the per-prompt mode, rescues the grouping at the same count.
- Command: `uv run python tasks/model_equality/analysis/evaluate.py --answers tasks/model_equality/answers/round1.jsonl --per-prompt 4 --draws 0` plus the options below (run under bash: an unquoted `$opts` in zsh is not split, and the argparse error was hidden at first).

| options | MCC | grouping |
|---|---|---|
| default (pooled, seq,bat,txt, factor 3) | 0.000 | all one |
| `--kinds seq` | 0.000 | all one |
| `--kinds seq,bat` | 0.000 | all one |
| `--kinds bat` | 0.000 | all one |
| `--kinds txt` | 0.000 | all one |
| `--kinds bat,txt` | 0.000 | all one |
| `--ids bat_trivia --kinds bat` | 0.000 | all one |
| `--mode prompt` | 0.000 | all one |
| `--factor 2` | 0.000 | all one |
| `--factor 2 --kinds bat,txt` | 0.000 | all one |
| `--factor 1 --kinds bat,txt` | 0.000 | all one |
| `--factor 1` | 0.351 | {01,02,03,05,06} {04,09} {07,08} |

- Conclusion: only a factor tuned down to 1 splits anything, and it pairs LoRA 1.5 with the temperature copy. That is a threshold chosen on 9 endpoints, not a method. At this count round 1 has no reference MCC above 0 under the rule as designed. Resampling at 4 repeats (old step 6) is moot until a statistic separates anything.

## E8. Free side channels on the answers already held (2026-11-22 00:03)

- Latency: `answer_row` (in `src/euroai/me_query.py`) saves neither a timestamp, nor `finish_reason`, nor token usage, so latency and the 256-token cap cannot be read from saved answers. Proxy for latency: the lag between each answer's planned slot (repeat, then prompt, then endpoint) and its line in the file, over the 218 answers of the second round-1 run. The mean lag per endpoint is between -0.04 and +0.04 jobs, and the median is 0 for all nine. Answers come back in submission order whatever the endpoint, so the order carries no latency signal.
- Length (all 482 reference answers in `smoke`, `cand1` and `round1`, characters):

| endpoint | n | median | p90 | share >= 1,000 | ends mid-sentence |
|---|---|---|---|---|---|
| 01 plain | 54 | 180 | 937 | 0.09 | 0.57 |
| 02 fp8 | 54 | 197 | 831 | 0.07 | 0.57 |
| 03 system prompt | 54 | 178 | 857 | 0.06 | 0.48 |
| 04 LoRA 0.7 | 54 | 233 | 1,112 | 0.15 | 0.31 |
| 05 persona | 54 | 199 | 1,070 | 0.11 | 0.57 |
| 06 watermark | 53 | 170 | 729 | 0.08 | 0.62 |
| 07 LoRA 1.5 | 53 | 612 | 1,250 | 0.34 | 0.64 |
| 08 temperature 1.0 | 53 | 217 | 953 | 0.09 | 0.49 |
| 09 LoRA 0.3 | 53 | 226 | 825 | 0.08 | 0.43 |

- On `txt_health` LoRA 1.5 hits the cap 4 times out of 4 (1,229 to 1,335 characters); the others vary from 211 to 1,248.
- Conclusion: length carries the LoRA dose at 0.7 and 1.5 and nothing at fp8 or LoRA 0.3. Recording `finish_reason`, `usage.completion_tokens` and the wall-clock time in `answer_row` would make the cap and latency checkable at no cost. That file is shared code outside this task's folder, so the change goes through the lead, before the scored spend if at all.

## E9. night_a: P1 (copy then continue) and P2 (binary sequence) at 10 repeats (2026-11-22 00:20)

- Hypothesis (`research_night.md`, sections 1 and 3): decoding is near-greedy on format-pinned long outputs, so the plurality token per position is stable within an endpoint. fp8 and LoRA then flip some positions, while the disguises leave the pluralities in place. A probe passes when a different-model distance from ref-01 exceeds the largest distance among the five disguises 01/03/05/06/08.
- Commands: `probes/build_night.py` wrote `probes/night_a.csv` and `probes/night_b.csv`, with the texts checked character by character against the code blocks of section 3. Then `uv run --extra api python scripts/me_query.py --prompts tasks/model_equality/probes/night_a.csv --repeats 10` (180 answers in 1,134 s, 9.5 per minute, 0 failed) and `uv run python tasks/model_equality/analysis/plurality.py --answers tasks/model_equality/answers/night-a.jsonl`.
- Statistic (`analysis/plurality.py`, tested): tokens are the words after the copied passage for P1, located by the passage's last 4 words, and the 0/1 symbols for P2, after stripping line-start list markers and splitting pure 0/1 runs. The plurality per position is taken over the 10 repeats, and a repeat without a token at a position counts against it. A position is kept at share >= 0.6. The distance is the disagreement rate on positions kept by both. No other parameter.
- Self-agreement (mean plurality share), kept positions, split-half distance (5 repeats against 5, shared positions in parentheses), and mean first-divergence position between two repeats:

| endpoint | P1 self-agreement | P1 kept of 80 | P1 self first-div (words) | P2 self-agreement | P2 kept of 60 | P2 split-half | P2 self first-div |
|---|---|---|---|---|---|---|---|
| 01 plain | 0.04 | 0 | 0.1 | 0.42 | 7 | 0.62 (13) | 2.7 |
| 02 fp8 | 0.01 | 0 | 0.0 | 0.34 | 8 | 0.33 (9) | 3.0 |
| 03 system prompt | 0.11 | 1 | 0.7 | 0.27 | 2 | 0.00 (2) | 1.5 |
| 04 LoRA 0.7 | 0.02 | 0 | 0.1 | 0.36 | 9 | 0.25 (4) | 1.2 |
| 05 persona | 0.08 | 1 | 0.9 | 0.41 | 2 | 0.33 (3) | 6.7 |
| 06 watermark | 0.13 | 3 | 3.2 | 0.46 | 31 | 0.00 (8) | 10.6 |
| 07 LoRA 1.5 | 0.18 | 0 | 0.4 | 0.57 | 28 | 0.37 (30) | 3.7 |
| 08 temperature 1.0 | 0.01 | 0 | 0.0 | 0.26 | 1 | nan (0) | 0.6 |
| 09 LoRA 0.3 | 0.04 | 0 | 0.1 | 0.38 | 2 | nan (0) | 1.7 |

- The deciding table (distance from ref-01, positions kept by both in parentheses):

| pair | P1 distance | P2 distance | P2 cross first-div |
|---|---|---|---|
| 01-03 same model | undefined (0) | 0.00 (2) | 2.2 |
| 01-05 same model | undefined (0) | 0.00 (2) | 1.5 |
| 01-06 same model | undefined (0) | 0.00 (7) | 5.0 |
| 01-08 same model | undefined (0) | 0.00 (1) | 1.0 |
| largest of the 10 disguise pairs | 0.00 (3 of 10 pairs defined) | 0.00 (all 10 defined) | |
| 01-02 fp8 | undefined (0): **fail** | 0.00 (5): **fail** | 2.9 |
| 01-09 LoRA 0.3 | undefined (0): **fail** | 0.00 (1): **fail** | 2.4 |
| 01-04 LoRA 0.7 | undefined (0): **fail** | 0.00 (3): **fail** | 2.2 |
| 01-07 LoRA 1.5 | undefined (0): **fail** | 0.00 (4): **fail** | 3.1 |

- P1, copy part: perfect copies out of 10 are 10 for plain, fp8 and watermark; 9 for LoRA 0.7 and 1.5; 8 for the system prompt; 7 for the persona and LoRA 0.3; 3 for temperature 1.0. The disguises alone span 3 to 10, and fp8 equals plain, so copy errors are not a weight signal either.
- The only non-zero P2 distances are between the endpoints that keep many positions: 04-07 0.67 (6), 04-06 0.22 (9), 06-07 0.20 (20), 02-07 0.17 (6) and 02-06 0.12 (8). The watermark copy (06) keeps more positions than plain because it falls into the same `0 1 0 1` alternation every time. That is an attractor, not a near-tie.
- Conclusion: both probes fail for every pair type: fp8, LoRA 0.3, LoRA 0.7 and LoRA 1.5. The premise does not hold on the reference model: it is not near-greedy on these prompts. Two repeats of plain ref-01 part at the first continuation word on P1 (mean first divergence 0.1 words), and on P2 the plain model keeps 7 positions out of 60, with its own halves disagreeing on 62% of the positions they share. The P2 positions that are kept are format attractors (a leading `0`, the alternation) that every endpoint shares, which is why all distances from ref-01 are 0. The scored models may be larger and closer to greedy, but nothing on the references supports spending scored answers on P1 or P2.

## E10. night_c design: three categorical batteries, not yet asked (2026-11-22 00:28)

- Constraint from the desk: each answer carries 20 to 30 independent categorical draws, with no worked example in the prompt.
- Constraint from the data: whether the plain model answers in format depends on one line of the instruction. Mean numbered items parsed per answer:
  - round-1 batteries, which say "Reply with exactly 30 lines in the form '1. A'": 20 to 30 of 30 on every endpoint, plain 24 and 28, temperature 1.0 25 and 28;
  - P4, which has no such line (`night_b`, first 4 repeats): plain 0.5 of 20, temperature 0.8, against 14 to 17.5 for fp8 and the LoRAs. P4's distances therefore measure format compliance first.
  - What the model copies is a plausible answer inside the example: `'1. 42'` gave forty 42s in E1. `night_c` keeps the format line and puts a placeholder in it (`'number. option'`), never an answer.
- `probes/night_c.csv` (built by `probes/build_night_c.py`, 225 to 289 input tokens, answers about 100 to 170 tokens):
  - `c1_equiv` (candidate a): 25 lines of four equivalent options with a fixed order, mixing function names that do the same thing, synonyms and arbitrary equal picks. The draw is the option written. A line counts only when exactly one of its options appears, so an echoed line is not read as a first-option pick.
  - `c2_edge` (candidate c, from E5): 25 one-word questions near the edge of a small model's knowledge (capitals that are not the largest city, rare chemical symbols, dates, two arithmetic items). The draw is the first word of the answer. In round 1, knowledge errors (`bat_trivia`) were the only weight signal larger than the disguises.
  - `c3_next` (candidate b): P4's 20 sentence starts with the format line added. The draw is the word after the echoed start, or a bare one-word answer. This is the fair test of the next-word idea, since P4 itself is confounded with compliance.
- Statistic: `analysis/choices.py` (tested). It counts the chosen option per item and endpoint and compares two endpoints by the mean per-item total variation, and by a temperature-robust mean per-item Kendall tau-b distance between the two count vectors, which sees only the order of preference. Each is calibrated by a permutation test that shuffles whole answers between the two endpoints. The pass rule is unchanged.
- Command for the references, when the desk schedules it (at 10 repeats: 270 answers, about 28 min): `uv run --extra api python scripts/me_query.py --prompts tasks/model_equality/probes/night_c.csv --repeats 10`. Analysis: `uv run python tasks/model_equality/analysis/choices.py --answers tasks/model_equality/answers/night-c.jsonl`.

## E11. night_b: P4 (next-word battery) and P3 (repetition count) at 10 repeats (2026-11-22 00:39)

- Hypothesis (`research_night.md`, section 3): fp8 flips a few near-tie next words and LoRA flips more (P4); the stop decision of a long repetition is a near-tie that the weights move (P3).
- Commands: `uv run --extra api python scripts/me_query.py --prompts <copy of the p4_nextword and p3_repeat rows of probes/night_b.csv> --name night_b --repeats 10` (180 answers in 1,077 s, 10.0 per minute, 0 failed). Then `uv run python tasks/model_equality/analysis/choices.py --answers tasks/model_equality/answers/night-b.jsonl --draws 500` and `uv run python tasks/model_equality/analysis/plurality.py --answers tasks/model_equality/answers/night-b.jsonl`. The P5 ids were not asked, on the desk's order: one draw per answer is too weak.
- P4 compliance per repeat: numbered items parsed (of 20) and next words extracted (the word after the echoed sentence start, or a bare one-word answer):

| endpoint | numbered items per repeat | mean | answers with >= 10 | next words per repeat | mean |
|---|---|---|---|---|---|
| 01 plain | 0 0 15 0 19 0 1 0 0 0 | 3.5 | 2 of 10 | 0 0 2 0 0 0 1 0 0 0 | 0.3 |
| 03 system prompt | 19 18 20 18 0 20 20 0 18 20 | 15.3 | 8 of 10 | 11 14 8 2 0 20 16 0 18 12 | 10.1 |
| 05 persona | 1 20 20 20 20 20 20 20 20 0 | 16.1 | 8 of 10 | 1 13 19 18 16 19 10 11 17 0 | 12.4 |
| 06 watermark | 0 20 0 0 0 0 0 19 0 20 | 5.9 | 3 of 10 | 0 17 0 0 0 0 0 0 0 6 | 2.3 |
| 08 temperature 1.0 | 0 1 1 1 19 0 1 0 20 20 | 6.3 | 3 of 10 | 0 1 1 1 0 0 1 0 14 14 | 3.2 |
| 02 fp8 | 19 20 20 20 20 19 0 19 20 20 | 17.7 | 9 of 10 | 14 9 18 15 9 0 0 1 11 18 | 9.5 |
| 09 LoRA 0.3 | 20 1 20 0 0 19 19 20 10 20 | 12.9 | 7 of 10 | 13 1 20 0 0 0 13 20 10 14 | 9.1 |
| 04 LoRA 0.7 | 20 20 19 20 20 20 19 20 20 19 | 19.7 | 10 of 10 | 19 19 12 20 15 7 10 16 14 11 | 14.3 |
| 07 LoRA 1.5 | 20 20 20 20 20 20 20 0 7 20 | 16.7 | 8 of 10 | 16 20 14 15 12 19 0 0 4 1 | 10.1 |

- Compliance is noise, not an fp8 signal. The five same-model copies alone span 2 to 8 answers in format of 10, and fp8 (9) sits with the system-prompt and persona copies (8 each), not apart from them. A system prompt or a persona makes this model more compliant.
- P4 next words against plain (mean per-item total variation, permutation p over whole answers, 500 permutations): the disguise spread is 0.67 to 1.00, and fp8 0.95, LoRA 0.3 1.00, LoRA 0.7 0.89, LoRA 1.5 0.85. **Fail for all four.** With the rank statistic, the spread is 0.61 to 1.00 against 0.92, 0.92, 0.63 and 0.54: **fail**. Plain yields 0.3 next words per answer, so every distance from it rests on a few items. Among compliant answers only (>= 10 next words), system prompt against persona, the same model, is 0.67, and system prompt against fp8 is 0.70 (p 0.76). Every pair falls between 0.62 and 0.78. Nothing separates fp8.
- P3 repetition counts per repeat:

| endpoint | counts of "river" | answers running to the token cap |
|---|---|---|
| 01 plain | 256 9 256 1 52 1 1 0 256 10 | 3 of 10 |
| 03 system prompt | 1 1 1 3 9 1 3 1 1 1 | 0 |
| 05 persona | 256 128 1 1 0 128 1 1 1 1 | 3 |
| 06 watermark | 9 128 8 1 1 36 1 1 0 256 | 2 |
| 08 temperature 1.0 | 2 1 3 17 1 12 7 1 5 13 | 0 |
| 02 fp8 | 5 7 128 1 1 1 1 1 1 256 | 2 |
| 09 LoRA 0.3 | 256 128 6 14 256 12 0 256 256 256 | 6 |
| 04 LoRA 0.7 | 128 256 256 256 128 256 57 256 6 256 | 8 |
| 07 LoRA 1.5 | 17 256 256 256 128 256 70 241 256 0 | 7 |

- No count lies between 71 and 127. An answer either stops early or runs to the 256-token cap (256 spaced or 128 comma-separated repetitions); the requested 150 is never written. On the raw counts, total variation and the rank statistic fail for every pair (disguise spread 0.30 to 0.70 and 0.20 to 0.76, against at most 0.70 and 0.66).
- The run-to-cap rate is the first statistic of the night that separates LoRA from the disguises. Two-sided Fisher exact tests against the pooled answers of the five same-model copies (40 when an endpoint is compared with the other four):
  - LoRA 0.3: 6 of 10 against 8 of 50, p 0.007;
  - LoRA 0.7: 8 of 10, p 0.0002;
  - LoRA 1.5: 7 of 10, p 0.001;
  - fp8: 2 of 10, p 0.67;
  - each disguise against the other four: p 0.18 to 0.65.
  Against plain alone (10 against 10) the p-values are 0.37, 0.07 and 0.18. Under the desk's pass rule (difference in rate from plain > largest difference among disguises, 0.3): LoRA 0.7 (0.5) and LoRA 1.5 (0.4) pass; LoRA 0.3 (0.3) ties, so fails; fp8 (0.1) fails.
- Conclusion: P4 fails for every pair type; its signal is format compliance, which the disguises move as much as the weights. P3 fails as a count but carries a LoRA signal in its stop decision. That was the research's prediction (the stop is a near-tie) on a binary feature: the LoRA copies fail to stop. The feature was defined after looking at these answers, at a natural gap, and needs a confirmation run on fresh answers before it decides a scored pair. It does nothing for fp8. On a scored family it would be read against that family's own majority of same-model copies, which needs the family and its disguised copies to be found first.

## E12. Length and cap hits per prompt from existing answers; which prompts carry the LoRA dose; a post hoc fp8 candidate (2026-11-22 00:55)

- Question (desk): for every prompt already asked, answer length and cap-hit rate per reference. For fp8, L0.3, L0.7 and L1.5, the gap from the same-model copies against the spread among the five copies. Which prompts carry the LoRA dose?
- Command: `uv run python tasks/model_equality/analysis/lengths.py --answers tasks/model_equality/answers/round1.jsonl tasks/model_equality/answers/night-a.jsonl tasks/model_equality/answers/night-b.jsonl` (tested).
  - Gap: the endpoint's value minus that of the pooled answers of the copies (for a copy, of the other four).
  - Spread: the largest absolute gap among the copies.
  - `*`: absolute gap above the spread.
  - Cap hit: the frozen rule of `stop.py`.
- Median characters per answer (plain), and each model's gap:

| prompt | n | plain | fp8 | L0.3 | L0.7 | L1.5 | copies spread |
|---|---|---|---|---|---|---|---|
| seq_count30 | 4 | 110 | +0 | +2 | +1 | +164 * | 22 |
| seq_alphabet | 4 | 108 | +106 * | +80 * | +130 * | +367 * | 58 |
| seq_months | 4 | 122 | -18 | -18 | -18 | +1048 * | 72 |
| seq_days | 4 | 63 | +12 | -0 | -0 | +1064 * | 48 |
| seq_words40 | 4 | 508 | +6 | +4 | -58 | +218 | 291 |
| bat_trivia | 4 | 387 | -137 * | -133 * | +22 | +4 | 58 |
| bat_choice | 4 | 170 | +0 | +0 | +0 | +332 * | 54 |
| txt_story | 4 | 1145 | -124 | +78 | +78 | +90 | 154 |
| txt_health | 4 | 753 | -34 | -144 | +114 | +472 * | 431 |
| idn_identity | 4 | 198 | -104 | +15 | +209 | +599 * | 242 |
| p1_copycont | 10 | 514 | -58 * | -37 * | -58 * | +730 * | 30 |
| p2_binary | 10 | 234 | -164 * | +0 | +0 | +60 | 80 |
| p4_nextword | 10 | 486 | +288 | +93 | +323 | +411 | 456 |
| p3_repeat | 10 | 56 | -32 | +1178 * | +1498 * | +1498 * | 115 |

- Cap hits per endpoint (01 03 05 06 08 02 09 04 07), and each model's rate gap:

| prompt | n | cap hits | fp8 | L0.3 | L0.7 | L1.5 | copies spread |
|---|---|---|---|---|---|---|---|
| seq_count30 | 4 | 0 0 0 0 0 0 0 0 0 | +0.00 | +0.00 | +0.00 | +0.00 | 0.00 |
| seq_alphabet | 4 | 0 0 0 0 0 1 0 0 2 | +0.25 * | +0.00 | +0.00 | +0.50 * | 0.00 |
| seq_months | 4 | 0 0 0 0 0 0 0 0 3 | +0.00 | +0.00 | +0.00 | +0.75 * | 0.00 |
| seq_days | 4 | 0 0 0 0 0 0 0 0 3 | +0.00 | +0.00 | +0.00 | +0.75 * | 0.00 |
| seq_words40 | 4 | 2 1 2 0 0 1 1 0 2 | +0.00 | +0.00 | -0.25 | +0.25 | 0.31 |
| bat_trivia | 4 | 0 0 1 1 0 1 0 0 0 | +0.15 | -0.10 | -0.10 | -0.10 | 0.19 |
| bat_choice | 4 | 0 1 0 0 0 0 0 0 3 | -0.05 | -0.05 | -0.05 | +0.70 * | 0.25 |
| txt_story | 4 | 3 2 2 2 2 2 4 4 4 | -0.05 | +0.45 * | +0.45 * | +0.45 * | 0.25 |
| txt_health | 4 | 1 0 3 2 3 2 2 2 4 | +0.05 | +0.05 | +0.05 | +0.55 | 0.56 |
| idn_identity | 4 | 0 0 0 0 0 0 0 0 1 | +0.00 | +0.00 | +0.00 | +0.25 * | 0.00 |
| p1_copycont | 10 | 0 0 0 0 0 0 0 0 8 | +0.00 | +0.00 | +0.00 | +0.80 * | 0.00 |
| p2_binary | 10 | 0 1 0 5 1 1 1 1 6 | -0.04 | -0.04 | -0.04 | +0.46 * | 0.45 |
| p4_nextword | 10 | 3 5 7 1 2 5 4 6 8 | +0.14 | +0.04 | +0.24 | +0.44 * | 0.42 |
| p3_repeat | 10 | 4 0 3 3 1 2 6 9 10 | -0.02 | +0.38 * | +0.68 * | +0.78 * | 0.28 |

- How to read the stars: if the five copies and one other endpoint were exchangeable, the other endpoint would pass by chance about once in six prompts. fp8 passes on length on 4 of 14 prompts, L0.3 on 4, L0.7 on 3. That is chance level on its own; only the rank tests below go further.
- The dose ladder:
  - **L1.5** runs on nearly everywhere. It overruns lists with a natural end (months, days, alphabet, count to 30: 3 of 4 cap hits or a median of +1,000 characters), the batteries, identity and P1. Any of these prompts finds it.
  - **L0.7 and L0.3** stop correctly at the end of a list. They are visible only where the stop is a judgment call: the P3 repetition (cap rate +0.68 and +0.38) and the 200-word story (4 of 4 cap hits each, against 11 of 20 for the copies; 4 repeats only).
  - **No prompt tonight orders L0.3 against L0.7** beyond the spread. On P3 the rates are ordered (6, 9, 10 of 10), but L0.7 and L1.5 are within one answer of each other, and L0.3 against L0.7 is Fisher p 0.30.
  - So the dose reads as two levels, "runs on everywhere" (1.5) and "runs on only at a judgment stop" (0.3, 0.7). Under the scoring each LoRA strength is its own model, so L0.3 against L0.7 remains unseparated.
- Timing (desk, `elapsed_s` in the scored rows, 168 rows at 00:55):
  - Response time is 19.63 to 24.89 s, median 19.80 s, standard deviation 0.67 s, against answers of 59 to 1,271 characters.
  - A linear fit gives a slope of -0.34 s per 1,000 characters (correlation -0.23), and the per-endpoint medians span 19.64 to 19.86 s. That is a server-side pace, not generation time.
  - **The timing idea is dropped.** `analysis/timing.py` was not written.
- A post hoc fp8 candidate: the stop right after the copy in P1. P1 asks for the passage copied, then about 120 more words. The copied passage alone is 437 characters, and it ends mid-sentence on "but". Answers of at most 447 characters (copy-only: stopped at the end of the copy or before it):
  - copies: 01 3, 03 1, 05 0, 06 0, 08 3, so 7 of 50;
  - **fp8 9 of 10** (Fisher against the 50 copies, p 7e-6);
  - L0.3 5 (p 0.021), L0.7 6 (p 0.004), L1.5 0 (p 0.59).
  - On length itself, a rank permutation test against the other copies (4,000 draws) gives fp8 p < 0.00025, L0.7 0.038 and L0.3 0.057. Every copy has p >= 0.07 on the six prompts tested.
- Statistics scanned before the find, so the fresh run is judged against that:
  - 14 prompts × 2 statistics × 4 models = 112 gap comparisons in the tables above;
  - then 6 prompts × 4 models = 24 length rank tests, on the prompts with starred gaps;
  - then the copy-only cut, defined after seeing P1's mass of answers at exactly 437 characters.
  - The length rank test, Bonferroni-corrected over 136, stays below 0.04. The cut's own p is post hoc and not evidence.
- Reading of the P1 candidate: not monotonic in the LoRA strength (5, 6, 0 of 10). It reads as a near-tie stop decision, whether to emit the end of answer right after the copied passage, that any weight change moves in its own direction:
  - fp8 and the lighter LoRAs stop more often;
  - L1.5, which has lost the stop habit, never stops.
  - On its own it cannot tell fp8 from L0.3 and L0.7. Together with P3, where fp8 sits with the copies (2 of 10 against 6 to 10), it would separate fp8 from every LoRA: **copy-only high, P3 cap rate low**.
- Confirmation: `d4_copycont` (P1 verbatim) was added to `probes/night_d.csv` on the desk's order. The copy-only rule was frozen in `stop.py` at 00:53, before any fresh P1 answer existed. The chain runs it at about 01:08, at 10 repeats.
- Other frozen rules for the scored endpoints (desk: a capable model may never reach the cap, so the cap-hit rate could be flat). `stop.py` now also counts:
  - "river" repetitions (P3, d1);
  - words (d2);
  - list items (d3).
  The count of each endpoint is compared with its family's remaining endpoints by a two-sided rank permutation test (20,000 draws, seed 0), using the same iterative setting apart at alpha 0.01. It was frozen at 00:54, before any scored P3 answer existed. On the old P3 answers it sets apart L0.7 (p 0.0007), L0.3 (p 0.0019) and L1.5 (p 0.0031), and keeps fp8 with the copies.

## Status at handoff (2026-11-22 01:15; night_d is running on the references, started by the desk at about 01:08)

**Done in this run (23:30 to 01:15)**
- Round 1 completed to 4 repeats, determinism, separation table, grouping at 40 answers, ablations, permutation tests (E4 to E7). Side channels (E8). night_a P1/P2 (E9): both fail.
- night_b P4/P3 (E11): P4 fails (compliance). P3's stop decision separates the LoRA copies.
- `probes/recon.csv`, `probes/night_c.csv` (E10) and `probes/night_d.csv` written for the desk. night_d holds:
  - `d1_repeat150`: P3 verbatim;
  - `d2_story100`, `d3_openlist`: two stop variants;
  - `d4_copycont`: P1 verbatim.
- `analysis/stop.py`: frozen stop rules, written before the fresh answers.
  - Cap hit (00:47): >= 192 pieces.
  - Copy-only (00:53): answer <= passage + 10 characters.
  - Counts (00:54): repetitions, words and list items, compared with a rank permutation test.
  - Decision: iterative setting apart against the family majority at alpha 0.01, by Fisher for the rates and by the rank test for the counts.
- E12: length and cap hits per prompt.
  - The LoRA dose has two levels: L1.5 runs on everywhere; L0.3 and L0.7 run on only at judgment stops. No prompt orders L0.3 against L0.7.
  - A post hoc fp8 candidate: P1 copy-only, 9 of 10 against 7 of 50. Its confirmation is `d4_copycont`.
  - Response time is a flat server pace (19.8 s), so timing was dropped.
- `analysis/recon_ep.md` and `analysis/recon_ep_families.csv`: the 33 scored endpoints are three families of 11, two Qwen-based (A, B) and one Llama (C). The per-family evidence and the place of each disguised endpoint are in the file. The CSV has one model per family: model MCC 0, never submit it.

**Measured**
- Throughput: 9.1 to 10.0 answers per minute on the references.
- Round 1 at 40 answers per endpoint: MCC 0.000 pooled and with `--mode prompt`.
- Separable from the same-model copies on the references:
  - L1.5: nearly everything;
  - L0.7: P3 cap rate and count;
  - L0.3: P3 count (rank p 0.0019), and the cap rate only against the pooled copies;
  - fp8: only the post hoc P1 copy-only candidate, unconfirmed.

**Next steps**
1. When the desk has pushed `answers/night-d.jsonl`, run `uv run python tasks/model_equality/analysis/stop.py --answers tasks/model_equality/answers/night-d.jsonl --prefix ref- --ids d1_repeat150,d2_story100,d3_openlist,d4_copycont`. The S-majority must keep ref-01/03/05/06/08.
   - Confirmed if d1 sets apart the LoRAs and not fp8, and if d4 sets apart fp8.
   - Judge d4 knowing that 136 statistics were scanned before the P1 find (E12).
2. If confirmed, the desk runs P3 (and d4) on the 33 scored endpoints. Read them per family with `--prefix ep- --families tasks/model_equality/analysis/recon_ep_families.csv`. Use the counts, since these models stop before the cap.
3. Then `night_c` on the references: `choices.py --answers .../night-c.jsonl --draws 500`.
4. `src/euroai/me_query.py` saves `elapsed_s` since 4fa9164. The API returns no token count and `finish_reason` lies, so the cap stays inferred from text.

**Channel**: the desk alone queries. Never append to `answers/round1.jsonl` and never commit the answer files the desk's chain writes.

**Tests**: the desk's fix is merged (4713f83), so `make test` is safe on this branch again.

**Frozen prompt files**: `probes/round1.csv`, `recon.csv`, `night_a.csv`, `night_b.csv`, `night_c.csv`, `night_d.csv`. Never edit any of them once answers exist.

**Tools**: `analysis/group.py`, `permtest.py`, `plurality.py`, `choices.py`, `recon.py`, `stop.py`, `lengths.py`, each with tests under `tasks/model_equality/analysis`.

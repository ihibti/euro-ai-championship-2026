# D1: the source of the task, verified

Primary source read in full (HTML, arXiv v2, 08 Apr 2025): https://arxiv.org/html/2410.20247
Every "Reported number" cell is a verbatim quote or table cell; the URL is always the HTML above plus the section/table/figure named.

| # | Claim | Source (URL and section/table/figure) | Reported number (verbatim) | Our setting | Check on refs | Changes |
|---|---|---|---|---|---|---|
| 1 | Test statistic: squared MMD on prompt-completion pairs; the joint kernel gates on the prompt | §3, Eq. 3 and surrounding text | "the kernel k(z, z') = 1{x = x'} · k̃(y, y')" pools all prompts into one joint test (brief §2, matches paper §3) | - | - | - |
| 2 | Hamming kernel compares completions token by token; shorter sequence right-padded | §3, Eq. 7 | "k̃_hamming(y,y′) = Σ_{i=1..L} 1{y_i = y′_i}, where y shorter than L is right-padded with a special token" | - | - | - |
| 3 | Released code normalises the kernel differently from Eq. 4 (cosine normalisation, zeroed diagonals, division by same-prompt pair count); tests are unaffected because the null uses the same statistic | github.com/i-gao/model-equality-testing (redirects to irenasaracay/model-equality-testing), brief §2 | no number; code differs from Eq. 4 on roughly m× scale | - | - | - |
| 4 | Prompts: uniform over m = 25 random 100-character Wikipedia strings in 5 languages; several completions per prompt | §4 | "The prompt distribution π is a uniform distribution over m=25 random 100-character strings sampled from English, German, Spanish, French, and Russian Wikipedia" | - | - | - |
| 5 | Sample size: N = 10m to 100m per side, i.e. 250 to 2,500 samples per side at m = 25; headline uses ~10 samples per prompt | §4.1 Results; Fig. 2 left | "we vary the number of samples from N=10m to N=100m, where m=25"; "at an average of just 10 samples per prompt ... median power of 77.4% across alternatives" | - | - | - |
| 6 | Completion length: L = 50 tokens (Wikipedia task, token space); L = 250 tokens on HumanEval/UltraChat; API audits run in L = 1000 or 5000 character space | §4; §5 Experiment details; App. C.2 | "The maximum completion length is L=50, and we sample using temperature 1"; "we evaluated tests in L=1000 or L=5000 character space" | - | - | - |
| 7 | Significance level α = 0.05 in all simulations; α = 0.01 in the API audits with a composite null | §4; §5 | "All tests are conducted at a significance level of α=0.05"; "Tests are conducted at level α=0.01 using N=10m samples" | - | - | - |
| 8 | Power at N = 10m = 250 per side, L = 50 tokens, per model (watermark / nf4 / int8 / FPR) | Table 8 | Llama-3 8B: 1.00/0.93/0.79/0.05; Llama-3.1 8B: 0.62/1.00/0.08/0.05; Llama-3 70B: 0.97/1.00/1.00/0.07; Llama-3.1 70B: 0.54/1.00/1.00/0.06; Mistral 7B: 0.72/0.89/0.30/0.04; OLMo 7B: 0.47/0.99/0.36/0.06; Gemma-2 9B: 0.43/0.12/0.06/0.06; Phi-3 Mini: 0.75/0.63/0.24/0.06 | - | - | - |
| 9 | fp16 is the weak case: power "increases slowly" and stays low; numbers appear only in figures, not tables | Fig. 2 right; Fig. 5 caption | "The test is significantly less powerful against the fp16 alternative"; "for this alternative, while other two-sample tests can attain perfect power with enough samples, the kernel test increases power slowly" | - | - | - |
| 10 | Character space (L = 1000) costs power; more samples compensate (Llama-3 8B) | Tables 10 and 11 | N=10m: watermark 0.62, nf4 0.38, int8 0.30, FPR 0.05; N=50m: watermark 1.00, nf4 1.00, int8 0.96, FPR 0.07 | - | - | - |
| 11 | Fine-tuning detected with > 50% power after a single epoch | §4.2, Fig. 3 | "The Hamming MMD test is always able to detect finetuning with nontrivial (greater than 50%) power, even after a single epoch (42 optimization steps)" (Llama-3 8B, lr 1e-6, 1,000-sample datasets incl. out-of-domain code) | - | - | - |
| 12 | Model substitution detected with near-perfect power, even in character space | §4.3; Table 10 | all 13×13 off-diagonal model-swap cells ≥ 0.75, mostly 1.00 at N=10m, L=1000 chars | - | - | - |
| 13 | Temperature mismatch is detected with near-perfect power | App. C.3, Table 12 | τ=0.5 vs τ=1: 1.00/1.00/0.97-1.00 across Llama-3 8B, Llama-3.1 8B, Mistral 8B; τ=1.5 vs τ=1: 1.00/1.00/0.97; "this decoding parameter mismatch can be detected with near-perfect power" | - | - | - |
| 14 | Null distribution: simulated from the reference P (1,000 resamples, Alg. 1) is the main-text method; permutation (Alg. 2) gives the same trends (App. C.7) | §4; App. A.2; App. C.7 | "We estimate p-values by simulating the empirical distribution of the test statistic under the null 1000 times; in Appendix C, we validate that the permutation procedure results in the same trends" (permutation: 100 permutations, 20 simulations, Mistral 7B and Llama-3 8B only) | - | - | - |
| 15 | No local reference needed in principle: P can be earlier samples from the same API | §1 | "To answer if the API is changing over time, P might be from the API at an earlier point in time" — a two-sample test needs samples from both sides, not weights | - | - | - |
| 16 | System prompts: never mentioned anywhere in the paper | full text search | no occurrence of "system prompt"; the only disguise they handle is the composite fp32/fp16 null (Eqs. 10-11) | - | - | - |
| 17 | API audit outcome and cost | §5; Fig. 1; Table 16 | "11 out of these 31 endpoints serve different distributions"; "each audit costing less than $1" (max $0.93); N=10m per test, two tests per endpoint | - | - | - |
| 18 | AgentProv: at ~100 queries per endpoint, rejection of distinct-checkpoint pairs is 60-92% | https://arxiv.org/html/2609.00052 Table 5 (K×N ablation, rejection rate on 630 pairs, α=0.05) | K=10,N=10 (100 queries): 91.90%; K=20,N=5 (100): 65.56%; K=5,N=10 (50): 66.67%; K=10,N=5 (50): 60.48%; K=20,N=50 (1,000): 100%; K=10,N=50 (500): 99.84% | - | - | - |
| 19 | RUT: each trial uses exactly 100 API queries (100 prompts × 1 call), but power is reported as AUC over the mixing fraction q, not a single number at N=100 | https://arxiv.org/html/2506.06975 §5.1-5.2, Table 2 | "Each trial samples 100 prompts. We query each prompt once to the target and 100 times to the reference model." Power AUCs: Gemma 8-bit 0.049, Llama 8-bit 0.132; 4-bit 0.39-0.64 (per research.md §1, sourced to same URL) | - | - | - |

## Against the desk's F3

F3 reads: "int8 0.08 to 1.00, nf4 0.89 to 1.00 (250 answers, 50 tokens); 1,000 chars: int8 0.30 at 10 per prompt, 0.96 at 50 per prompt; fp16 0.00 to 0.04; temperature 0.5 vs 1.0 about 1.00".

- "int8 0.08 to 1.00" — **corrected**: Table 8 (N=10m=250, L=50 tokens, 8 models) ranges 0.06 (Gemma-2 9B) to 1.00. The 0.08 is the Llama-3.1 8B cell, not the range minimum.
- "nf4 0.89 to 1.00" — **corrected**: Table 8 range is 0.12 (Gemma-2 9B) to 1.00. The 0.89 is Mistral 7B; on the five main-text models only, the minimum is 0.89/0.93/1.00/1.00/1.00, so the desk's range holds only if the three appendix models are excluded.
- "250 answers" — **corrected**: N=10m=250 is per side (250 from P and 250 from Q), so one comparison costs 250 API answers only when the local side is free; against another black-box endpoint it costs 250 from each.
- "1,000 chars: int8 0.30 at 10 per prompt, 0.96 at 50 per prompt" — **confirmed** verbatim (Table 11, Llama-3 8B only).
- "fp16 0.00 to 0.04" — **not found as a number**: fp16 power appears only in Figures 2 (right) and 5; the text says it "increases power slowly". The 0.04-0.07 range in Table 8 is the FPR column, not fp16 power. Direction confirmed (fp16 ≈ undetectable), exact numbers unverified.
- "temperature 0.5 vs 1.0 about 1.00" — **confirmed** (Table 12: 1.00, 1.00, 1.00).
- Missing from F3: watermark power at the same budget is 0.43-1.00 across the 8 models of Table 8 (0.54-1.00 on the five main-text models) — the trap variant is detected at roughly nf4-level power, not reliably low.

## Could not verify

- fp16 power numbers (figures only; the HTML text carries no values for the fp16 curves).
- Power at N < 10m: the smallest reported budget is N=10m=250 per side; nothing is reported at 40-100 completions per side. Our 100-answer budget is below every MET operating point.
- Whether the code's cosine normalisation changes power relative to Eq. 4 (code-level difference, no experiment in the paper).
- Any discussion of system prompts, personas, or instruction-tuned disguise variants in MET — absent entirely; the composite null (Eqs. 10-11) covers only fp32/fp16.

## What this changes for us (Devin, 00:20)

Extraction by a helper, checked by me against the source for row 18 (AgentProv Table 5: 60.48 / 91.90 / 98.89 / 99.84% for K = 10 at N = 5 / 10 / 20 / 50; 65.56 / 94.13% for K = 20 at N = 5 / 10).

| Finding | Decision it changes | Check on our data |
|---|---|---|
| Every MET power number is at 250 answers per side or more (rows 5, 8); nothing is published at 40 to 100 | No whole-answer Hamming MMD at alpha on the scored endpoints; supports the desk's change of plan away from round 1 as it stood | done: `noise_round1.md` shows sequences and free text separating disguises, not weights, at 40 answers |
| At a fixed 100 queries, 10 probes x 10 samples reach 91.9% against 65.6% for 20 x 5 (row 18) | Round 2 goes to more repeats of the probes that pass, not to more probes | the night_a refs at 10 repeats: per-probe z at 5 against 10 repeats |
| A temperature change is detected with near-perfect power, and a watermark at 0.43 to 1.00, at 250 per side (rows 8, 13) | A distribution test splits a temperature or watermark copy once it has enough answers. My night-probe statistic compares modes, not frequencies: cross-similarity below the smaller self-similarity | ref-08 and ref-06 against ref-01 on night_a |
| Fine-tuning is detected after one epoch at lr 1e-6, at 250 per side (row 11) | LoRA 0.3 (ref-09) may stay invisible at 10 to 20 answers per probe; the MCC arithmetic then decides that pair, not the test | ref-09 against ref-01 on night_a |

# model_equality: research notes

Our setting: 100 answers per endpoint, at most 256 output tokens, temperature and seed ignored, no system messages, 33 endpoints (528 pairs), and `ref-01..09` free. In the refs, the same-model variants (system prompt, persona, watermark, temperature 1.0) are the traps. fp8 and LoRA at strength 0.3 are the hard "different" cases. The score is 0.9 within-family MCC, so the job is to cluster variants, not to run one test per pair.

## 1. Cited papers

**Gao, Liang, Guestrin, MET (ICLR 2025)** [arXiv 2410.20247](https://arxiv.org/abs/2410.20247)
- Method: a two-sample MMD permutation test on completions. The kernel is the Hamming distance between token strings, and the prompts are random 100-character Wikipedia prefixes in 5 languages.
- Result: median power 77.4% using about 10 samples per prompt over 20–25 prompts. It detects int8, nf4, watermark and fine-tuning, but is weak on fp16 vs fp32. 11 of 31 Llama API endpoints deviated from the reference weights.
- What transfers: a ready baseline that compares two black boxes (no weights needed), and the budget fits about 100 answers. The catch: it flags watermarks, which are "same model" here. AgentProv reports it rejects honest endpoints that use injected system prompts. It therefore needs a threshold calibrated on the ref S-variants, not alpha = 0.05.

**Zhu et al., RUT (ICLR 2026)** [arXiv 2506.06975](https://arxiv.org/abs/2506.06975)
- Method: one API answer per prompt, ranked against 100 local reference samples by their mean token log-rank under the reference model. A Cramér–von Mises test checks the ranks for uniformity.
- Result: beats MMD and KS at a matched budget. 8-bit quantization stays near-undetectable (power AUC 0.049 on Gemma-8bit, 0.132 on Llama-8bit), while 4-bit reaches 0.39–0.64. It also flags temperature and top-p changes.
- What transfers: only if we can run the base checkpoint locally (logits needed). Its sensitivity to decoding changes is a trap for us. The idea of one answer on each of many prompts, ranked against a reference, does carry over.

**Wang, Zhao, Backes, Boenisch, Dziedzic, AgentProv (ICLR 2026, by the task supervisors)** [arXiv 2609.00052](https://arxiv.org/abs/2609.00052)
- Method: prompts offer several functionally equivalent tools, and the probe records which tool the model picks (a categorical answer). An MMD test with a delta kernel compares the choice distributions, with permutation calibration. Tool order is shuffled per sample, and descriptions are length-matched.
- Result: rejects 100% of 630 substituted pairs. False-positive rate under injected system prompts is 7%, vs 67% for MET and 53% for RUT. More samples per template help more than more templates. All three methods accept 8-bit quantization; 4-bit results depend on the model.
- What transfers: this is the strongest hint for our traps. Use categorical answers to near-tie choices, which system prompts barely move, and compare their distributions. We have no `tools` field, so this would have to be emulated in plain text (untested). Expect fp8 to be the hardest pair.

## 2. Newer work (2025–2026)
- [Cai et al. 2504.04715](https://arxiv.org/abs/2504.04715): text classifiers cannot separate quantized from full-precision outputs. MMD has about zero power on INT8 for small models. Greedy outputs differ across serving stacks by more than quantization changes them.
- [IRIS 2607.20860](https://arxiv.org/abs/2607.20860): text-only audit using random-number and random-string prompts (coin, digit, integer, bit string). A cheap pilot sizes the query budget before any real query. It flags 14 of 15 same-model provider pairs for quantization or kernel deviations. Closest to approach A.
- [API-TAIL 2610.36801](https://arxiv.org/abs/2610.36801): absorbs honest temperature, penalty and system-prompt shifts by fitting one effective temperature to the batch, then counts surprisal outliers. Needs a local reference.
- [Stability Monitor 2603.19022](https://arxiv.org/abs/2603.19022): energy distance on response embeddings, summed over a fixed prompt set, with a permutation test. Detected BF16→INT8 and vLLM→Transformers, but uses 800 requests per fingerprint.
- [LLMmap 2407.15847](https://arxiv.org/abs/2407.15847): 8 crafted queries identify 42 LLM versions with over 95% accuracy, robust to system prompts. Good for the family term, not for variants.
- [Idiosyncrasies 2502.12150](https://arxiv.org/abs/2502.12150): fine-tuned embedding classifiers reach 97.1% on 5 model families; the signal is word-level. Family term only.
- [LLMPrint 2509.25448](https://arxiv.org/abs/2509.25448): optimized prompt-injection fingerprints that survive post-training and quantization. Identifies lineage, so useful for family, not for variants.
- [AdaptPrint 2608.22213](https://arxiv.org/abs/2608.22213): adaptive probing (direct, continuation, follow-up) that is robust to system prompts and sampling settings.
- [You've Changed 2504.12335](https://arxiv.org/abs/2504.12335): a statistical test on linguistic features detects model changes. Cheap features worth adding.
- [Two-sample text test 2505.05269](https://arxiv.org/abs/2505.05269): entropy-based two-sample test that needs a language model.
- [GhostPrint 2606.16100](https://arxiv.org/abs/2606.16100): a LoRA can spoof fingerprints under finite query budgets. LoRA effects concentrate on the probes they were trained on.
- [Anosognosia 2610.06174](https://arxiv.org/abs/2610.06174): linear probes find almost no trace of quantization in generated text. A warning that fp8 may be at the noise floor.
- LoRA identification from text: no dedicated method found. MET and RUT detect fine-tuning after 1 epoch. The refs' LoRA strengths 0.3/0.7/1.5 should give distances from S that grow with strength, which is a useful check.

## 3. Public code
| Repo | Licence |
|---|---|
| [irenasaracay/model-equality-testing](https://github.com/irenasaracay/model-equality-testing) (MET; `i-gao/` redirects here) | none detected; treat as all rights reserved |
| [xyzhu123/RUT](https://github.com/xyzhu123/RUT) | MIT |
| [sunblaze-ucb/llm-api-audit](https://github.com/sunblaze-ucb/llm-api-audit) | MIT |
| [pasquini-dario/LLMmap](https://github.com/pasquini-dario/LLMmap) | MIT |
| [locuslab/llm-idiosyncrasies](https://github.com/locuslab/llm-idiosyncrasies) | MIT |
| [TokenRhythm/API-TAIL](https://github.com/TokenRhythm/API-TAIL) | Apache-2.0 |
| [Photen/IRIS-audit](https://github.com/Photen/IRIS-audit) | NOASSERTION (licence file not recognised) |
| [hifi-hyp/ACL-LLMPrint](https://github.com/hifi-hyp/ACL-LLMPrint) | none detected |
| AgentProv | no code link in the paper |

## 4. Ranked approaches
All three send one fixed probe set to every endpoint so all 528 pairs are comparable. They then cluster on the pairwise distance matrix, with the threshold chosen to maximise MCC on the refs. The refs have 36 pairs, 10 of them same-model (the 5 S-variants).

| # | Approach | Effort | Scored answers per endpoint | First experiment on the refs |
|---|---|---|---|---|
| A | **Categorical near-tie probes.** Short prompts with discrete answers (IRIS-style random digits and strings; AgentProv-style "pick one of these equivalent options" in plain text). Per probe, compare answer distributions with a delta-kernel MMD or chi-square, sum over probes, then cluster. Decision rules that depend only on which answer is most likely survive temperature changes, because temperature keeps the ranking of answers. | 6–8 h | about 10 for family (identity and LLMmap-style), about 85 for probes (e.g. 8–10 probes × 8–10 samples, per AgentProv's "more samples per template"), about 5 reserve | Run about 40 candidate probes × 30 samples on ref-01..09. Score each probe by its ratio of distance between S and T1..T4 to distance among the S-variants. Keep the top 8–10. Then repeat with only 10 samples per probe and check MCC on the 36 ref pairs. |
| B | **MET baseline with a calibrated threshold.** Hamming MMD on Wikipedia-prefix continuations and chat prompts (256 tokens), plus embedding energy distance. Set the threshold from the spread among the S-variants (composite null), not alpha = 0.05. | 3–4 h | 100 (e.g. 20 prompts × 5), reusable for C | Reproduce MET on the published `practice_answers.csv` plus about 50 free answers per ref. Measure the distance matrix and check whether watermark, persona or temperature 1.0 lands farther from ref-01 than fp8 (ref-02) or LoRA 0.3 (ref-09). If they do, B alone fails. |
| C | **Local-reference likelihood (RUT / API-TAIL).** Identify the base checkpoint per family and score answers by log-rank. Fit a temperature per endpoint to absorb decoding changes, then compare tail rates. Needs a GPU and the right checkpoint. | 8–12 h | 0 extra if it reuses A and B's long answers | Identify the ref model from its answers. If it is open-weight, score ref answers locally and check that ref-03/05/06/08 stay together after temperature alignment while ref-02/09 separate. |

Recommended order: B first (cheap; tells us whether the traps dominate), then A as the main bet. Spend scored answers only after A's probe set is fixed on the refs. Do not tune to the public leaderboard.

## Could not open or verify
- Some numbers in the arXiv HTML were lost in math rendering: MET's exact completion length and sample counts, AgentProv's template counts and query budget.
- OpenReview pages, any AgentProv code, and the licence file text in the MET and IRIS repos (only GitHub's licence field was checked).
- Papers cited only by others: DuFFin, Zhang et al. 2026b, Gu et al. on prompt caching.
- No dedicated work on picking prompts under a query budget beyond IRIS's pilot sizing.

# Model equality: how we know what we claim

Team Begg, European AI Championship, 22 November 2026.

## The task in three lines

There are 33 chat endpoints. For every pair we must say whether they share a base model (family) and whether they serve the same model. The same model means the same weights, quantization and adapter. A system prompt, a decoding setting or a watermark does not make a different model. The score is 0.1 × MCC on families plus 0.9 × MCC on same-model pairs inside each family. Every query costs from a budget of 100 answers per endpoint.

## Our ground truth: nine reference endpoints

The organizers expose nine reference endpoints of one small base model (SmolLM2-360M-Instruct), with known variants. They cost nothing from the budget. Every method below was tried on them first, and its numbers are given here.

| reference | variant | same model as plain? |
|---|---|---|
| ref-01 | plain | yes |
| ref-03, ref-05 | fixed system prompt; persona prompt drawn per request | yes |
| ref-06, ref-08 | decoding watermark; temperature 1.0 | yes |
| ref-02 | fp8 quantization | no |
| ref-09, ref-04, ref-07 | LoRA adapters at strengths 0.3, 0.7, 1.5 | no |

## The method, in order

**1. Families.** Three families of eleven endpoints:
- A (Qwen2.5-3B-Instruct) writes numbers as digits. Every one of 59 parsed answers to three number questions was in digits.
- B (Qwen2.5-1.5B-Instruct) writes them as words: 53 of 57 answers.
- C (Llama-3.2-1B-Instruct) says it is Llama, and makes errors the other two never make (Sydney for Canberra, 3 as the smallest prime).
- A and B also differ on facts: deepest ocean "Mariana Trench" against "Pacific".

The family submission scored 0.1: the family part was right, and the same-model part scored zero (see Submissions).

**2. Prompted copies, found by asking for the hidden prompt.** We asked each endpoint to repeat everything it was given before our message.
- On the references, the fixed-prompt copy quoted its fixed prompt. The persona copy gave a different persona topic in each of its 3 answers (golf, isotopes, a UI designer). The LoRA copies gave junk or refusals.
- On the scored endpoints, the same two patterns appeared, one copy of each per family:
  - a fixed generic prompt: ep-02 in A, ep-29 in C ("You are a helpful assistant for general questions...") and ep-08 in B;
  - a persona drawn per request: ep-23 in A, ep-20 in B, and ep-16 in C, which gave four different persona prompts in four answers.

**3. A local copy of each base model, and the likelihood of every answer token.** We ran the public base models (Qwen2.5-3B-Instruct, Qwen2.5-1.5B-Instruct and Llama-3.2-1B-Instruct) on a laptop. For C, the 1B model explains the answers better than the 3B: mean log-probability −0.41 against −0.80 on 60 shared answers. Under each one we scored every token an endpoint returned, and fitted the sampling temperature that best explains the answers.

On the references the fitted temperatures were:

| plain | fp8 | watermark | LoRA 0.3 | LoRA 0.7 | LoRA 1.5 | temperature 1.0 copy |
|---|---|---|---|---|---|---|
| 0.722 | 0.741 | 0.721 | 0.717 | 0.780 | 0.871 | 1.007 |

The fit recovers the temperature copy exactly, and it marks the two strong adapters.

A homogeneity test fits the temperature per probe and compares the fits across probes (chi-square). On the references the temperature copy is even at 1.0 (p 0.73). LoRA 0.7 and LoRA 1.5 are shifted and uneven (p 3e-6 and 5e-14). The watermark copy is uneven around the plain temperature (p 3e-5). On the scored endpoints it found one temperature-1.0 copy per family: ep-33 in A, ep-30 in B and ep-31 in C.

**4. Decoder fits.** A copy that truncates sampling never emits a token outside the kept set. We fitted temperature, top-p and top-k jointly and counted the tokens each setting could not have produced.
- In A, ep-14 is fitted by top-p 0.95 at temperature 1.0 with 0 impossible tokens, and it beats the best plain temperature by 23 to 62 in log-likelihood. With top-p 0.90 it has 41 impossible tokens.
- The strong adapters on the references reach the far tail of the distribution, so an adapter does not imitate this.
- B and C each hold one such copy: ep-07 and ep-27.

**5. The watermark signature.** A watermark that keys on the previous token pushes the same choice wherever the context repeats. The watermark copy therefore repeats whole answers, is sharp, and opens repeated stories the same way.
- On the references the watermark copy ranks first on all three. Its stories share a mean opening of 118.5 characters, against 28.9 for plain, 13.6 to 36.4 for the adapters and 1.6 for the temperature copy.
- We registered a prediction before the scored stories existed: the battery candidates ep-04 (A), ep-06 (B) and ep-10 (C) would have the longest story openings in their families.
  - A: ep-04 first, 145.9 characters against 93.0 for the next.
  - C: ep-10 first, 154.8 against 27.1.
  - B: ep-06 second, 83.0 behind 94.8.
  - So 2 of 3 held.
- In B, exact probabilities under the base model settled it.
  - The base model repeats the given story sentence with probability 0.50; ep-06 does so in 8 of 10 answers.
  - ep-06 also restates the trivia questions in 8 of 10 answers, an opening the base model chooses with probability 0.045.
  - A watermark that favours fixed choices does this. The other candidate, ep-22, restates no question and repeats the sentence in 3 of 10.
- Under the likelihood, ep-04's fitted temperature varies across prompts far beyond chance (chi-square 5.2 per degree of freedom, where 1 is expected).

**6. The stop decision.** Adapters change when a model decides to stop.
- A rule frozen before fresh answers existed, "the answer runs to the length cap when asked to repeat a word 150 times", separated LoRA 0.7 and LoRA 1.5 on fresh reference answers. With 20 answers it also caught LoRA 0.3: 12 of 20 against 18 of 100 for the same-model copies (p 0.0003). It left fp8 with plain.
- The exact version on the laptop uses the base model's probability of ending the answer at each step. It separates all three reference adapters, LoRA 0.3 at z = −7.3 on 144 answers.

## What we withdrew, and why that matters

Every rule that decided anything was written down and committed before the data that judged it existed. The record below is our main argument that we did not fit the public leaderboard.

| what | when | what happened |
|---|---|---|
| Distance clustering over the first probe round, 40 answers per endpoint | 21 Nov 23:58 | On the references it merged all nine into one model (same-model MCC 0.000). Withdrawn before any scored use. |
| Identity answers as evidence of changed weights | 22 Nov 02:05 | Five endpoints gave odd identities. We first read them as adapters and held the split back. The prompt-leak answers then showed they were prompted copies, the same model. Correction recorded at 02:15. |
| A registered yes/no split inside family A | 02:28, read 03:00 | Refuted on fresh rows: 14 of 40 against 9 of 32, p 0.62. Not used. |
| An fp8 candidate (stopping right after a copied passage) | 00:55, read 01:42 | Found after the fact on old rows, then tested by a frozen rule on fresh rows. Refuted. |
| Pairwise answer agreement on the references | 03:05 | With only 4 answers per reference, every pair passed, adapters included. The method was used only to describe, never to decide. |

## Limits

- **fp8 against plain.** No probe we have separates them on the references, including the likelihood (fitted temperatures 0.741 against 0.722). Our closest candidate for each family's plain copy is therefore about as likely to be the fp8 copy. Our best candidates are ep-12 (A), ep-32 (B) and ep-19 (C), each at a probability of 0.40 to 0.45. A second closeness test, run independently and validated on the references (where plain and fp8 tie), puts the same endpoint in each family's closest pair. No candidate passes one half, so none is joined to the same-model group.
- **The remaining unlabelled endpoints.** Each family holds five labelled copies of one model and six other endpoints. Among those six are the plain copy and possibly another disguise that we cannot see. Leaving them apart never creates a false "same" pair, but it misses true ones. If one more copy hides in a family, the same-model MCC there is 0.77; if two, 0.60.

## Submissions

| time (Paris) | what | public score | why |
|---|---|---|---|
| 21 Nov 23:36 | every endpoint its own model and family | 0.0 | Checks the path to the server; an all-different answer must score 0. |
| 22 Nov 01:30 | three families, one model per family | 0.1 | Our mistake: we expected more from it. The same-model score is computed only inside families, and there, calling every pair "same" gives an MCC of zero. Only the family part counted. |
| 22 Nov 04:23 | per family, the five labelled copies as one model, the rest alone | 0.7 | Each label rests on at least one method validated on the references. A: 02, 23, 33, 14, 04. B: 08, 20, 30, 07, 06. C: 29, 16, 31, 27, 10. |

The public score covers only part of the pairs. We used it only to check each submission, never to choose between candidate groupings: no grouping was submitted as a trial.

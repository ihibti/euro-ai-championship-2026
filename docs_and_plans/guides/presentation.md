# Presentation for the jury

Team Begg, European AI Championship, Paris, 22 November 2026. Five minutes, eight slides, about 35 seconds each. Every number comes from a committed file or a score posted on GitHub. Sources:
- `docs_and_plans/guides/judges-defense.md` on `main` (the defense pack);
- `tasks/unlearning/judges_brief.md` on `kkrgp/unlearning-first-moves`;
- the description of PR #27;
- the board figures the desk posted on #12 at 10:08.

Each slide: a title, at most three bullets, the one number that matters, and speaker notes to say aloud. Points marked **for the lead** are questions only the lead can answer.

---

## Slide 1. Team Begg: the result

- Public board at 10:01: **9th of 44 overall**.
- Model equality 1.00 (tied first), watermark localisation 0.74 (12th), unlearning 0.29 (18th).
- These are ranks on the public split (about 30% of the data). The final ranking uses the hidden 70%.

**Number: 0.79 overall.**

*Notes.* We are team Begg. On the public board at 10:01 we stood ninth of 44 with 0.79 overall: tied first in model equality, twelfth in watermark localisation, eighteenth in unlearning. The final ranking is on the hidden 70% of the data, so we judged every choice on held-out data, never on the board.

---

## Slide 2. How we worked: one human, a desk and three agents

- The lead set the rules and a night authorization, a file written at a terminal that capped answers and submissions, then slept from 23:30 to 08:05.
- One agent, the desk, alone touched the GPU cluster and the organizers' server. One agent per task did the analysis. All coordination ran through GitHub issues and commits.
- Three rules: references first; a written prediction before each score; one GPU job at a time for the whole team.

**Number: 17 night submissions, each with its reason written before its score** (model equality 5, unlearning 11, watermark 1).

*Notes.* One human set the rules and slept; one agent, the desk, was the only one allowed to submit or run on the cluster; the analysis agents could only propose. Every proposal, prediction and result went through an issue or a commit, so the whole night can be audited. Every deciding rule was committed before the data that judged it; the one decision outside a rule is on slide 4.

---

## Slide 3. Model equality: the method

- Families by base model: three families of eleven endpoints (Qwen2.5-3B, Qwen2.5-1.5B, Llama-3.2-1B), confirmed by the public pairs.
- "Same model" means a named disguise of the base: a system prompt, a temperature, truncated sampling, a watermark. Each was found by a method that recovered the same disguise on the nine free reference endpoints first.
- fp8 against plain: a laptop builds an fp8 copy of the base model the way the serving software does. It then compares, token by token, how well that copy and the plain model explain an endpoint's answers.

**Number: +9.0 standard errors on the fp8 reference** (plain −6.1, the seven others −2.6 to −6.7).

*Notes.* We never called two endpoints the same because they looked similar. Each label is a specific disguise, found by a method that first had to recover that disguise on the published references. The hardest case, fp8 against plain, was separated by simulating the quantization itself: on the references the fp8 endpoint stands out at nine standard errors, and every other endpoint is negative.

---

## Slide 4. Model equality: the path

- Public scores: 0.0 (23:36), 0.1 (families, 01:30), 0.7 (labelled copies, 04:23), 0.85 (06:44), 1.0 (08:07).
- The fp8 rule was committed at 06:03, before any scored number was read (PR #27): +3 standard errors or more names a family's fp8 copy. Family C's endpoint read +3.53, giving 0.85.
- The last step was outside the rule. Family B's candidate read +2.81, under the threshold. In the morning the lead chose to add the one endpoint the test pointed to, and we say so.

**Number: 1.00.**

*Notes.* Four of our five groupings were named by rules fixed before their data. The fifth was a human decision on a reading just under its own threshold: one endpoint, named in advance by the test, where a wrong guess could only lower the score. It is the only submission of the event made that way.

---

## Slide 5. Unlearning: the method

- Gold, the models retrained without the 3,500 images, is unknown. Our stand-in is a cross-fit: each image is predicted by a copy of the last blocks that never saw it, pushed 3 times further in the direction such a model errs.
- Fine-tune the last 6 of 12 blocks towards that target at temperature 2, with a mixup anchor to the original.
- Multiply the last layer by 2: no predicted class changes. Then average the weights of three seeds.

**Number: 0.066 to 0.2133, the same checkpoint with its last layer doubled.**

*Notes.* We had no gold model, no retained data and no labels beyond the model's own predictions, so we built gold's behaviour from images the network had never seen. Training at temperature 2 matches the average of gold models but leaves the network too soft everywhere. Doubling the last layer gives back single-model confidence and removes that drift, and on its own it tripled the score.

---

## Slide 6. Unlearning: the path and its limit

- 0.00 (22:19), 0.066, 0.2133, 0.2837, then **0.2858** (04:39). Each submission had its prediction posted first. Every neighbour of the best point scored lower (0.2222, 0.256, 0.2665), and each said why.
- What failed:
  - the night's score predictor, retired at 04:12 after three misses;
  - the thirteenth batch, which failed two of its four pre-posted conditions;
  - the fourteenth batch's cooler targets, which our morning score model put below the best.
- The ceiling: the morning's score model predicts each high line from the others within 1 to 2.5%, and finds nothing we hold more than about 2% above 0.2858. The board pays only 40 to 52% of what the model's terms give.

**Number: 0.2858.**

*Notes.* Every unlearning submission was a test with a written prediction, and the board was used to reject hypotheses, not to tune. The recipe tops out near 0.29, most likely because gold is an average of several retrained models, which no single sharpened network reproduces. Another day would go to a target that covers the classes several gold models would each pick.

---

## Slide 7. Watermark localisation: c6

- Each image carries one of three invisible watermarks (PixelSeal, TrustMark, StegaStamp) on part of the picture. We return the watermarked area as a mask, scored by mean IoU.
- c6 combines one segmentation network per method, each reading its watermark decoder's own feature maps. Each was trained on four fifths of the training images and scored on the fifth it never saw.
- The organizers' all-zeros baseline scores 0.2518 on the training images. Against c5 on held-out images, c6 gains +0.0104 (90% interval +0.0015 to +0.0201).

**Number: 0.7444 public, 0.7331 held-out.**

*Notes.* The train scores include images the networks learned from, so the number we defend is the held-out fifth. c6's gain over c5 there is small but its interval is above zero, and the public 0.7444 sits close to the held-out 0.7331.

**For the lead** (open questions from the defense pack):
1. Confirm that the networks read the decoders' own feature maps; the watermark work was done with the lead before the night.
2. What "snap3" names in `pixelseal_x1_snap3`.
3. Why c5 has two held-out numbers, 0.7326 in its own record and 0.7227 in c6's paired comparison.

---

## Slide 8. Another day, and what we would not claim

- Another day:
  - model equality: more answers on families A's and B's fp8 candidates;
  - unlearning: a target that covers several gold models' choices;
  - watermark: a crop search for watermarked rectangles, about a quarter of the images, where our networks are weakest (0.68 against 0.82 for StegaStamp).
- We would not claim:
  - that model equality's 1.00 rests only on registered rules;
  - that our unlearning score model predicts beyond the lines it has seen (it missed the line past the peak by 25% without that line's own score).
- Every public rank here is on 30% of the data.

**Number: 70%, the share of the data the final ranking uses, none of which we have seen.**

*Notes.* We know where each method stops, and we wrote it down before the scores came in. The one decision outside our own rules is stated on slide 4. The final ranking is on data none of us has seen, which is why we judged on held-out data all night.

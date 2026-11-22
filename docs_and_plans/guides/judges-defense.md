# Judges' defense pack

Team Begg, European AI Championship, Paris, 22 November 2026. Written during the night for the lead, who slept through it, and through the lead for the jury. Every number here comes from a committed file or a posted score. Points only the lead can settle are marked **for the lead**.

## 1. The night in ten lines

1. At 23:45 the public board had us 20th of 32, with an overall 0.33. The sixth team had 0.73. The overall is the mean over the three tasks of (ours − lowest) / (best − lowest).
2. Now: watermark localisation **0.7444** (submitted 23:36), model equality **0.85** (06:44; 0.7 at 04:23), unlearning **0.2858** (04:39).
3. Against the board's best scores of 22:33, that is an overall of about 0.85. Our current rank is **for the lead**: the agents cannot read the public board; read it before the jury.
4. One human, the team lead, set the rules and the night authorization, a file written at a terminal that caps answers and submissions until 09:00, and then slept.
5. One agent, the desk, was the only one allowed to touch the GPU cluster (JURECA) and the organizers' server. It spent scored answers, submitted, reviewed and merged.
6. One agent per task did the analysis: a model equality main line; a second, independent model equality method by another coding agent (Devin) on a second computer; an unlearning agent. None could submit.
7. Watermark work stopped at candidate c6, the 23:36 submission. The night's GPU time went to unlearning, the analysis time to model equality.
8. All coordination went through GitHub issues (#19 status, #12 unlearning submissions, #21 GPU requests) and committed files. Nothing lived only in a chat.
9. In model equality and unlearning, every rule or prediction behind a submission was committed or posted before the data or score that judged it.
10. The full log is the table in `docs_and_plans/work/doing/night-plan.md`. The lead's checklist is `docs_and_plans/work/doing/morning-brief.md`.

## 2. The three tasks

### Watermark localisation

**Task.** Each image carries one of three invisible watermarks (PixelSeal, TrustMark, StegaStamp) on part of the picture. We return a mask of the watermarked part. The score is the mean per-image IoU on a 128 × 128 grid (`tasks/watermark_localization/README.md`).

**What we did, in order** (records in `tasks/watermark_localization/results/` on the setup branch):

| step | train score | held-out score |
|---|---|---|
| organizers' all-zeros baseline | 0.2518 | |
| whole-image decode: all ones if the full-frame bit accuracy passes a per-method threshold | 0.4529 | |
| c3: a segmentation network for StegaStamp, trained on 4 of 5 folds | 0.5942 | |
| c4a: one network per method | 0.8216 | 0.684 |
| c5 (public 0.7251) | 0.9135 | 0.7326 |
| c6 (public 0.7444, submitted 23:36) | 0.9248 | 0.7331 |

- The networks read the watermark decoders' own feature maps, which the starter kit exposes. "PixelSeal on its own extractor" in the records means this, on the agents' reading. **For the lead** to confirm: the watermark work was done with the lead before the night.
- c6 combines `stegastamp_n1`, `trustmark_n1` and `pixelseal_x1_snap3`. The PixelSeal part is the same network as in c5 (its train score is identical in both records, 0.8738); what "snap3" names is not stated in the committed files. **For the lead.**

**Evidence it is sound.**
- Each network was trained on four fifths of the training images and scored on the fifth it never saw.
- The train scores include images the networks trained on, so they are not evidence. The held-out fifth is.
- c6 against c5 on the held-out images is a paired gain of +0.0104 (90% interval +0.0015 to +0.0201). The public 0.7444 sits close to c6's held-out 0.7331.
- Two held-out numbers exist for c5: 0.7326 in c5's own record and 0.7227 in c6's paired comparison (0.7331 − 0.0104). The committed files do not say which images or settings each used. **For the lead** to explain before the jury asks; the paired gain is the number to defend.

**Limits and another day.** About a quarter of the images are a watermarked rectangle on a clean picture, and the networks are weakest there (0.68 against 0.82 on the other kinds, for StegaStamp). A rectangle decodes almost perfectly from its exact crop. A crop search was planned for Devin (`docs_and_plans/work/doing/watermark-rectangle-search.md`); it was not run: by the night plan watermark stopped at c6, and Devin worked on model equality all night. A second idea, snapping masks to object outlines, was not started.

### Model equality

**Task.** 33 chat endpoints. For every pair, say whether they share a base model and whether they serve the same model: same weights, quantization and adapter. A system prompt, a decoding setting or a watermark does not change the model. Score: 0.1 × MCC on families + 0.9 × MCC on same-model pairs inside families. Each endpoint has a budget of 100 answers.

**What we did**, condensed from `tasks/model_equality/judges_brief.md`, which has every number:
1. **Families.** Three families of eleven: Qwen2.5-3B-Instruct, which writes numbers as digits; Qwen2.5-1.5B-Instruct, which writes them as words; Llama-3.2-1B-Instruct. The public pairs confirm them.
2. **Prompted copies.** We asked each endpoint for its hidden instructions. One copy per family quoted a fixed system prompt, and one gave a different persona each time.
3. **Likelihood under the public base model.** We ran each base model on a laptop and scored every token we had received. That gives each endpoint's sampling temperature, and shows whether the temperature is the same on every prompt.
4. **Decoder fits.** We fitted temperature, top-p and top-k jointly, counting the tokens each setting could not have produced. This finds one truncated-sampling copy per family.
5. **Watermark signature.** Repeated whole answers, and long identical story openings. This finds one watermark copy per family.
6. **Stop decision.** Adapters change when a model decides to end its answer.
7. **A simulated fp8 copy.** The laptop quantizes the base model to fp8 the way the serving software (vLLM) does and compares, token by token, how well that copy and the plain model explain an endpoint's answers. This finds a family's fp8 copy, the one variant no other method saw.

**Evidence it is sound.** All six were run first on nine free reference endpoints whose variants are published: plain, fp8, three LoRA strengths, system prompt, persona, watermark, temperature 1.0.
- The likelihood fit recovers the temperature copy (1.007), and it shows the strong adapters as shifted and uneven (p 3e-6 and 5e-14).
- The watermark copy has the longest story openings (118.5 characters against 28.9 for plain).
- The stop rule caught LoRA 0.3 at 20 answers (p 0.0003).
- A prediction registered before the scored stories existed held in 2 of 3 families. The third was settled by an exact probability under the base model.

**Result.** Per family, five endpoints are labelled as one model: two prompted copies, a temperature copy, a truncated-sampling copy and a watermark copy. The rest are alone. Public 0.7 at 04:23. At 06:44, after the fp8 test named family C's fp8 copy, C's two remaining plain-like endpoints joined its group: public 0.85.

**Limits and another day.**
- **fp8 against plain.** Until 06:00 no probe separated the two on the references. The simulated fp8 copy does: the fp8 reference stands at +9.0 standard errors over 338 answers, the plain one at -6.1, the seven others between -2.6 and -6.7. It held on both halves of the rows and on 112 answers bought after its first run. Two other ways of quantizing do not single out the fp8 reference, so the server's method is identified too.
- **The rule on the scored families was fixed before any of their numbers was read** (PR #27, 06:03, amended with Devin at 06:16): an endpoint at +3 standard errors or more is the family's fp8 copy; the family's two closest plain-like endpoints then join the same-model group, unless one of them is the fp8 copy. A second read after more answers was registered before buying them.
- **Outcome.** C: ep-15 at +2.94, then +3.53 on 72 answers; ep-13 and ep-19 joined; public 0.85. B: ep-22 at +2.71, then +2.81 on 78 answers: under the threshold, so no change, although it is B's only endpoint above zero. A: no endpoint above zero, no change.
- **Earlier attempt, kept as a negative result.** A stop-probability channel: its registered prediction held on fresh reference answers (fp8 above +2), but the gap to plain was about one standard error, too weak to decide a pair.
- With another day we would buy more answers on B's and A's candidates, and find out what the eleventh endpoint of each family is.

### Unlearning

**Task.** We get an image classifier and 3,500 of its training images to forget. The submission is scored against "gold", models retrained without those images: its losses and predictions on those images must match gold's. A control set penalises drift elsewhere, and a model under 70% validation accuracy scores 0. We had no gold model, no retained data and no labels beyond the original model's own predictions.

**What we did**, condensed from `tasks/unlearning/judges_brief.md` on the unlearning branch:
- **Gold, parametrised.** Since gold is unknown, a family of gold hypotheses was built from a cross-fitted prediction: each image predicted by blocks that never saw it.
- **A written prediction per submission.** Every candidate was scored under every hypothesis with the server's formulas, tested in code. Each submission carried a written prediction, posted on #12 before its score. Each score then reweighted the hypotheses.
- **The recipe.** Fine-tune the last 6 of 12 blocks towards that prediction, pushed further in the direction a model that never saw the image would err, softened by a temperature of 2. Then multiply the last layer by 2. That changes no predicted class and restores confidence.
- **The final submission.** The weight average of three seeds of the best recipe (0.2858). The three seeds match on every statistic we measure. They pick different classes on 16% of the forget images, but only between near-tied classes: at temperature 2, where the prediction term is measured, they agree within 0.6%. The average is the lower-variance file of the three, not a different model.

**Evidence it is sound.**
- Each of the first ten night submissions changed one setting and carried a prediction posted before its score. For example, the temperature-only test predicted about 0.008 and scored 0.0057. The eleventh, the seed average, was chosen without a score model.
- Spill probes on held-out images measured drift before anything was queued.
- The last batch of the night (07:12, a target mixed from three independent cross-fits) had four pass conditions posted on #21 before the job ran. It failed two, and the last submission of the night was left unused.
- A simulation of 4,000 random 30% splits shows that the ratio between two of our lines moves by about 1% (one standard deviation). Every gap the night acted on is far outside that: 6% between 0.2665 and 0.2837, 22% to 0.2222. The average's own +0.7% is inside it.

**Result.** From 0.00 at 22:19, through 0.066 (01:38), 0.2133 (02:08) and 0.2837 (02:38), to 0.2858 (04:39). The steps in between scored 0.2222, 0.256 and 0.2665, each lower, and each said why.

**Limits and another day.**
- Gold is modelled by a few parametric families, and the truth may lie outside them.
- The agent's own score models stop at about 5%. It retired its score predictor at 04:12, after the predictor missed three times.
- The ceiling is how far training gets towards the target: the training loss was still falling at 40 epochs. Another day would go to longer training at the same spill.

## 3. Why this is not leaderboard fitting

- **Frozen rules.** Each deciding rule was committed with its time before the data that judged it existed. Examples: the stop rule at 00:47, the batteries at 02:47, the watermark prediction at 03:51, the fp8 check at 04:33, the fp8 rule at 06:03 and its second read at 06:36. Each commit message says whether any of the data already existed and whether it had been read.
- **References first.** No prompt set was spent on scored endpoints before it had run on the nine free references with the same code.
- **Few submissions, each with a reason:**

  | task | at night | before | reason recorded |
  |---|---|---|---|
  | model equality | 4: floor 0.0, families 0.1, labelled copies 0.7, the same plus C's two plain-like endpoints 0.85 | none | yes, in the night log |
  | unlearning | 11: ten with a written prediction on #12, the last a seed average chosen without a score model | 1, at 22:19 (0.00) | yes |
  | watermark | 1 (c6, 0.7444) | 3: 18:11 (0.4421), 19:09 (0.6876), 22:19 (c5, 0.7251) | held-out gain |

  No model equality grouping was submitted to test a candidate against the public score.
- **Ideas withdrawn after failing their own test:**
  - Model equality main line:
    - distance clustering, which merged all nine references at 40 answers;
    - the identity reading, held back at 02:05 and contradicted by the leak answers at 02:15;
    - a registered yes/no split in family A, refuted at 03:00 (p 0.62);
    - an fp8 candidate, refuted at 01:42;
    - a battery agreement method, unvalidated at 4 answers and used only to describe.
  - Desk:
    - the family A split it proposed, which failed the pre-registered test;
    - the 01:30 submission, which rested on a misreading of the metric;
    - a variant of the fp8 simulation that also rounds activations: on the references the fp8 endpoint fell from +9.0 to +0.3, so it was dropped before touching a scored endpoint.
  - Unlearning agent:
    - the "whole classes removed" and "soft gold" readings, both ruled out by their own predictions;
    - a ledger error that treated the accuracy gate as a fraction, found and fixed at 01:13;
    - the line fixed for 03:38, withdrawn before its slot with the reason posted;
    - the score predictor, retired at 04:12 after missing three times.

## 4. The organizers' rules and how we kept them

| rule | how we kept it | source |
|---|---|---|
| one GPU job for the whole team, running or pending | jobs ran one after another, and the desk alone submitted them; the exception was 23:47 to 23:54 (incident below) | night log; morning brief, "What was spent" |
| team check before every job submission | logged as empty before each submission from 00:29 on | night log |
| job submitted from inside the team folder | every night job was submitted by one command that first changes into the team's task folder, stops unless the path is the team folder, and stops unless the team check prints nothing. Open question to the organizers: does a submission from a login node that does not show scratch, pointed at the team folder, count? | morning brief, decision 3 |
| a few hundred files at most on scratch | 209 files staged at 23:45 were removed at 01:35, when an organizer asked teams to free slots; from 02:13 every batch was staged by renaming and emptying old files, with no new file (the staging script counts the folder before and after: 689 and 689 for the tenth and eleventh batches); the batches between 00:03 and 01:16 wrote one checkpoint, one report and one log per run, a few dozen files each, and the output files of the 01:16 batch were created in advance so that a full quota could not break it | night log 01:35, 02:13 |
| 100 answers per scored endpoint | used 64 (A), 66 (B), 48 (C), and 78 and 72 on the two endpoints of the fp8 second read; night cap 80 | morning brief |
| 30 minutes between submissions per task | unlearning 23:37 to 04:39, every 30 to 31 minutes; model equality 23:36, 01:30, 04:23, 06:44; watermark 23:36, 77 minutes after the lead's own c5 submission | morning brief tables |

**Incidents, stated plainly:**
- **23:47, a held follow-up job.** The desk queued a 5-hour job behind the running one, to hold the node for later work. The organizers' text allows chained follow-ups. Our stricter rule allows no pending job and no job without ready work. The lead pointed out the ban risk, and the desk cancelled it at 23:54, seven minutes later.
- **00:22, one stray request.** A test run on the lead's laptop sent one request to the organizers' server with a test key, and the server refused it (HTTP 401). The script's tests had acted on the night authorization. It was fixed in PR #31 at 00:37, with two regression tests. The stray record was never committed.
- **01:30, a misjudged submission.** The desk submitted "one model per family", expecting more than 0.1. The model part is scored only inside families, where "all the same" scores zero. That cost one of the five night slots. It also confirmed the families.
- **00:41, the cluster link dropped.** Login nodes were rebooted during the night. The lead reopened the link at 00:54 and allowed one guarded reconnect attempt per host. No JURECA work was done while the link was down.

## 5. Ten questions a jury would ask

1. **How do you know two endpoints serve the same model?** We do not claim it from similarity. Each label is a specific disguise (prompt, temperature, top-p, watermark), identified by a method that recovered the same disguise on the published references.
2. **Did you tune to the public board?** No. Rules were committed before their data, and no grouping was submitted as a trial. In unlearning, each submission carried a written prediction, and the board was used to reject hypotheses.
3. **Why are so many model equality endpoints left alone?** Calling a pair "same" wrongly costs about as much as missing a true pair. We join only what a validated method names.
4. **Can you tell fp8 from plain?** Yes on the references, and in one family so far. A simulated fp8 copy of the base model explains the fp8 reference's answers better than the plain model does (+9.0 standard errors) and every other reference's answers worse. With the threshold fixed in advance it named C's fp8 copy (+3.53); B's candidate stopped at +2.81 and we did not act on it; A shows none.
5. **Isn't multiplying the last layer by 2 a trick?** It changes no predicted class, so accuracy is untouched. It undoes the softness that the temperature-2 target taught. The factor came from a measurement made before any score.
6. **Why average three unlearning models?** They are seeds of one recipe and match on every statistic we measure. The board cannot rank them: on 30% to 84% of random splits, the public and hidden parts disagree on which seed is better. Averaging removes seed noise without choosing on the public split.
7. **How sound is the watermark score?** It rests on a held-out fifth of the training images, not on the train score. c6's paired gain over c5 is +0.0104, with a 90% interval above zero.
8. **What was your biggest mistake?** In model equality, the 01:30 submission. In unlearning, the ledger's gate error, fixed at 01:13. In the first round, a clustering method, which we withdrew on the references before it touched a scored endpoint.
9. **How did agents work without a human?** Each had one goal, a stop rule and a file to write to. Only the desk could act on the cluster or the server, inside caps the lead wrote at a terminal. Everything is in issues and commits.
10. **What would you do with another day?** Model equality: more answers on B's and A's candidates for the fp8 test. Unlearning: longer training at the same drift. Watermark: the rectangle crop search.

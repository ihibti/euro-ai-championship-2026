# Unlearning: first-principles analysis and the plan that follows

Written 2026-11-21 by the owner's agent. `PLAYBOOK.md` holds the measurements and the working loop; this page holds the reasoning, the bets and the order of the experiments. Every number marked "estimate" is a guess to be replaced by a server measurement.

## 1. What the forget set is

The counts in the playbook fit one construction exactly: **each class has 130 training images** (13,000 in all).

- Nine classes lose all 130 (one image of class 6 is predicted elsewhere by the original model, hence 129).
- 36 classes lose between 35% and 80%: 0.35 × 130 = 45.5 and 0.80 × 130 = 104, and the observed range is 46 to 104. No class sits between 104 and 130.
- 55 classes are untouched.

So gold is trained on 9,500 images of 91 classes. Confidence: high, not certain. The first submission tests it (section 5).

The same counts also fit a cap of 10% on classes of 1,300 images, as in ImageNet. That reading lost its main support on 2026-11-21: 48 forget images, fetched by byte range from the public file, are **generated pictures of objects** (figurines, aloe plants, lanterns, boats, kites). The dataset is custom and synthetic, so a class of 130 generated images is plausible, and clean generated images of distinct objects explain how a ViT trained from scratch on so little passes 70%.

Second fact, read from the confidences: the original model's median top probability is 0.99999. Label smoothing 0.1 caps that near 0.90, and mixup or cutmix cap it too. **The training recipe has hard labels and no smoothing**, and gold is trained the same way. That decides how large gold's loss is on a class it never saw.

Open point: LayerScale at 1e-4 is the DeiT-III default, and DeiT-III trains with a binary cross-entropy. Under that loss a class gold never saw sits at the level of any other negative class, and an unseen-class image may fire no class at all: gold's loss would then be nearer 5 to 10 and its distribution at temperature 2 flatter. The check is on the logits: under a binary cross-entropy the non-top logits share one strongly negative level and their mean is far below zero.

## 2. What gold does, group by group

| Group | Images | Gold's behavior | Gold's loss on the true label (estimate) |
|---|---|---|---|
| Nine removed classes | 1,170 | Never predicts the class. With hard labels the absent class's logit only ever receives downward pushes, so it ends far below the others. The probability goes to present classes that resemble the image. | 15 to 25 per image, spread 2 to 4. With label smoothing it would be 6.9; the recipe says no smoothing. |
| 36 thinned classes | 2,330 | Knows the class from 26 to 84 images. Right on most forget images, wrong on a share that grows with the fraction removed, and overconfident when wrong (an interpolating model without smoothing). | bimodal: near 0 when right, 5 to 10 when wrong; mean 1 to 4 |

Consequences for the three terms:

- **mean_std_score** is decided mostly by the level of the loss on the nine classes. With a third of the images at loss `m` and the rest near 0, the mean is 0.334·m and the spread about 0.47·m: one dial moves both along a line, and gold sits close to that line. Getting the level right is worth roughly 0.8 of this term; the thinned classes hold the rest.
- **prediction_score** is a ratio of KLs that only counts images where gold is not on the true class: all 1,170 images of the nine classes, plus the wrong share of the thinned classes, which can be as many again. Half of this term may sit in the thinned classes.
- **retain** depends on what the control set contains (section 3).

## 3. Hidden assumptions, and which ones can break

| Assumption | Why it may be wrong | What it costs | Test |
|---|---|---|---|
| The nine classes are removed whole | 130 could be a cap on a larger class | The class dial then overshoots and the prediction term collapses: score 0 | Any class-dial submission: a score above 0 confirms it |
| The control set has no image of the nine classes ("the rest of the dataset") | It could be held-out images of all 100 classes | Lowering a class by `d` costs KL of about `d/2` on each control image of that class: at 9% of the control set and d = 30, retain falls to about 0.45. Gold itself would pay this, and so would every team. | The shape of the score along `d`; no direct test |
| Gold passes the 70% gate with room | The validation set may contain the nine classes, where gold and we score nothing | A candidate that only removes the nine classes keeps more accuracy than gold, so it passes whenever gold does. Damage to the other 91 classes is what could sink us. | Held-out forget images: accuracy must stay at 100% |
| The original predictions are the true labels | A few mislabeled or ambiguous images | Negligible: 1 image in 3,500 moves | none needed |
| The public 30% carries over | It is a hash split of the same images | A dial tuned on it carries over; a method picked among many does not | keep to one or two dials |
| The hidden altered copies are mild | They may be strong enough to lower the original model's own confidence | A class-level change carries over to any alteration; a change fitted to pixels does not | compare our altered views |
| Gold's wrong class is the original model's best remaining class | Gold's features are trained without the nine classes; the ranking may differ | Bounded: at temperature 2 the remaining 91 logits are flat, so a wrong order costs about 1 to 2 nats against 5 to 7 for the original | prediction term on the server |

## 4. Second-order effects worth designing around

1. **The class dial is free on forget and maybe expensive on retain.** A bias change acts on every image of the class. If control images of the nine classes exist, the cost grows linearly with `d` while the forget gain saturates: the best `d` is then lower than gold's level. The sweep finds the best `d` either way; do not assume the peak equals gold's loss.
2. **Hedging beats committing on the thinned classes.** We cannot see which of their images gold gets wrong. Flipping a random share confidently gains nothing: a wrong flip costs as much as a missed one. Lowering the class logit by a moderate amount (about 7 to 8 logits) leaves the prediction unchanged but, at temperature 2, moves a third of the probability to the runners-up. Expected gain on the prediction term: +10% to +60% depending on gold's accuracy there; cost: retain falls to about 0.9 if a quarter of the control set is retained images of those classes. The loss at temperature 1 barely moves, so this does nothing for mean_std_score.
3. **The thinned classes' heavy tail is out of reach without a per-image signal.** Matching it needs to know which images gold fails on. Candidates for that signal, all computable from the original model: the 109 images whose prediction changes under an 8-pixel shift, the low-confidence tail (minimum 0.925), and the depth at which a nearest-neighbour probe on intermediate features first gets the image right.
4. **A change in the last layer cannot tell a forget image from a retained image of the same class**, because the features of memorized training images of one class are nearly identical there. Only a change in earlier layers can be image-specific. Whether a fine-tune achieves that is measurable: train on 80% of a class's forget images and read the drift on the other 20%.
5. **The 55 untouched classes are invisible to us.** Every training-based method needs a bound on what moves: few steps, a pull to the original weights, last blocks only. Closed-form edits of the last layer touch them by a computable, tiny amount.

## 5. Submissions as measurements

Each submission returns one number, `mean_std × prediction × retain` on the public 30%. Planned so that each one answers a question:

| # | Candidate | Question | Reading |
|---|---|---|---|
| S1 | nine classes lowered by d = 25 | Are the nine classes removed in gold? | Above 0: yes. This `d` scores under both loss levels (smoothing or not). 0: the classes are capped, not removed, or the gate was hit. |
| S2 | d = 35 (or 18 if S1 is 0) | Where is gold's level? | Higher than S1: gold's loss is 15 or more, go on to 45. Lower: the peak is between 18 and 30. |
| S3 | best `d`, plus the thinned classes lowered by 8 × n_c/130 | Is hedging worth its retain cost? | Compare with the same `d` alone. |
| S4+ | training-based refinements, one at a time | Does image-specific unlearning beat the class-level edit? | Only candidates that beat the edit on the local proxies are submitted. |

All candidates along `d` are built at once and checked with `--check`; the human submits them from JURECA 30 minutes apart.

## 6. Method ladder

1. **Class dial** (`head_edit.py --full d`): training-free, exact effect from `orig_logits.npz`.
2. **Thinned-class hedge** (`head_edit.py --partial D`): same tool, second dial.
3. **Closed-form last-layer edit from features**: a ridge solve gives any wanted logit change per class, not only a lowered logit: for example sharpening or flattening where the nine classes' probability goes. Needs the saved features of the forget images; computed on a laptop.
4. **Soft-target fine-tune** (`csd.py`): starts from the edited model; nine-class images keep their edited outputs as anchors; thinned-class images get image-specific targets. Judged by the held-out drift.
5. **Per-image difficulty** for the thinned classes (shift sensitivity, probe depth): only if step 4 shows that image-specific change is possible.

## 7. Failure modes to watch

- A score of exactly 0: gate, cap hypothesis or overshoot. Never follow a 0 with a larger change.
- Float16 on the server: after a large shift the logits reach about 40; float16 resolves 0.03 there, which is harmless. Keep shifts below 100.
- A fine-tune that looks perfect on the trained images and has drifted on the held-out ones: the held-out drift is the number that counts.
- The public score rising while a local proxy falls: trust the proxy for the choice of method, the server for the dial.

## 8. Outside data

An organizer confirmed on 2026-11-21 that outside datasets may be used. Because the dataset is synthetic and undisclosed, no public set holds the retained images, and real photographs are a different domain. The uses, by value:

1. **None for the first submissions.** The class dial needs no data; do not delay it.
2. **A drift meter and anchor set for the 55 untouched classes.** Run the original model over a large public image set and keep, per class, the images it classifies with high confidence. They are the only view we get of the classes absent from the forget set. Anchoring a fine-tune on them is SCAR's recipe.
3. **A faithful replica, if the first submissions confirm whole-class removal.** Generate images in the same style with a text-to-image model, train original-like and gold-like ViTs on them, and score every method with the exact formula. It also gives a direct estimate of where a model that never saw the nine classes sends their images. This costs hours, so it runs beside the calibration, not before it.

Held-out forget images stay the best stand-in for the control set: same generator, same memorization.

## 9. Who does what

- **Owner's agent (laptop, the only one that talks to JURECA):** candidates, jobs, proxies, records, PRs.
- **Research agents:** three questions feeding `research.md` (issue #3): what a retrained model does when a class is removed, forget-only methods for a ViT, and where this metric and dataset come from.
- **Devin (cloud, public data only):** a replica on CIFAR-100 where gold is trained and known, with the same forget structure, scoring every method with the exact formula. It tells us what cannot be measured here: gold's loss on a removed class, how well the original model's remaining ranking predicts gold, and whether hedging pays.
- **Human:** onboarding steps, the token on JURECA, every submission.

## 10. Literature beyond research.md

`research.md` was written without the data and assumes no whole class is forgotten. Points from our own search that matter once the class structure is known:

- **Masked-teacher distillation, [DELETE](https://arxiv.org/abs/2503.23751).** Forget images only; the target is the original softmax with the forgotten class's logit removed. On ViT-S, retained test accuracy 77.2 against 76.4 for retraining. Our class dial is this target with a finite offset and no training.
- **Where a removed class goes.** Retrained models send it to one to three similar classes, more concentrated than the original's renormalized ranking ([TRW](https://arxiv.org/abs/2506.20893)); class-averaged renormalized outputs still "line up nicely" with 100 retrained models on a small CNN ([Baumhauer et al.](https://arxiv.org/abs/2002.02730)). No per-image number exists.
- **Partly removed classes behave like test images.** With half of a class removed, retraining is right on 47.4% of the removed images against 49.6% on that class's test images ([LTD](https://arxiv.org/abs/2605.31317)); retrained ViTs range from 22% to 95% by sub-class ([JiT](https://arxiv.org/abs/2402.01401)). The class-prior effect is negligible ([logit adjustment](https://arxiv.org/abs/2007.07314)).
- **[AMUN](https://arxiv.org/abs/2503.00917), [code](https://github.com/Ali-E/AMUN).** Fine-tune on each forget image's nearest adversarial example with the class it flips to. Gap to retraining without retain data 1.94, against 4.11 for SalUn, 4.20 for Boundary Shrink and 68.5 for gradient ascent. `adv.py` computes the same adversarial distance as a per-image signal.
- **[JiT](https://arxiv.org/abs/2402.01401)** under-forgets whole classes on a ViT (51.9% where retraining has 0) and over-forgets sub-classes. **[SCAR](https://arxiv.org/abs/2404.12922)** distils on 6–10K outside images in place of retain data; with noise instead, retained accuracy collapses to 16.6%.
- **Two separate targets.** The loss terms compare single gold models; the KL compares to their mean distribution, which is softer than any one of them. The loss at temperature 1 depends only on the label's probability, the KL at temperature 2 on the rest: set them separately, and hedge across the plausible wrong classes.

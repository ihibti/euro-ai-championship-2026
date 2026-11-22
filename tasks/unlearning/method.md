# Unlearning: method, results and limits

Written during the night of 21 to 22 November and the following morning by the unlearning agent. Every number below comes from a committed file or a score posted on #12; the commands at the end reproduce them. The working log is `night.md`.

## Summary

1. **The problem.** Make a ViT-B/16 forget 3,500 of its training images, so that it looks like models retrained without them, while it keeps its behaviour elsewhere. There are no labels, no retained data and no gold model.
2. **Our stand-in for gold.** We kept the first 9 blocks and retrained the last 3 from scratch, in five folds, on the forget images. Each image then gets a prediction from a copy that never saw it: the cross-fit. Extrapolated 3 times, it matches what the scores imply about gold: confidently wrong on a large share of the images, towards sibling classes.
3. **The recipe.**
   - Fine-tune the last 6 blocks towards that target at temperature 2, so the class distribution matches the average of gold models.
   - Keep a mixup anchor to the original.
   - Multiply the last layer by 2. That restores single-model confidence on the forget images and cancels the temperature-like drift on other images.
   - Average the weights of three seeds.
4. **How we chose.** We made twelve submissions from 22:19 to 04:39, each chosen before its score from a written prediction or argument, one change at a time. The board rejected hypotheses: soft gold, whole-class removal, more sharpening, sharpening to the drift's full temperature. No threshold was tuned on it.
5. **Why it is not luck.**
   - Every neighbour of the final point scored lower, by 6 to 25%. The board's own noise is about 1% between two lines (`split_noise.py`).
   - Three seeds of the recipe agree within 0.6% at temperature 2 (`twins.py`).
6. **Limits.**
   - Through the night no model of the score fitted better than about 5% near the peak. The morning's model, with retain read on a stand-in for the unseen control set, predicts each high line from the others within 1 to 2.5%. It still misses the least trained line (k7 x2) by 18%, and the line past the peak by 25% without its own score.
   - Retain is estimated on held-out forget images, not on the unseen control set.
   - The hidden altered copies are stood in for by our own augmentations.
   - Our stand-in for gold retrains only the last 3 blocks. Three independent cross-fits agree on 95% of classes, so it cannot show how gold models disagree. The last experiment, which tested this, was not submitted.

## The result in one paragraph

Our best public score is **0.3019** (`m16_k10s1_x1p95`, 10:48), up from 0 at 22:19, 0.2133 at 02:08, 0.2837 at 02:38, 0.2858 at 04:39 (the night's best) and 0.2938 at 10:18. The 0.2858 is the weight average of three seed runs of one recipe (0.2837 for the first alone). The 0.2938 keeps the two reruns and replaces the 0.2837's run with the same recipe and seed trained 40 epochs instead of 25, so it reaches a little further. It was picked in the morning by a score model that rebuilds every scored line, and it landed 1% above that model's prediction (0.291). The 0.3019 adds a fourth run to those three: the same recipe at twice the learning rate (2e-4), 40 epochs, seed 1, with the last layer multiplied by 1.95. The model predicted 0.2949; with the 0.3019 left out it predicts 0.2900, so this score is a 4% miss in our favour, not a confirmation. Details in the morning section.

**The recipe has two steps.**
1. **Fine-tune.** The last 6 of the 12 blocks are fine-tuned on the forget images (and augmented copies) towards a target built from a cross-fitted held-out prediction. The target is extrapolated 3 times in the direction where the model would err if it had never seen the image, and tempered by 2. A mixup anchor distils the original on blends of two forget images.
2. **Sharpen.** The last layer is multiplied by 2.

**Why it works:**
- **The loss terms compare against single gold models, which are confidently wrong.** The fine-tune makes the right images wrong towards the classes a model that never saw them would pick.
- **The tempered target spreads probability over the sibling classes,** which is what the gold models' average looks like at temperature 2.
- **Sharpening restores per-model confidence and undoes the drift.** It brings back the single models' confidence on the loss side. It also takes back the temperature-like drift that the soft target leaked to unseen images, which is what the retain score penalises.

The 0.2837 is the same recipe trained at learning rate 1e-4 instead of 5e-5; the 0.2858 averages the weights of that run and two reruns with other seeds, then multiplies the last layer by 2. Every submission was chosen before its score, from a prediction or an argument written on #12, one change at a time. Reproducing the 0.2837 takes the commands at the end and about 8 GPU minutes; the 0.2858 adds two reruns with `--seed 1` and `--seed 2` and their average (`soup.py --scale 0.5`). The 0.2938 averages the two reruns with the 40-epoch run (`--epochs 40`, `k9_e3t2_lr1e4_ep40`) the same way.

## The problem in four sentences

We get a ViT-B/16 trained from scratch on a synthetic 100-class image set, and 3,500 of its training images that it must forget. The submission is scored against "gold", several models retrained without those images. The loss mean and spread on the forget images, and on hidden altered copies of them, must match gold's, and so must the predicted class distribution; each term is measured relative to how far the original model is from gold. A control set penalises drift elsewhere, a model under 70% validation accuracy scores 0, and we have no retained data, no labels beyond the original model's own predictions, and no gold model.

## What the metric rewards, and what it does not

- **Matching, not forgetting.** Both forget terms are 1 when the submission behaves like gold and 0 when it behaves like the original. Going past gold is clipped to 0, just like not moving. Maximal forgetting (gradient ascent, random labels) scores 0.
- **Near the original, the score is roughly quadratic.** A candidate that moves a fraction t of the way scores about t squared on the forget side, so timid candidates score near 0.
- **The terms multiply.** Forget × retain, and within forget, mean/spread × prediction: one weak term sinks the product.
- **Retain is a drift penalty.** Even with accuracy intact, the KL between the original's and the submission's outputs on the control set divides the score: retain = 1 / (1 + KL).
- **The loss terms see single gold models, the KL term sees their average** (README, Metric). A good submission is confident in loss, like each gold model, and still spreads probability over the classes where the gold models disagree.
- **The metric reads as a population version of the likelihood-ratio membership test** (LiRA): if the forget set's loss mean and spread match gold's, a Gaussian test on the population cannot tell the submission from gold (`research.md`, row 21).

## How we worked through the night

Three agents coordinated on GitHub issues while the humans slept:
- **the desk**, the lead's agent, the only one that touches the cluster and submits;
- **the model-equality agent**;
- **the unlearning agent**, which had a laptop copy of the model and the forget images and no cluster access. It asked for GPU runs in batches (#21) and for submissions through a queue (`submit_queue.md`, #12). Each queue line states, before the submission, what its score will tell.

The method is a **pre-registered hypothesis search**:

1. **Gold is unknown, so it is parametrised.** A family of gold hypotheses is built from the original model's logits and a cross-fitted held-out prediction (`xfit.py`). The last blocks are retrained on the other forget images, so each image's prediction comes from blocks that never saw it. Two parameters:
   - K: how often and how far gold is wrong;
   - tau_g: how soft it is, which also stands for averaging several gold models.

   Families:
   - gold errs where our cross-fit says (`aligned`);
   - gold errs on other images of the same classes (`other`);
   - gold never saw the nine classes that lost all their training images (`removal`).
2. **Every candidate is scored under every hypothesis** with the server's formulas: `scoring.py`, tested in `test_scoring.py`, and `ledger.py`.
3. **The utility gate is part of the inference.** The original's validation accuracy A is unknown. A candidate that changes a share f of the validation predictions passes when A × (1 − f) ≥ 70%, and scores 0 otherwise. A and gold are inferred together: a 0 can be a gate failure, and a positive score bounds A from below.
4. **Each public score weighs the hypotheses** by how well they predicted it. The next submission is the one with the best predicted gain under the hypotheses that survive, or the one that best separates them.

This keeps the number of submissions small and makes each one an experiment with a written prediction (#12), not a lottery ticket. It also follows the lead's rule against fitting the public board: one dial at a time, and a family is accepted only when its neighbours agree.

## Submissions and what each one taught

| Time | Candidate | What it is | Prediction written before | Written | Public score | What it taught |
|---|---|---|---|---|---|---|
| 22:19 | `bias_full30` | nine classes' outputs lowered by 30 everywhere | positive under the reading that whole classes were removed | #12, 19:42 | 0.00 | that reading is wrong, or the gate failed (it loses the nine classes on the validation set) |
| 23:37 | `x3_ext_m80` | fine-tune of the last 3 blocks towards a blend of the original and the cross-fitted prediction | above 0 if the cross-fit's soft targets move towards gold; exactly 0 if their direction is wrong or the gate was hit (no number written) | queue `0ddc834`, 23:31 | 0.0063 | first positive score, far smaller than the golds the dial assumed |
| 00:07 | `x4_e20` | the same, towards the cross-fitted step extrapolated 2 times | much above 0.0063 if the dial points the right way | queue `1bb1b53`, 00:05 | 0.0336 | the dial points the right way: five times the previous score |
| 00:37 | `scale_t4` | every logit divided by 4 (last layer / 4), no training | about 0.008 (0.006 to 0.013) under the golds that fit the scores so far; 0.1 or more under soft golds | #12, 00:19 (`ledger.py` at `81feaee`) | 0.0057 | just under the range written for the confidently wrong golds, and a twentieth of the soft golds' 0.1: soft golds, and with them a temperature as the answer, are out. Our acknowledgement on #12 at 00:43 misquoted the range as 0.004 to 0.013 |
| 01:08 | `dial15` | the nine classes' bias lowered by 15, last layer only | weighted mean 0.015: 0 under the confidently wrong golds; 0.076 under removal with the gate; 0.29 under removal with no nine-class validation images | #12, 00:57 | 0.0 | removal survives only if A is between 73.4 and 75.4: 4% of the weight |
| 01:38 | `k7_e3t2_rm` | the last 6 blocks fine-tuned towards the cross-fitted step extrapolated 3 times and tempered by 2, with an anchor | mean 0.047, 89% chance above 0.0336; 0 if it fails the gate (it changes about 6% of validation predictions) | queue `fe014e7`, 01:34 | **0.066** | twice the previous best. It passed the gate, so A × 0.94 ≥ 70. The score sits at the upper end of the prediction: the ledger underpredicts this fine-tune by about 1.3 times |
| 02:08 | `k7_e3t2_rm_x2` | the same model with its last layer multiplied by 2 | mean 0.086, 10th to 90th 0.060 to 0.118, 75% chance above 0.066 (0.106 with the exact logits, logged at 02:06 but committed only after the score) | queue `e9eb29b`, 01:57 | **0.2133** | sharpening is the recipe: the same checkpoint scores more than three times as much once the spill's temperature is taken out. The ledger gets the direction right and the size wrong: the gain is about twice what it predicts |
| 02:38 | `k8_e3t2_rm1_lr1e4_x2` | one dial from the 0.2133: the same recipe trained at learning rate 1e-4, sharpened by 2 | ledger mean 0.229, 10th to 90th 0.142 to 0.301, 66% chance above 0.2133; 1.7 times `k7_e3t2_rm_x2` relative to the ledger | queue `7dde708`, 02:24 | **0.2837** | more reach pays at the same factor (1.33 times). Golds that err on other images of the same classes drop to 2% of the weight |
| 03:08 | `k8_e3t2_rm1_lr1e4_x2p5` | the same fine-tune with the factor one notch further, 2.5 | ledger mean 0.283, 10th to 90th 0.211 to 0.373, 54% chance above 0.2837 | #12, 02:45 | 0.2222 | **one notch more sharpening cost 22%**: the forget-set loss has a peak near the 0.2837's level. The ledger cannot represent it and is set aside for the factor (section below) |
| 03:38 | `k9_e3t1_lr1e4_x1p25` | the ninth batch's untempered (tau 1) fine-tune, last layer times 1.25 | `peak_check.py` mean 0.294, 0.281 to 0.316 over the 13 gold settings that reproduce both scores. It replaces the line fixed at 02:56, which the same check puts at 0.185 | queue `36df4a9`, 03:18 | 0.256 | **at the 0.2837's forget-loss level, half the probe spill scored 10% less.** The factor left a third of the drift's temperature (f/T = 0.76): on unseen images, sharpening must match the drift's temperature. The probes' lower spill does not carry over by itself |
| 04:09 | `k8_e3t2_rm1_ep40_x2p25` | the eighth batch's 40-epoch run, last layer times 2.25: exactly its measured spill temperature | `peak_check.py` 0.292 to 0.294 over three readings of retain; written down as a test, not a likely new best | queue `c13cd3c`, 03:50 | 0.2665 | its forget loss, probes and spill temperature match the 0.2837's base; only the factor relative to the temperature differs (1.00 against 0.89), and it scores 6% lower. Every neighbour of the 0.2837's point has now scored lower. With all five scores the check fits no setting and is retired as a predictor |
| 04:39 | `k11_b_soup3_x2` | the weight average of the 0.2837's base and its two seed twins (same recipe, seeds 1 and 2), last layer times 2 | no score model; written before the score: above 0.2837 averaging pays, about 0.28 the 0.2837 is a typical draw, clearly below a lucky draw | queue `b962a1d`, 04:28 | **0.2858** | a typical draw: +0.0022 on the server's unrounded scores (0.8%), inside the board's 1% noise between two lines. Our final line, for its lower seed noise on the hidden 70% |

Times, scores and sha256 values are those of the server's submission records in `tasks/unlearning/submissions/` on `main` (`attempted_at` in Paris time; every queued file's sha256 matches its queue line, and `bias_full30` predates the queue). "Written" is where the prediction stands before the submission: the queue's last commit before the slot (`git show <commit>:tasks/unlearning/submit_queue.md`), or the time of the comment on #12.

## What the evidence says about gold (as of 02:15)

- **After the 0.2133, two thirds of the weight sit on golds that err where our cross-fit says.** Golds that err on other images of the same classes fall to 11%: they cannot produce a gain of more than about 1.4 to 1.7 times from sharpening (1.0 to 1.2 on the forget side, times 1.38 from the retain model), and we observed 3.2 times (`night.md`, 02:15).
- **The ledger underpredicts sharpened fine-tunes by about 1.6 times.** Under the gold that fits best once both scores are in (aligned, K=3, tau 1; fitted at 02:15, after both), the unsharpened score is reproduced (0.063 against 0.066) and the sharpened one comes out at 0.162 against 0.2133. The gap fits a retain near 0.9 after sharpening, that is, a drift on unseen images that is almost purely a temperature. The probes, on training images of the original, also show changed predictions that the real control set apparently does not.

Before the 0.2133, as of 01:45:

- **Gold is confidently wrong on most of the forget images.**
  - Until 01:40 the weight sat on golds right on about a third of the images, with loss mean 7.6 to 8.1 and spread 8.8 to 9.1. That was the edge of the grid.
  - With the grid widened to golds right on only 10%, the weight spreads evenly over golds right on 10 to 33% of the images, with loss mean 8 to 15 (`ledger.py`, `37cd60c`). Our scores so far cannot tell where in that range gold is.
  - Soft golds, which would make one temperature the right answer, predicted `x3_ext_m80` at ten times its score and `scale_t4` at 0.1 or more.
- **This matches published retrains when a large share of a class is deleted.** Dolgova and Stich (CISPA, 2026) report, on CIFAR-100, that a retrain is right on 47% of a class's deleted images when half of the class is deleted, and on 17% at 90%, against 72% test accuracy (`research.md`, row 22). Our 36 thinned classes lost 35 to 80% of their images. The README says the same in words: the target covers "which wrong class it picks for each one".
- **The wrong class is a sibling.** The original and gold are ViT-B/16 trained from scratch (the organizers' template: `pretrained=False`, LayerScale), which generalise poorly from 130 images per class. The forget images concentrate on look-alike classes: labels 29, 30, 31 and 33 are four kinds of small boat on misty water, each among the nine classes that lost all 130 images, next to 32 (fishing boats, 104 images) and 23 (bridges over water). Our cross-fit errs 19→21, 43↔42, 2→0, 35→38; the golds that fit go about three times further in that direction.
- **The original's validation accuracy A is about 79%** (10th to 90th percentile 74 to 84), inferred from the five scores together.

## The ledger at the end of the night

`ledger.py` with all twelve scores and every dumped logit file (command under Reproduce; about 2 minutes on the laptop) says:
- **Gold errs where our cross-fit says (`aligned`): 75% of the weight.** The golds that carry it extrapolate the cross-fit's step 3 to 6 times and are right on only 10 to 33% of the forget images.
- **Whole classes removed: 25%** (`removal` 17%, `removal_b` 8%).
- **Gold errs on other images of the same classes (`other`): 0%.**
- **The original's validation accuracy A:** 79.4 on average, 75.0 to 84.0 (10th to 90th percentile).
- **It still ranks the factor 2.5 above 2 on the 0.2837's base** (0.274 against 0.241), which the board refuted (0.2222 against 0.2837). Its golds are single models, so it cannot represent the peak. We use it for the reading of gold, not for the factor (Honest limits).

## The morning: a score model that fits every scored line (09:00 to 09:14)

**Why a new model.** The end-of-night ledger skipped three of our best scores without saying so: k9 x1.25, ep40 x2.25 and the soup's 0.2858. It built sharpened candidates only at factors 1.5, 2, 2.5 and 3, and none for a weight average. It also never had the desk's 0.2687 (08:21). And its golds tie the loss statistics to the class distribution, so it could not show the peak.

**What `score_fit.py` does** (`434e59e`, `ca5576d`, 7 tests). It rebuilds every scored line with dumped logits exactly: 11 of the 15 scores, all but the dials and the owner's first file. A line is a run, or the logit average of a soup, times its factor. It then computes the scorer's terms as the server does, on both groups (the exact images, and our jitter view standing for the altered copies):
- the prediction term under each gold shape of the ledger;
- the loss term against a free gold (mean, std);
- retain's response to the factor, read on a stand-in for the control set.

The control set is unseen images, where the original is less sure than on its training images (where the probes read the drift). The stand-in is the cross-fit's held-out logits, softened by 1.25. One free level c scales everything. Each hypothesis is weighed by a 10% log-normal error.

**How well it fits.** Log-likelihood −8.0, against −14.8 with the probes' sharpening estimate and −43 with c fixed at 1. Leave-one-out, each line predicted from the other ten:

| line | public | predicted without it | 10th to 90th | error |
|---|---|---|---|---|
| soup3 x2 | 0.2858 | 0.2815 | 0.262 to 0.302 | −1.5% |
| base x2 | 0.2837 | 0.2773 | 0.258 to 0.298 | −2.3% |
| ep40 x2 | 0.2687 | 0.2649 | 0.247 to 0.284 | −1.4% |
| ep40 x2.25 | 0.2665 | 0.2690 | 0.244 to 0.295 | +0.9% |
| k9 tau 1 x1.25 | 0.256 | 0.2535 | 0.229 to 0.277 | −1.0% |
| base x2.5 | 0.2222 | 0.2787 | 0.228 to 0.342 | +25% |
| k7 x2 | 0.2133 | 0.1748 | 0.163 to 0.188 | −18% |

The two misses sit at the edges. x2.5 is the only line past the peak; with it in the data, the model reproduces it. k7 x2 is the least trained line.

**What it says:**
1. **Gold's forget loss is high:** mean 3.0 to 5.6, std 2.9 to 5.0 (10th to 90th), near 4.2/3.8. Single gold models are confidently wrong on many forget images, as the night concluded.
2. **Retain is not a lever once the factor is right.** On the stand-in, the temperature alone reaches each probe's KL; the fitted noise is 0. So retain depends only on f/T, the factor over the run's spill temperature: it peaks at 1.0, costs 1.4% at 0.9 and 10% at 0.76. A run with half the spill (tau 1) needs a lower factor and gains nothing once sharpened.
3. **This explains the 0.256 without any cost to tau 1.** It was sharpened to f/T 0.76 and lost 8 to 10% of retain there. Sharpened to 0.88, tau-1 runs still lose to tau 2 on the prediction term (0.83 to 0.86 against 0.87).
4. **The board pays 40 to 52% of what the model's terms give (c).** `x3_ext_m80` barely changes the model (retain 0.98), and it too falls below the fit. So the missing factor is on the forget side, most likely the prediction term.
   - Gold's class distribution is the average of several retrained models, further from any single sharpened network than our gold shapes.
   - Two other readings were checked and do not carry it.
     - **Altered copies:** the base run's forgetting carries over evenly to all nine of our alterations (loss mean 2.00 to 2.23, label probability 0.196 to 0.221, KL from the original 1.89 to 1.98, in its report).
     - **Golds that disagree:** averaging an aligned and an other gold's distributions (`--mixtures`) changes nothing with c free. With c fixed at 1 it fits better than single golds (log-likelihood −33 against −43), but far worse than a free level (−8).
   - So the level is not explained by any form we tried. Anchoring every prediction on a scored line cancels it.
   - This is the largest open lever. A next attempt would target a model whose distribution on the forget images covers the classes that several gold models would each pick, while keeping the loss of a single one.
5. **Nothing we hold beats 0.2858 by more than the noise.** The best of 920 logit blends of two or three runs is 1.8% above it under the model (`blend_search.py`, `1b6456a`): `k11_b_s1` + `k11_b_s2` + `k9_e3t2_lr1e4_ep40` at x2. Picking the best of 7,360 correlated lines gives about that much by chance.

**Predictions written before the morning's scores** (#12 and `submit_queue.md`):

| file | written | prediction | public |
|---|---|---|---|
| `m14_t1_blend3_x1p5.pt` (tau-1 runs 3:1, x1.5) | 08:32, two-condition rule | 0.29 to 0.31 | not submitted |
| the same | 09:01, model with the probes' retain | about 0.27 (0.26 to 0.30), 1 in 4 above 0.2858 | not submitted |
| the same | 09:09, model with the stand-in's retain | 0.267 to 0.272, 1 in 10 above 0.2858 | not submitted |
| `m15_k9t2ep40_x1p85.pt` (the ninth batch's 40-epoch tau-2 run, x1.85) | 09:09, model | 0.2855 to 0.2874 | no score: the 09:45 upload timed out on the client |
| `m15_s1s2ep40_x2.pt` (weight average of the 0.2858's two seed runs and that run, x2) | 09:46 and 09:57, model | 0.291, 64% above 0.2858 | **0.2938, our best** (10:18) |
| `m16_k10s1_x1p95.pt` (the 0.2938's three runs plus the same recipe at twice the learning rate, 40 epochs, seed 1; x1.95) | 10:27, model anchored on 0.2938 | 0.2949, 54% above 0.2938 | **0.3019, our best** (10:48) |
| the 10:46 line, by the rule of 09:57 as changed at 10:07 | 09:57, 10:07 | the spare `m15_k9t2ep40_x1p85` if 10:16 scores 0.2866 or above (gain 0.0022 to 0.0046 over the blend's score); otherwise `k14_t1_lr9e5_x1p47` (two-condition rule 0.29 to 0.31, model 0.272) | the spare was confirmed, then replaced at 10:27 by the four-run blend above, after a refit with the 0.2938 |
| `k10_t2_lr2e4_ep40_s1_x2.pt` (the 0.3019's fourth run alone, x2) | 10:32 rule, confirmed at 11:00 after the 0.3019 | mean 0.257, 20% above 0.3019 (wide: the model has no score of this run alone) | 0.2939 (11:20) |
| `m17_three40_x2.pt` (the three 40-epoch runs: lr 1e-4 seed 0, lr 2e-4 seeds 0 and 1; x2), by the rule of 11:00 | 11:00 rule; confirmed at 11:24 after the 0.2939 | mean 0.3005 on shared runs, 60% above 0.3019, 44% above +1% | pending (11:50) |

**What happened to the morning's plan.** At 09:44 the lead confirmed that only a team's best submission counts. The slots left before 11:00 therefore went to lines that could beat 0.2858, not to a test of the tau-1 blend, which the model had put below it. The first, the ninth batch's 40-epoch run at x1.85, would also have been the model's first test on an unscored recipe. Its upload timed out at 09:45 on the client, the attempt counted for the 30-minute gate, and two slots were left.

**The fourteenth batch (tau 1 and tau 1.5 at the operating point, 08:48).** All four runs pass the two conditions set on #21 at 08:26: the loss factor is 0.93 to 1.06 times 0.89 T, and 4.7 to 5.1% of predictions change. The model puts every one below 0.2858 (0.266 to 0.280 at their best factors). Its reason is the one it gives for the 0.256: once sharpened to the right factor, spill is no lever, and the cooler targets lose on the prediction term. The batch is where the two-condition rule and the model disagree most.

**The last two slots: expected gain over the best (09:57).** With one slot after the next and only the best counting, the mean is the wrong criterion. A plan is worth the mean, over the model's draws, of max(0, its best line's score anchored on 0.2858, minus 0.2858) (`score_fit.py --draws-out`, `slot_gain.py`, 3 tests).
- The blend `m15_s1s2ep40_x2` alone: 0.0060.
- With `k9_e35t2_lr1e4` at x1.95: 0.0097. That run's target is extrapolated 3.5 times instead of 3, and its forget loss is 4.4 against the blend's 4.0. It pays when gold's forget loss sits high, which is when the blend also scores high.
- With any fourteenth-batch line: still 0.0060. Inside the model, whenever such a line beats 0.2858, the blend beats it by more.

So the 10:46 slot depends on the 10:16 score, by a rule written before it.
- **High score:** the blend held, so take a line that reaches further. At 09:57 that was `k9_e35t2`. At 10:07, still before the score, it became the spare (the ninth batch's 40-epoch run at x1.85), for the gate. `k9_e35t2` changes an estimated 7.0% of validation predictions, outside the 6.0% that has passed. If the gate is taken per split, a line can pass on the public 30% and fail on the hidden 70%. If it is also the best public line, the task scores 0. That tail costs about as much as its edge in the model, and every submission must be one we would accept as final.
- **Low score:** inside the model no line has a real expected gain. Take the two-condition rule's line, whose upside lies where the model is wrong.

**The weight-average caveat, measured.** The blend averages runs of 25 and 40 epochs, and the model stands it in with the average of their logits. The only real soup with dumped logits is the thirteenth batch's seed twins: its forget loss sits 0.8% below its logit average's, with a KL of 0.0004 at T=2. The blend's members sit 1.3 times further apart than seed twins (logit RMS 0.53 against 0.42). The second-order gap grows with the square of that distance, so we expect about 2% in forget loss and under 1% in score.

**The fourth run (10:48), and what the last two slots test.** The 0.3019 averages the 0.2938's three runs with a fourth: the same recipe at learning rate 2e-4, 40 epochs, seed 1 (`k10_t2_lr2e4_ep40_s1`), last layer x1.95. The model predicted 0.2949; with the 0.3019 left out it predicts 0.2900, a 4% miss in our favour. Two readings fit, and the last two slots were set to tell them apart, by rules written before each score (#12 at 10:32, 11:00 and 11:04).
- **The fourth run is better than the model thinks.** Then it scores near 0.30 alone (11:18), and the soup with the most lr 2e-4 runs, `m17_three40_x2`, comes next (11:48).
- **The gain comes from averaging in a run further from the others.** Then the run alone scores lower, and the wider average `m17_five_x2` comes next. The soups' leave-one-out residuals are consistent with this: +0.4% for the seed twins (logit RMS 0.42 apart), +0.9% for the 0.2938 (0.53), +4.1% for the 0.3019, whose fourth run sits 0.60 to 0.72 from the others. Three points, against about 1% of board noise between two lines.

The 11:48 cut is 0.2900, where the two follow-ups cross in refits with six possible 11:18 scores. **Outcome:** the fourth run alone scored 0.2939, against 0.2486 predicted for it with its own score left out. The first reading holds: the run itself is much better than the model thought, and on its own it equals the three-run average of 10:18. The four-run average added 2.7% on top. So the last slot went to `m17_three40_x2` (mean 0.3005, 60% to beat 0.3019, 44% by more than 1%). Without an 11:18 score, 11:48 is `m17_three40_x2`: `m17_five_x2` would win only inside the noise (35% to beat 0.3019, 4% to beat it by more than 1%).

**Anchoring on shared runs (10:55).** The model's level is pinned by observed scores. Anchoring every candidate on the best line alone inflates a candidate when the best is a single run unlike it: in hypothetical refits it gave gains up to 0.013, against at most 0.004 otherwise. From 10:55 each candidate is anchored on the scored lines that share runs with it, weighted by the weight they share (`slot_gain.py --shared`, 4 tests). A line that shares no run keeps the best line's anchor.

## The utility gate, and localisation

A submission under 70% accuracy on the hidden validation set scores 0. Inferred A leaves a margin: a candidate may change about 5 to 17% of validation predictions (1 − 70/A for A from 74 to 84), depending on where A really is.

- Dividing all logits by a temperature never changes the predicted class, so it is gate-safe by construction. That is why `scale_t4` was a safe test.
- Fine-tunes must keep their changes on the forget images and their slight alterations. The spill probes measure this before anything is queued, on held-out forget images and on whole classes the fine-tune never saw.
- **The seventh batch's probes show the hard part.** A mixup anchor (distilling the original on blends of two forget images) slows the forget images and leaves the spill unchanged: held-out KL 1.1 to 1.2 and spare-class KL 0.74 to 0.79 at anchor weights 0, 1 and 3. The fine-tunes also reach only about 40% of their target's loss (1.80 against 4.38). The eighth batch (`run_eighth.sh`, `5ac28b2`) therefore tries further and sharper targets and stronger training at the best anchor setting.

## Sharpening: taking the temperature out of the spill

The seventh batch's probes showed that most of a fine-tune's drift on classes it never trained on behaves like one global temperature:
- the spare classes' label probability at T=2 is 0.52, with KL 0.74 to the original;
- the original divided by 2.25 gives 0.57 and 0.53.

Multiplying the last layer by a factor s undoes a temperature. It also never changes a predicted class, so the gate is exactly the fine-tune's own. The dumped logits show two effects:

- **Forget side:**
  - on the forget images, sharpening raises the loss of the images the fine-tune made wrong, which moves the loss mean and spread towards gold's confident errors;
  - under the golds that err where our cross-fit says (tau 1), `k7_e3t2_rm`'s forget term roughly doubles at s = 2 (K=3: 0.127 to 0.238; `night.md`, 02:15).
- **Retain side:**
  - on images whose prediction the fine-tune kept, sharpening divides the KL to the original (1.36 to 0.50 at s = 2);
  - on images whose prediction it changed, the KL stays put (2.11 to 1.99);
  - the ledger mixes the two at each probe's measured share of changed predictions (`sharpened_retain`, `ledger.py`, tested in `test_ledger.py`). `k7_e3t2_rm3`'s retain goes from 0.48 to 0.67 at s = 2.
- **Where it stops paying:**
  - over the golds that fit our scores, the ledger peaks at s = 2 to 2.5 (`k7_e3t2_rm`: 0.086 at 2, 0.093 at 2.5, 0.089 at 3, 0.078 at 4);
  - golds that err where our cross-fit says keep rising with s, while golds that err on other images of the same classes fall to 0 at s = 4;
  - `x4_e20`'s targets were never tempered (tau 1), so it has no temperature for sharpening to remove.

We queued `k7_e3t2_rm` and then the same model at s = 2, so the two scores isolate this one change.

## The peak at 03:08: one notch more sharpening cost 22%

The rule fixed before 03:08 assumed the ledger's view of sharpening: more sharpening keeps paying up to a factor of 3 (#12 at 02:21; on the 0.2837's base, 0.283 at x2.5 and 0.296 at x3 in the ledger at 02:45, `night.md`, posted on #12 at 02:56). The score said otherwise.

| line | forget-set loss (mean/spread) | public |
|---|---|---|
| `k7_e3t2_rm_x2` | 2.02/2.49 | 0.2133 |
| `k8_e3t2_rm1_lr1e4_x2` | 3.80/3.79 | 0.2837 |
| `k8_e3t2_rm1_lr1e4_x2p5` | 4.67/4.76 | 0.2222 |

**Why the ledger missed it.** Each of its golds is a single model, and the ones that carry the weight are confidently wrong, with loss means of 8 to 11. Against such a gold, more confident errors always help. The real target differs on two counts:
- The mean/spread term compares against the average loss of several retrained models.
- The prediction term compares against their averaged output, which is softer than any one of them.

A single model cannot be both as confident as one retrained model and as soft as their average. So the score has a peak, and the ledger has no hypothesis that produces one. With the 0.2222 included, it still predicts x2.5 above x2.

**What replaced it for the factor: `peak_check.py`** (`9f7a917`, tested in `test_peak_check.py`).
- It scores a line as mean/spread term × prediction term × retain:
  - the mean/spread term against a gold loss level;
  - the prediction term from KL at T=2 to a reference target;
  - retain from the spill probes.
- Each setting is scaled so that the 0.2837 matches, and kept only if it then reproduces the 0.2222 within 0.02.
- 13 settings survive, all with a gold loss level of about 3.3 to 4.0 (`peak_check.py` at `9f7a917`, as queued at 03:18; the comment on #12 at 03:16, from the version just before it, gave 12 settings, 0.183 for the withdrawn line and 0.295 for its replacement).
- They put the line fixed for 03:38 (`k9_e3t2_lr2e4_ep40` at x2, loss 5.29/4.86) at 0.185. We withdrew it before its slot and said why on #12 (03:16).
- In its place went the run with the least spill, lightly sharpened to the 0.2837's loss level: `k9_e3t1_lr1e4` at x1.25, predicted 0.294.
- The desk pointed out that a check calibrated on two public scores can follow the public 30%. The choice therefore also had to be backed by the probes: its spill is half that of the 0.2837's base (held-out KL 0.45 against 0.99, spare-class KL 0.17 against 0.62), with fewer changed predictions (4.4% against 5.2%).

**What the 0.256 added.** The tau 1 run at x1.25 sat at the 0.2837's forget-loss level with half its probe spill, and scored 0.256. Its factor was 0.76 times its spill temperature (1.65); the 0.2837's was 0.89 times (2.25) and the 0.2222's 1.11 times. On images the model never saw, a leftover factor of 0.76 costs about eight times the KL of 1.11. The two drops therefore have different causes:
- **the 0.2222 is a forget-side overshoot;**
- **the 0.256 is a retain loss.**

The good region is narrow: the factor near the drift's temperature, and the forget loss near 3.8/3.8. `peak_check.py` now has a temperature reading of retain. With all four scores, its best line is `k8_e3t2_rm1_ep40` at x2.25 (f/T = 1.00, loss 3.62/3.83), about 3% above the 0.2837.

**What the 0.2665 closed.** `k8_e3t2_rm1_ep40` at x2.25 matched the 0.2837's base on everything we measure: forget loss 3.62/3.83 against 3.80/3.79, the probes, and the spill temperature. Only the factor relative to that temperature differed (1.00 against 0.89), and it scored 6% lower.

With all five sharpened scores, `peak_check.py` no longer fits any setting. We stopped there: adding a fifth reading fitted to five public scores would be fitting the public 30%. Two explanations remained at 04:10:
- the best factor is near 0.89 times the drift's temperature, as at the 0.2837;
- models with equal statistics differ by about 5% on the public split.

**What settled it (04:39 to 04:53).** The second explanation is out, on two counts:
- the public split moves the ratio between two of our lines by about 1% (1 SD), not 5% (`split_noise.py`);
- the 0.2837's two seed twins are the same model at temperature 2 (`twins.py`), and their weight average scored 0.2858, +0.0022 from its base on the server's unrounded scores (0.8%).

So the 0.2665 is a real 6% below the 0.2837, from a difference our statistics do not capture: the factor relative to the drift's temperature, or something else in the 40-epoch, lower-rate run. The last night submission was the weight average alone. No eleventh-batch line beat the base on its probe, and 05:08 was left empty.

## Board noise and seed twins

**How much the public 30% moves a score** (`split_noise.py`, tested in `test_split_noise.py`, 4000 random 30% splits of the 3500 forget images, jittered copies standing in for the hidden altered ones, labels = the original's predictions):
- the ratio between two of our lines moves by about 1% (1 SD);
- one line's own forget term moves by about 2%, so a private score can sit that far from the public one;
- seed twins cannot be ranked by the board: their true gap is near 0, so the 30% and the 70% often disagree on which is better;
- every gap the night acted on is far outside that noise: 6% between 0.2665 and 0.2837 (about 6 SD), 10% to 0.256, 22% to 0.2222.

The stand-in for gold is one model, the cross-fit extrapolated 3 times. Its ratios on all images are not predictions: it puts the 0.2222 above the 0.2837. The spreads depend little on it, since they come from how unevenly the per-image losses fall across images.

**Seed twins.** The 0.2837's recipe rerun with seeds 1 and 2 matches its base on every statistic we measure:
- realized forget loss 2.23/1.82 and 2.26/1.82 against 2.23/1.84;
- held-out KL 0.96 and 1.01 against 0.99;
- changed predictions 4.9% and 5.8% against 5.2%;
- forget loss at x2 3.78/3.75 and 3.82/3.75 against 3.80/3.79.

The three pick different classes on 16% of the forget images, but only between near-tied classes. At temperature 2, where the prediction term is measured, they are the same model (`twins.py`, tested):
- their probability mixture sits 0.0013 nats from their logit average, 0.03% of its 4.33 nats from the original;
- each twin sits 0.026 nats from the mixture, 0.6%.

**The weight average** of the three, x2 (`k11_b_soup3_x2`, built with the same arithmetic as `soup.py`), scored **0.2858**, against 0.2837 for its base. That is +0.0022 on the server's unrounded scores (0.8%), inside the board's 1%. So the 0.2837 was a typical draw of its recipe. The average is our final line because it carries less seed noise into the hidden 70%, not because of its public margin. The twins' near-identity at T=2 also says that more seeds, or distilling their mixture into one network, cannot move the score. The twelfth batch (eight more seeds) was withdrawn for that reason.

## The last experiment: a mixture of independent cross-fits (06:24)

**Why.** Gold is the average of several independently retrained models. Our target comes from one cross-fit, so every seed of the recipe chases the same target, and the seeds end up as one model at T=2 (previous section). The cross-fit retrains the last 3 blocks from scratch, the way gold retrains everything. Two more cross-fits with other seeds (other folds, other initialisations) should therefore disagree the way gold models do. The fine-tune then aims at the mixture of the three extrapolated targets at temperature 2. This changes the target, not a dial; it is the one such line we had left.

**How.**
- `csd.py --extra-targets` and `lib.mixture_targets` (tested; smoke-tested on the laptop) are new.
- The thirteenth batch (`run_thirteenth.sh`, job 15905366) runs:
  - two cross-fits;
  - the 0.2837's recipe aimed at the mixture, with seeds 0 and 1, and a probe;
  - the average of the two seeds, x2: `k13_mix_soup_x2`.

**The rule, fixed on #21 before any result** (`check_thirteenth.py`, tested). The line goes in the last night slot only if:
1. the new cross-fits' held-out class differs from the first's on more than 5% of the images;
2. at T=2 the line differs from the 0.2858 by more than the seed twins differ (KL above 0.026 nats);
3. its forget loss at x2 is within 0.4 of 3.8, mean and spread;
4. its probe changes at most 6.0% of predictions.

**Result (07:15, `check_thirteenth.py` on the job's reports): two checks failed, so the slot stayed empty.**

| check | result |
|---|---|
| 1. cross-fits disagree on more than 5% | 4.7% and 5.3% (new against first), 5.1% between the new ones: **fail** |
| 2. differs from the 0.2858 at T=2 | KL 0.232: pass |
| 3. forget loss at x2 within 0.4 of 3.8 | 2.91/3.03: **fail** |
| 4. probe changes at most 6% | 4.9%: pass |

- **Independent cross-fits agree on about 95% of the held-out classes.** Retraining the last 3 blocks from scratch is not as diverse as we hoped gold's full retrains are.
- **Mixing their targets in probability space softens the target:** at x2, loss 6.46/7.82 against 8.12/9.14 for the single target (`check_thirteenth.py`).
- **The fine-tune reaches less** (realized 1.91/1.52 against 2.23/1.84) **and spills more** (held-out KL 1.07 against 0.99, spare-class KL 0.73 against 0.62; `fold_reports.py`, `ledger_inputs.json`).
- **No other factor rescues it** (`check_thirteenth.py`).
  - Its forget loss reaches 3.8 only at x2.71. That is 1.15 times its spill temperature of 2.35, and every line sharpened to or past its temperature scored lower (0.2665 at 1.00 times, 0.2222 at 1.11 times).
  - At 0.89 times its temperature (x2.09), the 0.2837's ratio, its loss is 3.02/3.17.
  - The softer target cut the reach and raised the drift temperature at once. The two conditions that held together at the 0.2837's point cannot both hold here.

## Where the ceiling is: training reach

The ledger can score a fine-tune that reached its target exactly: the target logits themselves, against the golds that carry the weight.

| Target | Forget term if reached |
|---|---|
| K3, tau 2 (the seventh batch's target) | 0.34 |
| K4, tau 1.5 | 0.51 |
| K4, tau 2 | 0.50 |
| K5, tau 2 | 0.51 |

- **The fine-tunes reach only part of that.** The dumped fine-tunes get 0.085 (`k7_e3t2_rm3`), or 0.111 sharpened x2, with about 40% of their target's loss.
- **The target is not what limits the score; how far training gets towards it is.** With retain about 0.6, a fine-tune that reached a K4, tau 2 target would score about 0.3.
- **The eighth batch measures which change buys the most reach for the least spill:** 40 epochs, a doubled learning rate, 9 blocks instead of 6, and K4 and K5 targets.

## The eighth batch: what reaches further with less spill

Each run changes one argument of `k7_e3t2_rm` (the 0.066). Its spill probe measures the drift on held-out forget images and on classes it never trained on.

| change | trained loss | held-out KL | spare-class KL | held-out predictions changed |
|---|---|---|---|---|
| none (`k7_e3t2_rm`) | 1.61 | 1.11 | 0.74 | 11% |
| untempered target, tau 1 | 2.02 | 0.61 | 0.27 | 10% |
| learning rate 1e-4 instead of 5e-5 | 2.34 | 0.99 | 0.62 | 10% |
| target tempered by 1.5 | 1.82 | 0.81 | 0.45 | 11% |
| K=4 instead of 3 | 1.95 | 1.21 | 0.76 | 20% |

- **The untempered target and the higher learning rate each win on both counts:** more reach and less spill. The ninth batch (`run_ninth.sh`) combines them.
- **K=4 doubles the changed predictions,** which puts the gate at risk. Moving further has to come from training, not from a further target.
- **Why tau 1 spills less:** a tempered target teaches the network to be less confident everywhere, and that softness leaks to images it never trained on. That leak is the temperature-like spill that sharpening removes after the fact. An untempered target avoids it during training.

## The ninth batch: more reach at tau 2

Each run changes one or two arguments of `k8_e3t2_rm1_lr1e4`, the base of the 0.2837. The probe columns are as in the eighth batch. The ledger columns use each candidate's exact logits and every score up to 02:38.

| change | trained loss (probe) | held-out KL | spare-class KL | held-out predictions changed | ledger x2 | ledger x2.5 |
|---|---|---|---|---|---|---|
| none (`k8_e3t2_rm1_lr1e4`) | 2.34 | 0.99 | 0.62 | 10% | 0.248 (observed 0.2837) | 0.283 |
| 40 epochs | 2.73 | 0.93 | 0.56 | 10% | 0.303 | 0.338 |
| learning rate 2e-4 | 2.80 | 1.05 | 0.75 | 10% | 0.299 | 0.332 |
| **2e-4 and 40 epochs** | **3.02** | 1.01 | 0.67 | 10% | **0.330** | **0.367** |
| tau 1 (no sharpening needed) | 3.12 | 0.45 | 0.17 | 8% | 0.289 | 0.260 |
| tau 3 (meant for x3, where the ledger gives 0.195) | 1.88 | 1.66 | 1.31 | 12% | 0.125 | 0.164 |
| K=3.5 | 2.75 | 1.02 | 0.62 | 13% | gate risk | gate risk |

- **Longer training reaches further and spills less.** The mixup anchor gets more steps to pull unseen images back while the forget images keep moving. At 40 epochs the training loss is still falling.
- **Tau 1 spills least but scores lower once sharpened.** On the forget images, a tau 2 target sharpened by 2 is the same logits as a tau 1 target. The difference is on unseen images: the tempered run's drift is mostly a temperature, which sharpening removes. So a tempered run plus sharpening reaches further for the same spill.
- **Tau 3 and K=3.5 are out.** Tau 3 spills twice as much. K=3.5 changes 13% of held-out predictions, which puts the gate at risk.
- **Next:** the tenth batch (`run_tenth.sh`) pushes reach further from the 2e-4, 40-epoch run: 80 epochs, learning rates 3e-4 and 4e-4, tau 1.5, a stronger anchor, and a second seed to measure run-to-run noise.

## The plateau at 0.16

Seven unrelated teams share exactly 0.16, and four share 0.12. Equal scores from different code point to a recipe whose result does not depend on the code: either the limit of training the forget images to uniform outputs, or the best point of a one-knob family such as dividing every logit by one temperature. `plateau.py` shows which golds make each recipe land on 0.16, and the ledger asks which of those golds also predict our own scores. The temperature family is out after `scale_t4`.

## Limits

- Gold is modelled by a few parametric families along our cross-fit's direction. The truth may be outside them.
- Until the dumped logits arrive (`dump_logits.py`, at the end of the seventh and eighth batches), fine-tuned candidates enter the ledger through their measured loss statistics plus a fitted proxy of their output distribution. Checked on the eighth batch, the proxy ran 3 to 8% below the exact logits (#12, 02:56).
- **The ledger had a modelling error, found and fixed at 01:13.** It multiplied each prediction by the share of the accuracy band where the candidate passes the gate. A submission either passes or scores 0, so a 0 was read as a miss rather than as a possible gate failure. That made `dial15`'s 0 rule out the removal reading entirely. Since `7b51a3c`, A is inferred jointly with gold, and the removal reading keeps the 4% it deserves.
- **Until 01:40 the posterior sat on the edge of the gold grid** (gold right on a third of the images, the furthest gold modelled). Widening the grid lowered every prediction by 10 to 20% and left the ranking unchanged. The predictions written on #12 before 01:40 come from the narrower grid.
- **The ledger's golds are single models, and the real target is several.** Against a single confidently wrong gold, sharpening always helps; against the real mixture it peaks. The 0.2222 at 03:08 showed it. From 03:15 to 04:10, the factor and the base were chosen with `peak_check.py`, calibrated on the public scores and therefore able to follow the public 30%. Choices also had to show their advantage in the probes. Its two predictions missed (0.294 for a 0.256, 0.292 for a 0.2665). With all five sharpened scores it fit no setting, and it was retired rather than refitted. The last line, the weight average, was chosen by argument alone.
- **Labels.** The forget images come without labels. We use the original's predictions, which it makes with a mean top probability of 0.9997 on these training images. The `labels` field in our cross-fit file holds the cross-fit's own predicted classes; `split_noise.py` used it by mistake until 04:51, which doubled its noise estimates.
- **Our models of the score stopped at about 5% through the night.** Near the best point, lines that match it on every statistic we measure still scored up to 6% lower (0.2665 against 0.2837). Choices below that level were made on the probes and on noise reduction (weight averaging), not on a predicted score. The morning's `score_fit.py` brings the leave-one-out error on the high lines to 1 to 2.5%, with a free level c of 0.40 to 0.52 that it cannot attribute to one term (section above).
- The hidden altered copies are not modelled. Our own alterations (flip, crop, rotation, shift, blur, JPEG, jitter, noise) track the exact images closely in every report.
- Retain is estimated from held-out forget images and spare classes, not from the real control set.
  - Those are training images of the original, which it fits with near certainty.
  - The real control and validation sets are unseen images, on which the original is right about 79% of the time. There, a fine-tune changes predictions mostly where the original is unsure, and therefore often wrong.
  - The ledger counts every changed prediction as a lost correct one, so its retain accuracy and its gate are conservative.
- The seventh batch's fine-tunes were still improving when training stopped (training KL at T=1 still falling about 0.02 per epoch over the last four epochs, in the reports' `history`), so longer or faster training reaches further. The eighth batch measures how much spill that costs.

## Design choices

- **No gradient ascent or random labels.** The metric clips overshoot to 0, and both spill onto nearby retained images (`research.md`, rows 12 and 27).
- **Checking that the model still works.** Dividing logits by a positive number never changes the predicted class. Fine-tunes are checked on held-out forget images and on classes they never trained on (`acc_vs_reference` in every probe), and the ledger turns those flips into a pass or fail at the gate for every value of A.
- **No tuning to the public board.** Each submission was chosen from predictions written before its score, one dial at a time. The board was used to reject hypotheses, not to pick thresholds.
- **The biggest mistakes.** Treating the gate as a fraction in the ledger (above), and earlier, reading `bias_full30`'s 0 as proof against removal when the gate could explain it.
- **Multiplying the last layer by 2.**
  - It changes no prediction: every image keeps its predicted class, so validation accuracy and the gate are exactly the fine-tune's.
  - It changes confidence only. The fine-tune was trained towards a target tempered by 2, so it learned to be about twice as soft as a normal model. On images it never trained on, that softness is drift away from the original, and the retain score penalises it. Multiplying by 2 gives the network back the confidence it had before the fine-tune.
  - On the forget images, the same factor makes the wrong predictions as confident as a retrained model's. The retrained models are trained without temperature, so their errors are confident.
  - The factor came from a measurement before any score: the probes put the drift on unseen classes at a temperature of about 2.4 (`night.md`, 01:20).
- **Why the ledger predicted 0.086 for a 0.2133.** Its retain estimate comes from the probes, on images the original memorised. There the fine-tune also changes some predictions, and sharpening cannot undo those. On the real control set, unseen images, the drift is apparently almost purely a temperature, so sharpening removes nearly all of it. The direction was right and the size was about twice too small. The ledger's ranking held, and the later lines use its ranking rather than its levels.
- **The factor was not tuned on the board.**
  - It moved one notch at a time with a written prediction each time. 2 scored 0.2837; 2.5 scored 0.2222.
  - We did not continue upwards, and we did not fine-tune the factor around 2 on the board. The 03:38 line keeps the 0.2837's loss level with a different base, chosen for its lower spill in the probes.
- **Why an average of three models.** The three runs share the recipe and differ only by seed, by at most 0.073 in any weight, so their average is still a fine-tune of the same model, loads into the same architecture and keeps the gate. The twins agree on every statistic and disagree on the class of 16% of the forget images, between near-tied classes: that disagreement is noise, and averaging removes part of it. The board cannot tell the twins apart (section above), so choosing the best-scoring twin would be choosing on the public split's noise.
- **Why the night stopped at 0.2858.** Every direction we could test from the 0.2837's point scored lower, by 6 to 25%:
  - more sharpening;
  - sharpening to the drift's full temperature;
  - a less-spilling run lightly sharpened;
  - less training reach (0.2133, one step back).
  The board's noise between two lines is about 1%, and the recipe's seeds are the same model at the metric's temperature, so the remaining dials cannot show a gain the board would register.
  The one step that changes the stand-in for gold itself, a mixture of three independent cross-fits, was run and not submitted (section "The last experiment"). The cross-fits agree on 95% of classes, and the line came out softer, with no factor that fixes it.
  A more diverse stand-in would need models retrained the way gold was, which takes the retained data we do not have.
- **The pre-registered rule set aside at 03:38.** The rule assumed that further-reaching fine-tunes pay at the same factor, and the 03:08 score refuted the model behind it. Submitting the rule's line would have tested a model we already knew to be wrong. We withdrew it before its slot, posted the reason and the replacement's prediction on #12 before its score, and kept the slot for a line whose advantage shows in the probes.

## Reproduce

**The 0.2837 and the 0.2133, in three steps**, from the unlearning data folder with the task's environment. The 0.2837 is the same with `--lr 1e-4` and the name `k8_e3t2_rm1_lr1e4` in step 2: sha256 `d907f3cc…` before sharpening, `522e5851…` after.

1. **The cross-fitted targets** (GPU): `output/kkrgp/xfit_extras.npz`, as in `run_third.sh`.
2. **The fine-tune** (GPU, about 8 minutes): `k7_e3t2_rm.pt`, sha256 `107d98f8…`, as in the seventh batch.
3. **The sharpening** (CPU, seconds): `k7_e3t2_rm_x2.pt`, sha256 `f69f9ba1…`. Only `head.weight` and `head.bias` differ from step 2, each exactly 2 times.

```
python $WT/tasks/unlearning/xfit.py --name extras --reading extras
python $WT/tasks/unlearning/csd.py --full 0 --hedge 0 --blocks 6 --lr 5e-5 --epochs 25 --name k7_e3t2_rm --targets output/kkrgp/xfit_extras.npz --extrapolate 3 --tau 2 --retain-mix 1
python $WT/tasks/unlearning/splice_head.py --model output/kkrgp/k7_e3t2_rm.pt --scale 0.5 --save output/kkrgp/k7_e3t2_rm_x2.pt
```

**The 0.2858**: step 2 with `--lr 1e-4` three times, as `k8_e3t2_rm1_lr1e4` (seed 0), `k11_b_s1` (`--seed 1`) and `k11_b_s2` (`--seed 2`), then their average with the last layer times 2 (sha256 `53dfdc9d…`; the desk built it with the same arithmetic on its laptop):

```
python $WT/tasks/unlearning/soup.py --models output/kkrgp/k8_e3t2_rm1_lr1e4.pt output/kkrgp/k11_b_s1.pt output/kkrgp/k11_b_s2.pt --scale 0.5 --save output/kkrgp/k11_b_soup3_x2.pt
```

**Where the exact arguments are.** Every run's report records the arguments it ran with (`reports/21/<batch>/<name>.json` on `desk/unlearning`, field `args`); the scripts can lag behind. Checked at 06:10 for the scored runs:
- `k7_e3t2_rm` (0.066, and 0.2133 at x2): blocks 6, learning rate 5e-5, 25 epochs, K=3, tau 2, retain mix 1, seed 0. `run_seventh.sh` lists only its retain-mix-3 sibling.
- `k8_e3t2_rm1_lr1e4` (0.2837 at x2): the same with learning rate 1e-4.
- `k11_b_s1` and `k11_b_s2`: the same as the 0.2837's run, with seeds 1 and 2. Averaged with it and multiplied by 2, they make the 0.2858.

**Every scored file and what built it** (scripts in `tasks/unlearning/`; the sha256 of each file is in its queue line and in the server's record):

| Score | File | Built by |
|---|---|---|
| 0.00 | `bias_full30.pt` | the owner's first candidate, before the night's work (#12 at 19:42; `analysis.md`, section 1b) |
| 0.0063 | `x3_ext_m80.pt` | `run_third.sh` (`csd.py --name x3_ext_m80 ... --mix 0.8`) |
| 0.0336 | `x4_e20.pt` | `run_fourth.sh` (`--name x4_e20 --extrapolate 2.0`) |
| 0.0057 | `scale_t4.pt` | `run_sixth.sh` (`splice_head.py --scale 4`) |
| 0.0 | `dial15.pt` | `splice_head.py --shift 15` |
| 0.066 | `k7_e3t2_rm.pt` | the seventh batch; exact arguments in its report (below) |
| 0.2133 | `k7_e3t2_rm_x2.pt` | `splice_head.py --model output/kkrgp/k7_e3t2_rm.pt --scale 0.5` |
| 0.2837 | `k8_e3t2_rm1_lr1e4_x2.pt` | `run_eighth.sh` (`--name k8_e3t2_rm1_lr1e4 ... --lr 1e-4`), then `splice_head.py --scale 0.5` |
| 0.2222 | `k8_e3t2_rm1_lr1e4_x2p5.pt` | the same base, then `splice_head.py --scale 0.4` |
| 0.256 | `k9_e3t1_lr1e4_x1p25.pt` | `run_ninth.sh` (`--name k9_e3t1_lr1e4 ... --tau 1 --lr 1e-4`), then `splice_head.py --scale 0.8` |
| 0.2665 | `k8_e3t2_rm1_ep40_x2p25.pt` | `run_eighth.sh` (`--name k8_e3t2_rm1_ep40 ... --epochs 40`), then `splice_head.py --scale 0.4444444` |
| 0.2858 | `k11_b_soup3_x2.pt` | `run_eleventh.sh` (`k11_b_s1`, `k11_b_s2`), then `soup.py ... --scale 0.5` with the 0.2837's base (above) |
| 0.2687 | `k8_e3t2_rm1_ep40_x2.pt` | the desk's line at 08:21: the 0.2665's base (`k8_e3t2_rm1_ep40`) with the last layer times 2 |

**Which command gives which number:**
- scores, times and sha256 values: the records in `tasks/unlearning/submissions/` on `main`;
- each prediction: `git show <commit>:tasks/unlearning/submit_queue.md` for the commit in the submissions table, or the #12 comment at the time given there;
- probe statistics (realized loss, held-out and spare-class KL, changed predictions): `fold_reports.py <batch folder>`, folded into `ledger_inputs.json`;
- the ledger's weights: `ledger.py`;
- board noise: `split_noise.py`;
- seed twins: `twins.py`;
- the thirteenth batch: `check_thirteenth.py`.

**The analysis**, from the same folder (`source /Volumes/euroai-apfs/ops/env.sh; cd $T` on the lead's laptop):

```
python $WT/tasks/unlearning/fold_reports.py /Volumes/euroai-apfs/r21/eighth
python $WT/tasks/unlearning/ledger.py --targets output/kkrgp/xfit_extras.npz --logits /Volumes/euroai-apfs/r21/{seventh,eighth,ninth,tenth,eleventh}/logits.npz
python $WT/tasks/unlearning/plateau.py --targets output/kkrgp/xfit_extras.npz
python $WT/tasks/unlearning/split_noise.py --draws 4000
python $WT/tasks/unlearning/twins.py
python $WT/tasks/unlearning/check_thirteenth.py /Volumes/euroai-apfs/r21/thirteenth
python $WT/tasks/unlearning/score_fit.py --retain control --proxy-temperature 1.25 --draws-out output/kkrgp/score_fit_draws.npz
python $WT/tasks/unlearning/blend_search.py $WT/tasks/unlearning blends.json
python $WT/tasks/unlearning/slot_gain.py output/kkrgp/score_fit_draws.npz --first m15_s1s2ep40x2 --lines k9_e35t2_lr1e4x1.95,k9_e3t2_lr1e4_ep40x1.85,k14_t1_lr9e5x1.45,k14_t15_lr1e4x1.7
python -m pytest $WT/tasks/unlearning -q
```

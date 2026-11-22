# Unlearning: what we did and why (brief for the judges)

Written during the night of 21 to 22 November by the unlearning agent, for the team to read in the morning. Every number below comes from a committed file or a score posted on #12; the commands at the end reproduce them. The brief is updated through the night; the last update is at the top of the log in `night.md`.

## The result in one paragraph

Our best public score is **0.2133** (`k7_e3t2_rm_x2`, 02:08), up from 0 at 21:19.

**The recipe has two steps.**
1. **Fine-tune.** The last 6 of the 12 blocks are fine-tuned on the forget images (and augmented copies) towards a target built from a cross-fitted held-out prediction. The target is extrapolated 3 times in the direction where the model would err if it had never seen the image, and tempered by 2. A mixup anchor distils the original on blends of two forget images.
2. **Sharpen.** The last layer is multiplied by 2.

**Why it works:**
- **The loss terms compare against single gold models, which are confidently wrong.** The fine-tune makes the right images wrong towards the classes a model that never saw them would pick.
- **The tempered target spreads probability over the sibling classes,** which is what the gold models' average looks like at temperature 2.
- **Sharpening restores per-model confidence and undoes the drift.** It brings back the single models' confidence on the loss side. It also takes back the temperature-like drift that the soft target leaked to unseen images, which is what the retain score penalises.

Every submission was chosen from predictions written on #12 before its score, one change at a time. Reproducing the 0.2133 takes the commands at the end and about 8 GPU minutes.

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

| Time | Candidate | What it is | Prediction written before | Public score | What it taught |
|---|---|---|---|---|---|
| 21:19 | `bias_full30` | nine classes' outputs lowered by 30 everywhere | positive under the reading that whole classes were removed | 0.00 | that reading is wrong, or the gate failed (it loses the nine classes on the validation set) |
| 23:37 | `x3_ext_m80` | fine-tune of the last 3 blocks towards a blend of the original and the cross-fitted prediction | small but positive | 0.0063 | first positive score, far smaller than the golds the dial assumed |
| 00:08 | `x4_e20` | the same, towards the cross-fitted step extrapolated 2 times | much above 0.0063 if the dial points the right way | 0.0336 | the dial points the right way: five times the previous score |
| 00:38 | `scale_t4` | every logit divided by 4 (last layer / 4), no training | about 0.008 (0.004 to 0.013) under the golds that fit the scores so far; 0.1 or more under soft golds | 0.0057 | the prediction held; soft golds, and with them a temperature as the answer, are out |
| 01:08 | `dial15` | the nine classes' bias lowered by 15, last layer only | 0 under the confidently wrong golds; 0.076 under removal with the gate; 0.29 under removal with no nine-class validation images | 0.0 | removal survives only if A is between 73.4 and 75.4: 4% of the weight |
| 01:38 | `k7_e3t2_rm` | the last 6 blocks fine-tuned towards the cross-fitted step extrapolated 3 times and tempered by 2, with an anchor | mean 0.047, 89% chance above 0.0336; 0 if it fails the gate (it changes about 6% of validation predictions) | **0.066** | twice the previous best. It passed the gate, so A × 0.94 ≥ 70. The score sits at the upper end of the prediction: the ledger underpredicts this fine-tune by about 1.3 times |
| 02:08 | `k7_e3t2_rm_x2` | the same model with its last layer multiplied by 2 | mean 0.086, 10th to 90th 0.060 to 0.118, 75% chance above 0.066 (0.106 once the exact logits arrived at 02:04) | **0.2133** | sharpening is the recipe: the same checkpoint scores more than three times as much once the spill's temperature is taken out. The ledger gets the direction right and the size wrong: the gain is about twice what it predicts |
| 02:38 | `k8_e3t2_rm1_lr1e4_x2` | one dial from the 0.2133: the same recipe trained at learning rate 1e-4, sharpened by 2 | ledger mean 0.229, 10th to 90th 0.142 to 0.301, 66% chance above 0.2133; 1.7 times `k7_e3t2_rm_x2` relative to the ledger | pending | |

## What the evidence says about gold (as of 02:15)

- **After the 0.2133, two thirds of the weight sit on golds that err where our cross-fit says.** Golds that err on other images of the same classes fall to 11%: they cannot produce a gain of more than about 1.2 times from sharpening, and we observed 3.2 times.
- **The ledger underpredicts sharpened fine-tunes by about 1.6 times.** Under the best-fitting gold, the unsharpened score is predicted exactly (0.063 against 0.066), and the sharpened one at 0.162 against 0.2133. The gap fits a retain near 0.9 after sharpening, that is, a drift on unseen images that is almost purely a temperature. The probes, on training images of the original, also show changed predictions that the real control set apparently does not.

Before the 0.2133, as of 01:45:

- **Gold is confidently wrong on most of the forget images.**
  - Until 01:40 the weight sat on golds right on about a third of the images, with loss mean 7.6 to 8.1 and spread 8.8 to 9.1. That was the edge of the grid.
  - With the grid widened to golds right on only 10%, the weight spreads evenly over golds right on 10 to 33% of the images, with loss mean 8 to 15 (`ledger.py`, `37cd60c`). Our scores so far cannot tell where in that range gold is.
  - Soft golds, which would make one temperature the right answer, predicted `x3_ext_m80` at ten times its score and `scale_t4` at 0.1 or more.
- **This matches published retrains when a large share of a class is deleted.** Dolgova and Stich (CISPA, 2026) report, on CIFAR-100, that a retrain is right on 47% of a class's deleted images when half of the class is deleted, and on 17% at 90%, against 72% test accuracy (`research.md`, row 22). Our 36 thinned classes lost 35 to 80% of their images. The README says the same in words: the target covers "which wrong class it picks for each one".
- **The wrong class is a sibling.** The original and gold are ViT-B/16 trained from scratch (the organizers' template: `pretrained=False`, LayerScale), which generalise poorly from 130 images per class. The forget images concentrate on look-alike classes: labels 29, 30, 31 and 33 are four kinds of small boat on misty water, each among the nine classes that lost all 130 images, next to 32 (fishing boats, 104 images) and 23 (bridges over water). Our cross-fit errs 19→21, 43↔42, 2→0, 35→38; the golds that fit go about three times further in that direction.
- **The original's validation accuracy A is about 79%** (10th to 90th percentile 74 to 84), inferred from the five scores together.

## The utility gate, and localisation

A submission under 70% accuracy on the hidden validation set scores 0. Inferred A leaves a margin: a candidate may change about 5 to 15% of validation predictions, depending on where A really is.

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
  - `k7_e3t2_rm3`'s forget score goes from 0.099 to 0.136 at s = 2 (mean/spread term 0.237, prediction term 0.602).
- **Retain side:**
  - on images whose prediction the fine-tune kept, sharpening divides the KL to the original (1.36 to 0.50 at s = 2);
  - on images whose prediction it changed, the KL stays put (2.11 to 1.99);
  - the ledger mixes the two at each probe's measured share of changed predictions (`sharpened_retain`, `ledger.py`, tested in `test_ledger.py`). `k7_e3t2_rm3`'s retain goes from 0.48 to 0.67 at s = 2.
- **Where it stops paying:**
  - over the golds that fit our scores, the ledger peaks at s = 2 to 2.5 (`k7_e3t2_rm`: 0.086 at 2, 0.093 at 2.5, 0.089 at 3, 0.078 at 4);
  - golds that err where our cross-fit says keep rising with s, while golds that err on other images of the same classes fall to 0 at s = 4;
  - `x4_e20` gets worse when sharpened, because its targets were never tempered.

We queued `k7_e3t2_rm` and then the same model at s = 2, so the two scores isolate this one change.

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

## The plateau at 0.16

Seven unrelated teams share exactly 0.16, and four share 0.12. Equal scores from different code point to a recipe whose result does not depend on the code: either the limit of training the forget images to uniform outputs, or the best point of a one-knob family such as dividing every logit by one temperature. `plateau.py` shows which golds make each recipe land on 0.16, and the ledger asks which of those golds also predict our own scores. The temperature family is out after `scale_t4`.

## Honest limits

- Gold is modelled by a few parametric families along our cross-fit's direction. The truth may be outside them.
- Until the dumped logits arrive (`dump_logits.py`, at the end of the seventh and eighth batches), fine-tuned candidates enter the ledger through their measured loss statistics plus a fitted proxy of their output distribution. The proxy is known to be off by up to about two times on single candidates.
- **The ledger had a modelling error, found and fixed at 01:13.** It multiplied each prediction by the share of the accuracy band where the candidate passes the gate. A submission either passes or scores 0, so a 0 was read as a miss rather than as a possible gate failure. That made `dial15`'s 0 rule out the removal reading entirely. Since `7b51a3c`, A is inferred jointly with gold, and the removal reading keeps the 4% it deserves.
- **Until 01:40 the posterior sat on the edge of the gold grid** (gold right on a third of the images, the furthest gold modelled). Widening the grid lowered every prediction by 10 to 20% and left the ranking unchanged. The predictions written on #12 before 01:40 come from the narrower grid.
- The hidden altered copies are not modelled. Our own alterations (flip, crop, rotation, shift, blur, JPEG, jitter, noise) track the exact images closely in every report.
- Retain is estimated from held-out forget images and spare classes, not from the real control set.
  - Those are training images of the original, which it fits with near certainty.
  - The real control and validation sets are unseen images, on which the original is right about 79% of the time. There, a fine-tune changes predictions mostly where the original is unsure, and therefore often wrong.
  - The ledger counts every changed prediction as a lost correct one, so its retain accuracy and its gate are conservative.
- The seventh batch's fine-tunes were still improving when training stopped (training KL at T=1 falling about 0.03 per epoch at epoch 25), so longer or faster training reaches further. The eighth batch measures how much spill that costs.

## Likely questions

- **Why not gradient ascent or random labels?** The metric clips overshoot to 0, and both spill onto nearby retained images (`research.md`, rows 12 and 27).
- **How do you know the model still works?** Dividing logits by a positive number never changes the predicted class. Fine-tunes are checked on held-out forget images and on classes they never trained on (`acc_vs_reference` in every probe), and the ledger turns those flips into a pass or fail at the gate for every value of A.
- **Did you tune to the public board?** No. Each submission was chosen from predictions written before its score, one dial at a time. The board was used to reject hypotheses, not to pick thresholds.
- **What was the biggest mistake?** Treating the gate as a fraction in the ledger (above), and earlier, reading `bias_full30`'s 0 as proof against removal when the gate could explain it.
- **Isn't multiplying the last layer by 2 a trick?**
  - It changes no prediction: every image keeps its predicted class, so validation accuracy and the gate are exactly the fine-tune's.
  - It changes confidence only. The fine-tune was trained towards a target tempered by 2, so it learned to be about twice as soft as a normal model. On images it never trained on, that softness is drift away from the original, and the retain score penalises it. Multiplying by 2 gives the network back the confidence it had before the fine-tune.
  - On the forget images, the same factor makes the wrong predictions as confident as a retrained model's. The retrained models are trained without temperature, so their errors are confident.
  - The factor came from a measurement before any score: the probes put the drift on unseen classes at a temperature of about 2.4 (`night.md`, 01:20).
- **Why did the ledger predict 0.086 for a 0.2133?** Its retain estimate comes from the probes, on images the original memorised. There the fine-tune also changes some predictions, and sharpening cannot undo those. On the real control set, unseen images, the drift is apparently almost purely a temperature, so sharpening removes nearly all of it. The direction was right and the size was about twice too small. The ledger's ranking held, and the later lines use its ranking rather than its levels.
- **Is the factor tuned to the board?**
  - It moves one notch at a time (2, then 2.5), each with a written prediction.
  - The stopping point is set by the theory: past the drift's temperature, retain falls again. Under some golds, which err on other images of the same classes, the forget term collapses beyond 3.
  - The night plan forbids a factor above 3 without a score at 2.5 first.

## Reproduce

**The 0.2133, in three steps**, from the unlearning data folder with the task's environment:

1. **The cross-fitted targets** (GPU): `output/kkrgp/xfit_extras.npz`, as in `run_third.sh`.
2. **The fine-tune** (GPU, about 8 minutes): `k7_e3t2_rm.pt`, sha256 `107d98f8…`, as in the seventh batch.
3. **The sharpening** (CPU, seconds): `k7_e3t2_rm_x2.pt`, sha256 `f69f9ba1…`. Only `head.weight` and `head.bias` differ from step 2, each exactly 2 times.

```
python $WT/tasks/unlearning/xfit.py --name extras --reading extras
python $WT/tasks/unlearning/csd.py --full 0 --hedge 0 --blocks 6 --lr 5e-5 --epochs 25 --name k7_e3t2_rm --targets output/kkrgp/xfit_extras.npz --extrapolate 3 --tau 2 --retain-mix 1
python $WT/tasks/unlearning/splice_head.py --model output/kkrgp/k7_e3t2_rm.pt --scale 0.5 --save output/kkrgp/k7_e3t2_rm_x2.pt
```

**The analysis**, from the same folder (`source /Volumes/euroai-apfs/ops/env.sh; cd $T` on the lead's laptop):

```
python $WT/tasks/unlearning/fold_reports.py /Volumes/euroai-apfs/r21/eighth
python $WT/tasks/unlearning/ledger.py  --targets output/kkrgp/xfit_extras.npz --logits /Volumes/euroai-apfs/r21/seventh/logits.npz /Volumes/euroai-apfs/r21/eighth/logits.npz
python $WT/tasks/unlearning/plateau.py --targets output/kkrgp/xfit_extras.npz
python -m pytest $WT/tasks/unlearning -q
```

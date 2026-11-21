# Unlearning: what we did and why (brief for the judges)

Written during the night of 21 to 22 November by the unlearning agent, for the team to read in the morning. Every number below comes from a committed file or a score posted on #12; the commands at the end reproduce them. Updated through the night; the last update is at the top of the log in `night.md`.

## The problem in four sentences

We get a ViT-B/16 trained from scratch on a synthetic 100-class image set and 3,500 of its training images to forget. The submission is scored against "gold", several models retrained without those images: the loss mean and spread on the forget images and on hidden altered copies of them must match gold's, and the predicted class distribution must match gold's, each term measured relative to how far the original model is from gold. A control set penalises any drift elsewhere, and a model under 70% validation accuracy scores 0. We have no retained data, no labels beyond the original model's own predictions, and no gold model.

## What the metric rewards, and what it does not

- **Matching, not forgetting.** Both forget terms are 1 when the submission behaves like gold and 0 when it behaves like the original; going past gold is clipped to 0 just like not moving. Maximal forgetting (gradient ascent, random labels) scores 0.
- **Near the original, the score is roughly quadratic.** A candidate that moves a fraction t of the way scores about t squared on the forget side, so timid candidates score near 0.
- **The terms multiply.** Forget × retain, and within forget, mean/spread × prediction: one weak term sinks the product.
- **Retain is a drift penalty.** Even with accuracy intact, the KL between the original's and the submission's outputs on the control set divides the score (retain = 1 / (1 + KL)).

## How we worked through the night

Three agents coordinated on GitHub issues while the humans slept: the desk (the lead's agent, the only one that touches the cluster and submits), the model-equality agent, and the unlearning agent. The unlearning agent had a laptop copy of the model and the forget images, and no cluster access; it asked for GPU runs in batches (#21) and for submissions through a queue (`submit_queue.md`, #12). Each submission line states, before submission, what its score will tell.

The method is a **pre-registered hypothesis search**:

1. Gold is unknown, so it is parametrised: a family of gold hypotheses built from the original model's logits and a cross-fitted held-out prediction (`xfit.py`: the last blocks retrained on the other forget images, so each image's prediction comes from blocks that never saw it). Two parameters: how often and how far gold is wrong (K), and how soft it is (tau_g, which also stands for averaging several gold models). Two families: gold errs where our cross-fit says, or on other images of the same classes.
2. For every candidate and every gold hypothesis, the server's formulas give a predicted score (`scoring.py`, tested in `test_scoring.py`; `ledger.py`).
3. Each public score weighs the hypotheses by how well they predicted it. The next submission is the one with the best expected gain under the hypotheses that survive, or the one that best separates them.

This keeps the number of submissions small and makes every submission an experiment with a written prediction (#12), not a lottery ticket. It also follows the lead's rule against fitting the public board: we move one dial at a time and accept a family only when its neighbours agree.

## Submissions and what each one taught

| Time | Candidate | What it is | Prediction written before | Public score | What it taught |
|---|---|---|---|---|---|
| 21:19 | `bias_full30` | nine classes' outputs lowered by 30 everywhere | positive under the reading that whole classes were removed | 0.00 | that reading is wrong, or the gate failed (nine classes lost on the validation set) |
| 23:37 | `x3_ext_m80` | fine-tune of the last 3 blocks towards a blend of the original and the cross-fitted prediction | small but positive | 0.0063 | first positive score; far smaller than the golds the dial assumed |
| 00:08 | `x4_e20` | the same, towards the cross-fitted step extrapolated 2 times | much above 0.0063 if the dial points the right way | 0.0336 | the dial points the right way; five times the previous |
| 00:38 | `scale_t4` | every logit divided by 4 (last layer / 4), no training | about 0.008 under the golds that fit the scores so far; 0.1 or more under soft golds | pending | pending |

## What the evidence says about gold (as of 00:30)

- The two positive scores together favour golds that are **confidently wrong on most forget images** (accuracy about a third, loss mean 5 to 8). Soft golds, which would make one temperature the right answer, predicted `x3_ext_m80` at ten times its score.
- This fits the data: the original and gold are ViT-B/16 trained from scratch (the organizers' template: `pretrained=False`, LayerScale), which generalise poorly from small data, and the forget images include near-duplicate classes (six "sunset over water" classes among the labels 29 to 35).
- Under such golds, the best targets for the forget images are the cross-fitted step extrapolated about 3 times and tempered by 2 to 4 (predicted forget side 0.2 to 0.35). The obstacles are realising them (our fine-tunes reached a fifth of the target loss) and keeping the change on the forget images (unseen images of the same classes took 85% of the change in `d5_e30_probe`).

## The plateau at 0.16

Seven unrelated teams share exactly 0.16, and four share 0.12. Equal scores from different code point to a recipe whose result does not depend on the code: either the limit of training the forget images to uniform outputs, or the best point of a one-knob family such as dividing every logit by one temperature. `plateau.py` shows which golds make each recipe land on 0.16; the ledger then asks which of those golds also predict our own scores. Under the golds favoured at 00:30, near-uniform outputs on the forget images alone score 0.06 to 0.27 on the forget side, which brackets 0.16 when retain stays near 1.

## Honest limits

- Gold is modelled by a two-parameter family along our cross-fit's direction, plus a shuffled variant. The truth may be outside it; the best-fitting golds still over-predict `x3_ext_m80` by about 1.8 times.
- Fine-tuned candidates enter the ledger through their measured loss statistics plus a fitted proxy of their output distribution, until their logits come back (`dump_logits.py`).
- The hidden altered copies are not modelled; our own alterations (flip, crop, rotation, shift, blur, JPEG, jitter, noise) track the exact images closely in every report.
- Retain is estimated from held-out forget images and spare classes, not from the real control set.

## Likely questions

- **Why not gradient ascent or random labels?** The metric clips overshoot to 0, and both spill onto nearby retained images (`research.md`, row 12).
- **How do you know the model still works?** Dividing logits by a positive number never changes the predicted class, so accuracy and the gate are exactly the original's. Fine-tunes are checked on held-out forget images and on classes they never trained on (`acc_vs_reference` in every probe).
- **Did you tune to the public board?** No: each submission was chosen from predictions written before its score, one dial at a time; the board was used to reject hypotheses, not to pick thresholds.

## Reproduce

From the unlearning data folder, with the task's Python environment (`source /Volumes/euroai-apfs/ops/env.sh; cd $T` on the lead's laptop):

```
python $WT/tasks/unlearning/plateau.py --targets output/kkrgp/xfit_extras.npz
python $WT/tasks/unlearning/ledger.py  --targets output/kkrgp/xfit_extras.npz
python -m pytest $WT/tasks/unlearning/test_scoring.py -q
```

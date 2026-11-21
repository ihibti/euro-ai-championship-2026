# Unlearning: research notes

Issue #3. Sources were opened by three research agents on 2026-11-21; a claim without a link is our own inference and is marked so. Licences were not checked unless stated.

## What decides the method here

- **Nothing public describes this metric, the dataset or the gold recipe.** The Hugging Face repo has no card; the lab's unlearning papers are in other settings ([contrastive models](https://arxiv.org/abs/2406.03603), [image generators](https://arxiv.org/abs/2606.25548)). The NeurIPS 2023 challenge is the closest relative: retraining as reference and a forget × utility product, but 512 runs per method and per-example tests instead of one model, loss mean/std and a KL to gold's mean prediction ([findings](https://arxiv.org/abs/2406.09073)).
- **The images are generated.** 48 forget images looked at on 2026-11-21 are text-to-image pictures of objects (figurines, aloe plants, lanterns, boats, kites). The dataset is custom and synthetic, so no public dataset holds the retained images.
- **Outside data is allowed.** An organizer answered on 2026-11-21: "Yes you can use external dataset. The dataset that was used to train model.pt is not disclosed."
- **No label smoothing.** Smoothing 0.1 caps training confidence near 0.90 and the margin near 6.8 logits; the original model shows 0.99999 and about 12 logits (inference). LayerScale 1e-4 is the DeiT-III default, and [DeiT-III](https://arxiv.org/abs/2204.07118) trains with a binary cross-entropy: to check on the logits (under BCE all non-top logits share one strongly negative level).
- **Whole classes or a cap?** 130 fits a whole class of a 130-per-class dataset (then 46–104 is exactly 35–80%) and also 10% of a 1,300-per-class one. A ViT-B from scratch needs about 1,000 real images per class to pass 70% ([Liu et al.](https://arxiv.org/abs/2106.03746)), but clean generated images of distinct objects are easier. Only a submission settles it.

## The four papers the organizers cite

| Paper | Method in short | What transfers |
|---|---|---|
| [SCRUB](https://arxiv.org/abs/2302.09880) | Student starts as the teacher; alternate ascent on KL to the teacher over the forget set and descent on KL + CE over the retain set. SCRUB+R rewinds to the checkpoint whose forget error matches held-out data. | The idea of stopping at a reference level, with the server as the reference. The method itself needs retain data; code under a non-commercial licence ([repo](https://github.com/meghdadk/SCRUB)). |
| [Zero-shot unlearning](https://arxiv.org/abs/2201.05629) | Error-minimizing/maximizing noise, or a generator with gated knowledge transfer; no real data. | Little: on a ViT the noise method leaves forget accuracy at 94% and the generator method collapses retained accuracy to 1% ([JiT tables](https://arxiv.org/abs/2402.01401)). |
| [SalUn](https://arxiv.org/abs/2310.12508) | Random labels on the forget set, restricted to weights with large forget gradients. | The weight mask as a way to limit what moves. Without retain data its gap to retraining is 4.11 against 1.94 for AMUN ([AMUN](https://arxiv.org/abs/2503.00917)). |
| [NPO](https://arxiv.org/abs/2404.05868) | Not opened by our agents. | — |

## Newer work that fits "forget set only, match the retrained model"

- **Masked-teacher distillation, [DELETE](https://arxiv.org/abs/2503.23751) (CVPR 2025).** Forget images only. Target: the original softmax with the forgotten class's logit removed, matched by KL. On ViT-S, retained test accuracy 77.2 against 76.4 for retraining. Whole classes only; no public code found. A finite offset `d` instead of removal gives it a dial: this is our class dial, without training.
- **[TRW](https://arxiv.org/abs/2506.20893).** Retrained models send a forgotten class to one to three similar classes, more concentrated than the original's renormalized ranking; TRW tilts the target towards classes with similar last-layer rows. Counterpoint: class-averaged renormalized outputs "line up nicely" with 100 retrained models on a small CNN ([Baumhauer et al.](https://arxiv.org/abs/2002.02730)), who also show that removing a class at the output folds into the last layer.
- **[AMUN](https://arxiv.org/abs/2503.00917) (ICML 2025), [code](https://github.com/Ali-E/AMUN).** Fine-tune on each forget image's nearest adversarial example, labelled with the class it flips to. Lowers confidence on the forget images towards unseen-image levels and stops there by itself. Best reported forget-only result for partly removed data; no ViT results.
- **[JiT](https://arxiv.org/abs/2402.01401), [code](https://github.com/jwf40/Zeroshot-Unlearning-At-Scale).** Smooths the output around forget images. On ViT it under-forgets whole classes (51.9% where retraining has 0) and over-forgets sub-classes (0% where retraining has 22%).
- **[Boundary Shrink](https://arxiv.org/abs/2303.11570).** Relabels to the nearest wrong class with hard labels: overconfident, which a KL to gold punishes.
- **[SCAR](https://arxiv.org/abs/2404.12922), [code](https://github.com/jbonato1/SCAR).** Distils on 6–10K outside natural images in place of retain data; with noise instead, retained accuracy collapses to 16.6%. Now usable, since outside data is allowed.
- **Partly removed classes behave like test images.** With half of a class removed, the retrained model is right on 47.4% of the removed images against 49.6% on that class's test images ([LTD](https://arxiv.org/abs/2605.31317)); retrained ViTs range from 22% to 95% by sub-class ([JiT](https://arxiv.org/abs/2402.01401)). The class-prior effect is negligible ([logit adjustment](https://arxiv.org/abs/2007.07314)): the change is per image.
- **What to avoid.** Gradient ascent without retain data: gap 68.5 ([AMUN](https://arxiv.org/abs/2503.00917)). [Cha et al.](https://arxiv.org/abs/2301.11578): retained accuracy falls from 87 to 65.
- **Limit what moves.** Low-rank or sparse updates beat full fine-tuning without retain data ([Poppi et al.](https://zenodo.org/records/14031683)); ViTs want small learning rates ([Zhao et al.](https://arxiv.org/abs/2602.20114)).

## Three ranked approaches

| # | Approach | Effort | First experiment |
|---|---|---|---|
| 1 | **Class dial**: lower the last-layer bias of the nine 130-image classes by `d` (`head_edit.py`). Training-free, exact effect from saved logits, carries over to altered copies. | 1 h, no GPU | Submit `d = 25`. Above 0 confirms whole-class removal; then two more points along `d` locate gold's loss level. |
| 2 | **Hedge for the 36 thinned classes**: lower each by an amount that grows with its removed share, as a second dial; then an image-level version by fine-tuning towards soft targets, with adversarial distance as the per-image signal (AMUN's idea) and outside images as anchors (SCAR's idea). | 3 h, GPU jobs of minutes | Train on 80% of each class's forget images; read on the other 20% how much of the change spills over (`csd.py`). |
| 3 | **Tilt where the nine classes' probability goes** towards the most similar remaining classes (TRW), as a closed-form change of the last layer. | 2 h, laptop | One submission with and one without the tilt at the best `d`. |

## Local proxy score

1. Forget side, exact: loss mean and spread against the original predictions, and the distribution at temperature 2, on exact images and nine altered views.
2. Retain side: hold out 20% of every class's forget images from the method. They are memorized training images, like retained ones: accuracy and KL at temperature 2 against the edited model on them stand in for the retain term, and agreement under 95% means the 70% gate is at risk.
3. Untouched classes: outside images that the original model classifies with high confidence, as a drift meter.
4. A replica with a known gold (same forget structure on a public dataset, scored with the exact formula) ranks methods; the server calibrates the dials.

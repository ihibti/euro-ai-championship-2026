# Unlearning: research notes (forget images only, ViT-B/16, gold = average of retrained models)

**What decides the score.** The gold models never saw the forget images, so they behave on them as on held-out images. Their loss is higher than orig's, it is more spread out, and their wrong guesses go to semantically similar classes. A method only helps if, without labels and without retain data, it (a) brings the loss mean and std on the forget images down to held-out levels, (b) keeps orig's ranking of the non-top classes, since that ranking is the best guess of gold's wrong class, (c) carries over to altered images, and (d) leaves orig almost unchanged elsewhere, because retain KL and the 70% gate multiply the score. Throughout, the pseudo-label is orig's top-1 prediction.

## 1. Cited papers

| Paper | Method in three lines | What transfers to our setting |
|---|---|---|
| **SCRUB** ([arXiv 2302.09880](https://arxiv.org/abs/2302.09880)) | The original model is a frozen teacher and the student starts from it. Loss: α·KL(teacher‖student) + γ·CE on the retain set, minus KL on the forget set. Max-steps (forget) and min-steps (retain) alternate, then extra min-steps. SCRUB+R "rewinds" so the forget error is not abnormally high. | The max-step needs only forget images, and KL to the teacher needs no labels. We have no retain set, so the min-step needs proxy inputs (see §5). Pushing KL up without limit overshoots gold. Keep the rewind idea of stopping at a target loss level. |
| **Zero-Shot MU** ([arXiv 2201.05629](https://arxiv.org/abs/2201.05629)) | (a) Learns error-maximising noise for the forget class and error-minimising noise as a stand-in for retain classes. (b) Gated knowledge transfer: a generator makes pseudo-samples, a band-pass filter drops those the teacher assigns to the forget class, and the student distils the teacher on the rest. | Built for forgetting whole classes, which we are not doing. The useful idea is to synthesise retain-like inputs and distil orig on them, which gives a retain KL anchor with no data. Our filter would have to drop near-copies of forget images rather than a class. |
| **SalUn** ([arXiv 2310.12508](https://arxiv.org/abs/2310.12508)) | Saliency mask: weights where \|∇ CE on the forget set\| is at least the median. Only masked weights are updated. Objective: CE on random wrong labels for forget images + α·CE on the retain set. | The mask is label-free with pseudo-labels and limits drift, which helps retain KL. Random labels send each image to an arbitrary wrong class, which goes against the per-image KL to gold. Use the mask, not the random labels. A ViT benchmark found SalUn underperforms on ViTs (see §2). |
| **NPO** ([arXiv 2404.05868](https://arxiv.org/abs/2404.05868)) | L = −(2/β)·E log σ(−β·log(π_θ/π_ref)) on forget data. Its gradient is gradient ascent reweighted so updates shrink as π_θ drops below π_ref, which makes collapse exponentially slower than gradient ascent. Variant NPO+RT adds a retain loss. | A bounded forget loss on the pseudo-label is a drop-in replacement for gradient ascent. β controls how far the confidence drops. It still says nothing about which wrong class wins. |

## 2. Newer work (all opened)

- **NeurIPS 2023 competition findings** ([arXiv 2406.09073](https://arxiv.org/abs/2406.09073)): forget quality is measured per example by comparing output distributions of many unlearned vs. retrained models, close to our gold average. All top-8 entries use the retain set (re-initialise, add noise, fine-tune). The only forget-only piece is Fanchuan's first pass: KL to a uniform distribution on forget batches.
- **TREW** ([arXiv 2506.20893](https://arxiv.org/abs/2506.20893)): retrained models spread forget-class probability by class similarity. The target "tilts" the renormalised orig distribution by class similarity (cosine of head weights). This is the closest existing idea to matching gold's wrong-class choice, though the paper is about class unlearning.
- **Loss-Free SSD** ([arXiv 2402.19308](https://arxiv.org/abs/2402.19308)), built on SSD ([arXiv 2308.07707](https://arxiv.org/abs/2308.07707)): label-free and training-free. Weight importance is the gradient of ‖output‖², and weights that matter most for the forget set are dampened. Tested on ViT. SSD also needs importance on the full training data, which we must approximate.
- **WIN-U** ([arXiv 2604.13438](https://arxiv.org/abs/2604.13438)): one Newton step without retain data, using Woodbury + Gauss-Newton on the forget-set curvature. Exact for linear models.
- **Source-free MU** ([arXiv 2508.15127](https://arxiv.org/abs/2508.15127)): estimates the Hessian of the unseen retain data for a zero-shot Newton step. Tested on CIFAR, Stanford Dogs and Caltech-256.
- **Benchmarking MU for ViTs** ([arXiv 2602.20114](https://arxiv.org/abs/2602.20114), CoLLAs 2026): NegGrad+ is a consistently strong method on ViTs. SalUn underperforms. Splitting the forget set by memorisation (RUM) helps.
- **LetheViT** ([arXiv 2508.01569](https://arxiv.org/abs/2508.01569), abstract only): a masked image gives the positive logits and the original image the negative, for forgetting random samples in ViTs.
- **Forget vectors** ([arXiv 2412.16780](https://arxiv.org/abs/2412.16780)): a universal input perturbation can unlearn. Not usable here because the server feeds raw images, but it shows that augmentations carry the forgetting signal.
- **Why forget-only unlearning needs memorization** ([arXiv 2610.10519](https://arxiv.org/abs/2610.10519)): lower bounds proving that forget-only unlearning cannot always match retraining. Expect a ceiling below 1.
- **BatchNorm illusion** ([arXiv 2609.08901](https://arxiv.org/abs/2609.08901)): reported "forgetting" can be a BatchNorm-statistics artefact. ViT uses LayerNorm, so this does not apply to us.

## 3. Public code

| Repository | Licence |
|---|---|
| [meghdadk/SCRUB](https://github.com/meghdadk/SCRUB) | Academic licence, no commercial use |
| [ayushkumartarun/zero-shot-unlearning](https://github.com/ayushkumartarun/zero-shot-unlearning) | MIT |
| [OPTML-Group/Unlearn-Saliency](https://github.com/OPTML-Group/Unlearn-Saliency) | MIT |
| [licong-lin/negative-preference-optimization](https://github.com/licong-lin/negative-preference-optimization) | No licence file found |
| [if-loops/selective-synaptic-dampening](https://github.com/if-loops/selective-synaptic-dampening) (SSD) | MIT |
| [google-deepmind/unlearning_evaluation](https://github.com/google-deepmind/unlearning_evaluation) | Apache-2.0 |
| [unlearning-challenge/starting-kit](https://github.com/unlearning-challenge/starting-kit) | Apache-2.0 |
| [UCR-Vision-and-Learning-Group/source-free-unlearning](https://github.com/UCR-Vision-and-Learning-Group/source-free-unlearning) | MIT |
| [kairanzhao/Unlearning_VTs](https://github.com/kairanzhao/Unlearning_VTs) | No licence file found |
| [Changchangsun/Forget-Vector](https://github.com/Changchangsun/Forget-Vector) | MIT |

The TREW repository link given in the paper returns 404.

## 4. Ranked approaches

GPU estimates are my own guesses (3,500 images, ViT-B, fp16, one A100-class GPU), not measured.

1. **Distil a constructed "held-out-like" target (TREW-style), with an anchor to orig.** Effort: about 6 h; 5–15 GPU-min per run.
   - Target per forget image: q(x) = softmax of orig's logits after lowering the pseudo-label logit by δ and softening by τ. This keeps orig's ranking of the other classes, optionally tilted by head-weight cosine similarity.
   - Training: minimise KL(q‖p_sub) at T = 2 on forget images under random augmentations, which covers the altered copies. Add KL(p_orig‖p_sub) on anchor inputs (mixup/CutMix across classes, patch shuffles) and an L2 penalty to orig. Optionally update only the SalUn-masked weights or the last blocks and head.
   - Two scalars, δ and τ, set the loss mean and std, so this is the only option that targets both factors of the forget score.
   - First experiment: sweep δ on a grid, read the pseudo-label loss mean and std and the proxy retain KL (§5), and submit two or three δ values to read the public leaderboard.
2. **SCRUB with a bounded forget loss.** Effort: about 4 h; about 10 GPU-min per run.
   - Max-step: NPO on the pseudo-label (β about 0.1–1) instead of maximising KL. Min-step: KL to orig on the anchor inputs from approach 1.
   - Stop as soon as the mean pseudo-label loss reaches the chosen target (the rewind idea).
   - Simpler, but less control over which wrong class each image ends up in.
   - First experiment: NPO-only vs. NPO + anchor at the same stopping loss, compared on the proxy.
3. **Training-free LF-SSD dampening, or a WIN-U-style Newton step.** Effort: about 3 h; under 5 GPU-min.
   - Compute importance on forget images and on anchor inputs (standing in for the training data), then dampen with α swept as in the paper, λ = 1.
   - Fast to try, and useful as a first step before approach 1.
   - First experiment: sweep α and plot proxy retain KL against the pseudo-label loss shift.

**Cross-cutting.** Submit merged float32 weights only. Score locally under fp16 autocast. Without labels, the pseudo-label equals the true label only to the extent orig fits its training images. Check orig's confidence histogram on the forget set.

## 5. Proposed local proxy score

- **Held-out split.** Use 5-fold splits of the forget set. Unlearn on 80% (A) and keep 20% (B) untouched. B images are training members from the same distribution, so they stand in for the retain/control set:
  - R̂ = agree_top1(sub, orig on B) × 1/(1 + mean KL₂(p_orig‖p_sub on B)).
  - Also report the same KL on anchor inputs.
  - The final submission unlearns all 3,500 images with the hyperparameters chosen this way.
- **Utility guard.** Require at least 98% top-1 agreement with orig on B and on mildly augmented B. This is a stand-in for the 70% gate.
- **Forget side (A, on exact images and our own alterations).**
  - M̂ = 1 − clip(d(sub, target)/d(orig, target), 0, 1), where d is |Δmean| + |Δstd| of the pseudo-label loss and the target is set by δ.
  - P̂ = Spearman correlation between sub's and orig's non-top-class ranking. It is a sanity check that wrong-class choices are not random.
  - F̂ = M̂ × P̂, averaged over exact and altered images. Track the exact-vs-altered loss gap; it should not grow.
- **Proxy score.** F̂ × R̂ ranks variants. The forget side is circular because we choose the target, so the one free level, δ, is set with a few public-leaderboard submissions (about 30% of the scoring data).

## Not opened or not verified

- Kaggle write-ups of the NeurIPS 2023 challenge (I used the organisers' findings paper instead).
- LetheViT, Forget vectors, IDU ([arXiv 2609.36099](https://arxiv.org/abs/2609.36099)) and the zero-glance few-shot paper ([arXiv 2511.13116](https://arxiv.org/abs/2511.13116)) beyond their abstracts. IDU and the zero-glance paper target generative models or class forgetting.
- The TREW repository (404).
- Licences for the NPO and Unlearning_VTs repositories (no licence file).
- Whether the RUM repository has code for ViTs.
- The brief file papers/briefs/2302.09880_scrub.md (not opened, as instructed).

## How these notes were checked

- Written on 2026-11-21 by a research agent from the task statement, without access to the repository or the data.
- Checked afterwards: every arXiv identifier above resolves to a paper with the cited title, and every repository exists with the licence shown.
- Not checked: the method summaries, which come from one reading of each paper, and the effort and GPU estimates.
- The measurements of the forget set in `PLAYBOOK.md` are ours and take precedence where they disagree with these notes.

## Night research

Written during the night of 21 to 22 November for the desk's orders on #28. Each row: the claim, the source, the number, how it maps to our setting, the check on our data, the decision it changes.

### U1: the plateaus at 0.16 and 0.12 (00:05)

Method: `plateau.py`. Gold is unknown, so it is parametrised as `gold = (orig + K * step) / tau_g`, where `step` goes from the original logits to the cross-fitted held-out prediction (`xfit_extras.npz`). K sets how often and how far gold is wrong (gold accuracy on the forget images: K=0 → 1.00, 1 → 0.95, 1.25 → 0.87, 1.5 → 0.77, 2 → 0.58, 3 → 0.33); tau_g sets how soft it is, which also stands for the averaging of several gold models. For every gold, the server's formulas score each cheap recipe on the exact images. Retain is computed from our own logits, with the control set taken either as memorised training images (the original logits on the forget images) or as unseen images (the cross-fitted prediction).

| # | Claim | Source | Number | Maps to | Check on our data | Decision |
|---|---|---|---|---|---|---|
| 1 | Equal scores from unrelated teams mean a recipe whose result does not depend on the code: a limit, or the best point of a one-knob family tuned to its maximum | the board on #28: seven teams at exactly 0.16, four at 0.12 | | | | look for one-knob, training-free recipes |
| 2 | Uniform outputs on the forget images do not give 0.16 | `plateau.py` | uniform scores 0.000 on every gold with accuracy 0.58 or more; 0.06 to 0.29 only when gold is right on a third of the images (K=3) | loss log 100 = 4.6 with spread 0 overshoots the mean term unless gold's loss mean is above about 2.3 | grid of 25 golds | flat or uniform targets are dropped |
| 3 | **Dividing every logit by one temperature t (the last layer divided by t) is the one-knob, training-free recipe whose best value is 0.13 to 0.25 over most golds** | `plateau.py` | best t = 3 or 4; best score 0.141 (K=1, tau_g=2), 0.152 (K=1.25, tau_g=3), 0.165 to 0.167 (K=1.5 to 2, tau_g=5), 0.20 to 0.24 (K ≤ 0.5, tau_g 3 to 5) | the predicted class never changes, so accuracy, the utility gate and retain accuracy equal the original's. Retain falls with t through the control-set KL: t=1.5 → 0.93, 2 → 0.74, 3 → 0.47, 4 → 0.35 (0.87, 0.67, 0.44, 0.35 if the control images are unseen) | grid of 33 golds; retain from our logits | the cheapest candidate on the plateau is `scale_t3` |
| 4 | 0.12 reads as the same recipe one step of t off its best point | `plateau.py`, fixed-t table | K=1, tau_g=2: t=3 → 0.141, t=4 → 0.113; K=1.25, tau_g=3: t=4 → 0.152, t=3 → 0.074; K=0.5, tau_g=5: t=4 → 0.155, t=3 → 0.073 | a consistent reading, not proven | | none alone |
| 5 | **Our `x3_ext_m80` at 0.0063 conflicts with every gold on which our extrapolation dial works** | `plateau.py`, m80 column (the K=0.9 target, which matches m80's measured losses, times retain 0.97) | m80 predicted 0.017 or more on every gold with K ≥ 1, and 0.06 to 0.63 on the golds the dial assumed (K 1.2 to 1.6). It predicts 0.000 only on golds that are right on nearly every forget image and merely soft (K ≤ 0.5, tau_g 2 to 3) | there, any flip overshoots the loss spread, so the whole dial scores 0, while scaling scores 0.12 to 0.74 and forget-only tempering 0.26 to 1.0 | 0.0063 against the m80 column | `x4_e20` at 00:08 is the test: about 0 confirms it and the queue switches to scaling and tempering; clearly above 0.0063 keeps the dial |
| 6 | Same reading from published retrains: a retrained model is as accurate on the forgotten images as on test images, so gold is mostly right on our forget images | Deep Unlearn, https://arxiv.org/html/2410.01276v2, Table IV; AMUN, https://arxiv.org/html/2503.00917v3, Table 2; ReGUn, https://arxiv.org/html/2603.11210 | retrain forget accuracy 0.55 against test 0.55 (CIFAR-100, ResNet-18); 94.49 against 94.33 (CIFAR-10); 61.48 against 60.89 (Swin-T, Tiny-ImageNet) | a pretrained ViT-B/16 fine-tuned on a 100-class set is typically right on 85 to 95% of unseen images; gold, an average of several models, is also softer than any one of them | consistent with row 5 | the target is soft, mostly right outputs: tempering, not flipping |
| 7 | Tempering only the forget images keeps the forget side of row 3 without the control-set drift | `plateau.py`, temper column; `d3_ext_probe` (#14) for the spill | forget-only tempering at t=3, retain 1: 0.10 to 1.0 on golds K ≤ 1.25. With a third of the softening reaching unseen images (#14), retain about 0.75 to 0.9: 0.12 to 0.39 on the same golds | matches the group at 0.25 to 0.40 | the probes in `run_sixth.sh` measure the spill | GPU request: `run_sixth.sh` (forget-only fine-tunes toward orig / t, t = 2 to 5, two spill probes) |

**Cheapest candidate on the plateau: `scale_t3.pt`**, the original model with its last layer divided by 3 (`splice_head.py --scale 3`, CPU, seconds, no GPU). Predicted: 0.07 to 0.47 over the golds consistent with rows 5 and 6; 0 if gold is barely softer than orig / 2, in which case `scale_t2` is next. Then t=4 if t=3 scores. The gate cannot fail: the predicted class of every image is unchanged.

Limits: exact images only (the hidden altered copies are not modelled); gold is a two-parameter family along our cross-fit's direction plus softening; m80's retain is taken from the #14 probe.

### U2 and U3: first rows (00:05; the full sections follow by 01:30 and 02:30)

| # | Claim | Source | Number | Maps to | Decision |
|---|---|---|---|---|---|
| 8 | No public source of this metric was found | web, arXiv and GitHub code search for `mean_std_score`, `prediction_score` with `forget_score`, the task authors' names | no match | our proxy scorer (`plateau.py`, `dial_matrix.py`) is the reference | none |
| 9 | Closest relative, the NeurIPS 2023 challenge: moving past retrain fails like not moving | https://arxiv.org/html/2406.09073v1 | retrain forget accuracy 85.6%, original test accuracy 96.43% | the same two-sided idea | aim for gold's level, never for the most forgetting |
| 10 | Every top NeurIPS 2023 solution used the retain set; the part that carries over is perturb, then distil from the original model on other inputs | same, appendix | | protects `retain_score` without retained data | a later candidate: temper the forget images, distil the original on augmented or surrogate inputs |
| 11 | Forget-only, label-free smoothing: JiT | https://arxiv.org/html/2402.01401v4 | ViT, random forgetting: lr 0.01, sigma 0.5, about 1% retain accuracy lost | smooths the neighbourhood of each image, so it also reaches the hidden altered copies | candidate after the tempering runs |
| 12 | Forget-only gradient ascent and random labels damage nearby retained images | Dolgova and Stich (CISPA), https://arxiv.org/html/2605.31317 | average gap 7.7 and 12.2 against 2.4 for a local teacher (CIFAR-100) | the control set shares classes with the forget set | not used |

### U2: where gold is (01:05)

Method: the hypothesis ledger (`ledger.py` at `ef13498`) weighs a grid of gold hypotheses by our four public scores. Measurements on our own logits use `orig_exact.npy` (the original on the 3,500 forget images) and `xfit_extras.npz` (the cross-fitted prediction: the last blocks retrained without each image). Probes come from the seventh batch (`b23dad1` on `desk/unlearning`). Then the task statement.

| # | Claim | Source | Number | Maps to | Check on our data | Decision |
|---|---|---|---|---|---|---|
| 13 | **Gold is wrong on most forget images, and confident when it is wrong** | `ledger.py` on the scores of #12 | 80% of the weight on golds right on 33% of the forget images, with loss mean 7.6 to 8.1 and spread 8.8 to 9.1. 20% on removal golds: right on 64 to 67%, loss 3.8 / 5.7. Soft golds are out: `scale_t4` scored 0.0057, against 0.008 predicted before the score and 0.1 or more under soft golds | the forget images' target is far from the original, whose loss there is about 0 | `x3_ext_m80` 0.0063, `x4_e20` 0.0336, `scale_t4` 0.0057, `bias_full30` 0 | withdraws row 6's "gold is mostly right" for this task. Aim the fine-tunes at confident wrong outputs |
| 14 | The task statement says the same | README, Objective: the target covers "which wrong class it picks for each one" | | gold picks a wrong class on the forget images, and the KL term scores that class | consistent with row 13 | |
| 15 | Published retrains are right on forgotten images only when those images are a random sample of classes the model still sees | row 6's sources, all random forgetting. Under class-wise forgetting, a class with no training image is never predicted, since every sample only pushes its logit down | 0.55 against 0.55 (CIFAR-100), 94.49 against 94.33 (CIFAR-10) | our forget set is not random: nine classes lost all 130 images, 36 lost 35 to 80% (`analysis.md`) | 1,169 of the 3,500 forget images (33%) belong to the nine classes | the two readings of the counts (removal or extra images) decide where a third of gold is. The dials test them (#12) |
| 16 | **Which wrong class:** a model that did not train on an image errs towards a sibling class | `xfit_extras.npz` | right on 94.8% (nine classes 93.4%, thinned 95.5%). Most frequent errors: 19→21, 43↔42, 2→0, 35→38, 83→8, 32→29 | an upper bound on gold's accuracy: the cross-fit's early blocks saw every image, gold's saw none | the ledger's best golds extrapolate this step about 3 times (accuracy 0.33) | targets `orig + 3 to 4 × step` |
| 17 | Under removal, gold sends each image of the nine classes to its best class outside the nine | `orig_exact.npy` | 29→32 (51%), 30→32 (35%), 31→32 (27%), 33→32 (24%) or 23 (12%), 0→2 (49%), 38→36 (40%), 7→5 (34%), 6→65 (21%), 8→5 (16%). The cross-fit agrees on 59% of these images | a class dial sends each image exactly there: the four small-boat classes go to 32, fishing boats | the margin to that class has a median of 13.9 on the original and 10.1 on the cross-fit | `dial15`, `boats20` |
| 18 | The gate pins the original's validation accuracy A, if `bias_full30` scored 0 because of the gate | `ledger_inputs.json`, `ledger.unseen_flip` | 73.5 ≤ A < 76.9. `dial15` needs A ≥ 75.4 (44% of the band), `boats20` needs A ≥ 72.8 (all of it) | the validation set "comes from the training distribution" (README, Rules), so it probably holds the nine classes | flips taken from the cross-fit | `boats20` is the gate-safe test of removal |
| 19 | **The loss terms see single gold models, the KL term sees their mixture** | README, Metric: "mean(L_gold) and std(L_gold) are those quantities averaged over the gold models, and p_gold(x) is the average of their predicted distributions" | | the loss statistics are those of single, confidently wrong models (spread about 9). The KL target is softer wherever the gold models disagree, for example between sibling classes | the best golds have tau_g = 1 | a submission should be confident in loss and still spread over the siblings at temperature 2: targets tempered by 1.5 to 2, not more |
| 20 | Our fine-tunes fall short of their targets in loss, and they spill | seventh-batch probes | `k7_e3t2`: trained loss 1.80 / 1.51, against 4.38 / 4.54 for its target. Held-out images: KL 1.1 to 1.2, agreement 0.86 to 0.89. Spare classes: KL 0.74 to 0.79, agreement 0.97 to 0.99. A mixup anchor at weight 1 or 3 barely changes the spill | retain about 1 / (1 + 0.78) = 0.56 if the control set behaves like the spare classes | five probes | eighth batch (`4276fe9`): a K=4 target, which lands nearer gold's loss, and the removal learned on the forget images only |

**Where gold is, in short.** On the forget images, gold is confidently wrong on one to two thirds of them. It errs towards the sibling classes our cross-fit names, about three times further than the cross-fit goes. If the nine classes were removed, it errs on all of their images, mostly towards the nearest class outside the nine (fishing boats for the four small-boat classes). The open question is which of these two holds for the nine classes. It decides where a third of the forget set must go, and `dial15` and `boats20` test it.

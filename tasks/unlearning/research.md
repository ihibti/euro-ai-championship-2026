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

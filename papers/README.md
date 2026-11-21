# Reference papers (provided by the organizers)

PDFs are gitignored because arXiv licences generally do not allow redistribution. Download them with `make papers`, which uses `scripts/fetch_papers.sh` and checks SHA-256 hashes.

Detailed briefs are in `briefs/`. Each brief covers the problem, the method, the authors' findings with table references, limitations, the official code, reusable components, and ideas.

**No paper is assumed to define the challenge.**

In each paper summary below, "Authors' findings" reports only what the paper claims. "Hackathon use" holds our own ideas.

## 1. Pixel Seal: Adversarial-only training for invisible image and video watermarking

[arXiv:2512.16874](https://arxiv.org/abs/2512.16874) · Souček, Fernandez, et al. (Meta FAIR), Dec 2025 · [brief](briefs/2512.16874_pixel-seal.md)

- **Problem.** Multi-bit invisible watermarking of images and video has to be both robust and imperceptible. Pixel-loss proxies (MSE/LPIPS) leave visible artifacts, and training is unstable.
- **Method.**
  - The embedder is a U-Net and the extractor is ConvNeXt-v2; messages are 256 bits.
  - Training is adversarial-only: message BCE plus a patch discriminator, with no MSE/LPIPS.
  - A three-stage α schedule first learns robustness, then imperceptibility.
  - JND attenuation and a high-resolution training step that simulates inference.
  - Temporal pooling at inference time handles video.
- **Authors' findings.**
  - On Meta AI images, bit accuracy is 0.91 under combined attacks, and −log₁₀p is 50.3 (Table 1). PSNR is 48.9 (Table 2).
  - Ablations show that MSE at λ=1, a fixed α, or an early discriminator each collapse training (Table 3).
  - Our reading: it is not uniformly best. MBRS and Video Seal 1.0 win some compression columns.
- **Limitations.**
  - Training cost (600k steps × batch 256) is far beyond a hackathon.
  - The JND quality metric and the Meta AI images are internal.
  - The FPR guarantee is untested empirically.
  - Regeneration and removal attacks are not evaluated.
- **Code.** [facebookresearch/videoseal](https://github.com/facebookresearch/videoseal) (MIT). `videoseal.load("pixelseal")` auto-downloads a ~1.24 GB checkpoint, which we have not downloaded. Install from Git, because the PyPI release predates Pixel Seal.
- **Hackathon use (ours).**
  - Evaluate public checkpoints under a deterministic attack suite.
  - Check the false-positive rate empirically against the binomial p-value.
  - Test training-free regeneration attacks.

## 2. Model Equality Testing: Which Model Is This API Serving?

[arXiv:2410.20247](https://arxiv.org/abs/2410.20247) · Gao, Liang, Guestrin, ICLR 2025 · [brief](briefs/2410.20247_model-equality-testing.md)

- **Problem.** Detect whether a black-box API serves the same output distribution as reference weights. Providers may have changed the model by quantization, watermarking, fine-tuning or swapping. The paper frames this as a two-sample test over (prompt, completion) pairs.
- **Method.**
  - The statistic is MMD² with a prompt-masked kernel; the Hamming kernel over padded token or character sequences works best.
  - p-values come from a simulated null (resampling from the reference) or from permutations.
- **Authors' findings.**
  - The Hamming MMD test reaches a median power of 77.4% with about 10 samples per prompt (Abstract, Fig. 2).
  - Power is weak against fp16.
  - In the audit, 11 of 31 commercial endpoints served a distribution different from the reference (Fig. 1).
- **Limitations.**
  - Any setup mismatch looks like a distortion: chat template, top_p, Unicode handling.
  - Hamming is position-wise and only a pseudo-metric.
  - Building the reference needs local inference at matched precision.
  - The official code has memory and correctness issues at scale (see the brief).
- **Code and data.** [i-gao/model-equality-testing](https://github.com/i-gao/model-equality-testing), installable with `pip install model-equality-testing`. The licence is ambiguous (MIT placeholder text). The dataset is 37 GB on Google Drive and we have not downloaded it.
- **Hackathon use (ours).**
  - Map how sensitive the test is to inference-stack changes on small local models.
  - Try alignment-robust kernels.
  - Try sequential (e-value) auditing.

## 3. Towards Unbounded Machine Unlearning (SCRUB)

[arXiv:2302.09880](https://arxiv.org/abs/2302.09880) · Kurmanji, Triantafillou, Hayes, Triantafillou, NeurIPS 2023 · [brief](briefs/2302.09880_scrub.md)

- **Problem.** Remove the influence of a forget set from a trained network more cheaply than retraining. The paper treats "forgetting" as application-specific:
  - removing biases: maximise forget error;
  - resolving confusion: fix mislabels;
  - user privacy: match a retrained model under membership inference.
- **Method.**
  - SCRUB uses teacher–student distillation with alternating epochs: maximise KL(teacher‖student) on the forget set, then minimise KL + CE on the retain set.
  - SCRUB+R rewinds to a checkpoint whose forget error matches a same-distribution validation set, so that over-forgetting does not leak membership.
- **Authors' findings.**
  - SCRUB is consistently among the top methods for forget quality and utility across the three applications.
  - On ResNet CIFAR-10 selective forgetting, rewinding lowers basic MIA accuracy from 78.0 to 58.7 (Retrain: 54.0; Table 2).
  - Runtime is 5–8× faster than retraining on CIFAR-10 (Table 6).
- **Limitations.**
  - No guarantees, and the min-max optimisation can oscillate.
  - α/γ per experiment are unreported; the code defaults differ from the paper's sweep.
  - The basic MIA is weak.
  - Only small CNNs are tested.
  - One dataset (Lacuna) is derived from face images, so we avoid it under the no-personal-data constraint.
- **Code.** [meghdadk/SCRUB](https://github.com/meghdadk/SCRUB). The academic, non-commercial licence applies. The repository is mostly notebooks, and no checkpoints are provided.
- **Hackathon use (ours).**
  - A CPU-scale reproduction with Retrain as the reference.
  - LiRA-lite evaluation.
  - A bounded max-step variant.

## Cross-cutting observations (ours)

All three papers verify a property of a model or its outputs from limited evidence. Each one ends in a **detection decision with a controlled false-positive rate**:

| Paper | Decision | Test statistic | False-positive control |
|---|---|---|---|
| Pixel Seal | Is this image watermarked? | bit matches | Binomial(n, ½) p-value |
| MET | Is the API serving the same model? | MMD² | permutation or simulated null |
| SCRUB | Was this example a training member? | per-example loss or confidence | MIA AUC, TPR at low FPR |

This shared evaluation layer is the only part we have implemented so far (`src/euroai/metrics.py`, `stats.py`, `kernels.py`). It is task-independent.

The organizers' Paris slides show a submission of `id,label` probabilities in [0,1]. That format fits any of these detection framings, but it may not apply to this event.

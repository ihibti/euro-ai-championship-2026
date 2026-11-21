# Candidate baselines, by challenge type (not implemented)

The challenge is still **TBD**. This page is a menu to pick from once the task is announced. It is not a plan.

Every entry is our own suggestion. Paper references point to where each idea comes from; the briefs are in `papers/briefs/`.

## Rule zero, whatever the task

1. **Trivial baseline first.** Start with constant, random or prior predictions, written in the exact submission format. Score it with the official scorer, or with a local copy of the metric (`euroai.metrics`), and submit it once if that is allowed. This checks the whole pipeline end to end: data loading, the `id` mapping, the CSV and the upload.
2. **Strongest cheap baseline.** Next, build the strongest baseline that runs in under an hour on one node. The best-performing public checkpoint usually beats anything trained during a hackathon.
3. **Record every run.** Each run gets a config, a seed, a manifest and metrics. `scripts/run_experiment.py` produces all four.
4. **Report uncertainty.** Give every metric a confidence interval (`euroai.stats.bootstrap_ci`), not just a point estimate.

## A. Membership inference, or unlearning evaluation (SCRUB)

| Level | Baseline | Notes |
|---|---|---|
| 0 | Random scores | AUC ≈ 0.5 sanity check |
| 1 | Loss-threshold or confidence attack | `euroai.metrics.best_threshold_accuracy`, `roc_auc`, `tpr_at_fpr`. Needs per-example losses only. |
| 2 | Logistic regression on clipped losses | SCRUB §11.1 "Basic MIA" |
| 3 | LiRA-lite with 8–32 shadow models | SCRUB §11.2. Compute-bound, so report TPR@1%FPR. |

## B. Producing an unlearned model (SCRUB)

| Level | Baseline | Notes |
|---|---|---|
| 0 | Original model (no unlearning) | Upper bound on leakage |
| 0 | Retrain on the retain set | Gold reference, if the compute budget allows |
| 1 | Fine-tune on the retain set only | Cheap |
| 1 | NegGrad+ | β·CE_retain − (1−β)·CE_forget (Eq. 4) |
| 2 | SCRUB, then SCRUB+R | Few max-epochs, a learning-rate decay schedule, and rewinding for privacy |

## C. Watermark detection, embedding or removal (Pixel Seal)

| Level | Baseline | Notes |
|---|---|---|
| 0 | Identity (no attack, or no watermark) | Defines the bit-accuracy and PSNR baselines |
| 1 | Public extractor plus binomial p-value thresholding | `euroai.metrics.bit_accuracy`, `binomial_tail_pvalue`. Calibrate the threshold on clean images. |
| 2 | Deterministic attack suite | JPEG, crop, resize, brightness and blur at the Fig. 8 parameters, with a fixed seed |
| 3 | Training-free regeneration (VAE round-trip) | GPU required. Report the trade-off against PSNR. |

## D. Model or API auditing, and distribution-shift detection (MET)

| Level | Baseline | Notes |
|---|---|---|
| 0 | Exact-match or one-hot statistic | Weak, but a useful sanity check |
| 1 | Hamming MMD with permutation p-values | `euroai.kernels.prompt_hamming_gram` + `euroai.stats.mmd_permutation_test` |
| 2 | Log-likelihood feature test | Needs white-box access to P |
| 3 | Better kernels, or sequential testing | Kernel selection must not touch the evaluation pairs |

## E. Generic supervised task (`id,label` probabilities, as in the Paris slides)

| Level | Baseline | Notes |
|---|---|---|
| 0 | Class-prior constant | Pipeline check |
| 1 | Linear model on simple features, or a frozen pretrained encoder plus logistic regression | CPU-friendly. Needs the `sci` extra. |
| 2 | Fine-tune a small pretrained model on 1 GPU node | Needs the `torch` and `hf` or `vision` extras |

## Compute notes

- **Node limits.** One JURECA-DC node has 4× A100 40 GB. Jobs get at most 24 h on `dc-gpu` and at most 2 h on `dc-gpu-devel`.
- **No internet on compute nodes.** Download all weights and data on a login node beforehand (see `jsc/README.md`).
- **Mind the queue.** Prefer baselines that run in under an hour, because queueing and the reservation are shared.

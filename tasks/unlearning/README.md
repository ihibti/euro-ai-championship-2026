# Task 3: Unlearning

Source: the organizers' task description "Unlearning" (European AI Championship Paris 2026). Supervisors: Adam Dziedzic and Franziska Boenisch. Developed by Maitri Shah, Jack Le and Vincent Hanke. Rewritten here, with numbers and formulas copied exactly.

Owner: TBD. Server task id: `31-unlearning`.

## Goal

You get a pretrained image classifier (a ViT-B/16) trained on a custom 100-class image dataset, and a designated subset of **3,500 training images** that the model must forget. Submit an unlearned model, derived from the provided checkpoint, that behaves as though it had never been trained on those 3,500 images while continuing to perform well on the rest of the dataset.

## Challenge

- A model damaged on the forget set, for example by naive gradient ascent, can show low accuracy on those images without having forgotten them in any meaningful sense. Its behavior on them can still be clearly told apart from a model that never saw them.
- The objective is to match a model retrained from scratch with the forget set excluded entirely. That covers how confident it is on the forget images, and also **which wrong class it picks for each one**, on the exact images and on slightly altered versions of them.
- No labeled training data is provided beyond the forget-set images themselves, so retraining on a clean version of the dataset is impossible. The method has to work from the original model and the forget set alone.

A successful submission:
- brings the model's behavior on the forget set, and on altered versions of those images, close to a model that never saw them during training;
- preserves classification accuracy and output behavior on the rest of the dataset;
- keeps a reasonable baseline accuracy on a general validation set. Damaging the model overall to inflate the forgetting signal is not valid unlearning.

## Metric

Three models appear in every formula:
- **orig**: the given `model.pt`, trained on all the data, including the forget images.
- **gold**: retrained from scratch with the forget images excluded entirely; the average of several independently retrained models. This is the behavior to reproduce.
- **sub**: your submission.

For a model m and an image x:
- `L_m(x)` is the cross-entropy loss of m on x, using the image's true label.
- `p_m(x)` is the class distribution predicted by m on x: softmax of the logits divided by a temperature of 2.
- `acc_m(control)` is the accuracy of m on the control set.
- For gold, `mean(L_gold)` and `std(L_gold)` are those quantities averaged over the gold models, and `p_gold(x)` is the average of their predicted distributions.
- All means and stds are taken over the images of the set being scored.

**Utility gate:** at least **70%** accuracy on a separate held-out validation set, not provided. Below it, the submission is disqualified: forget and retain scores are both 0.

**Forget quality** is measured on the forget images and on hidden altered copies of them. The two groups count equally. For each group:

```
d_sub  = |mean(L_sub)  − mean(L_gold)| + |std(L_sub)  − std(L_gold)|
d_orig = |mean(L_orig) − mean(L_gold)| + |std(L_orig) − std(L_gold)|

mean_std_score   = 1 − clip(d_sub / d_orig, 0, 1)
prediction_score = 1 − clip( mean KL(p_gold(x) ‖ p_sub(x)) / mean KL(p_gold(x) ‖ p_orig(x)), 0, 1 )
```

Each score is averaged over the exact images and the altered copies, and then the two are multiplied:

```
forget_score = mean_std_score × prediction_score
```

0 means no unlearning was performed (sub = orig). 1 means the submission matches the gold models. Matching only the average loss is not enough.

**Retain quality** combines accuracy and output-distribution similarity on a held-out control set:

```
retain_accuracy_score = clip(acc_sub(control) / acc_orig(control), 0, 1)
retain_kl_similarity  = 1 / (1 + mean KL(p_orig(x) ‖ p_sub(x)))
retain_score          = retain_accuracy_score × retain_kl_similarity

final_score = forget_score × retain_score
```

The KL term penalizes prediction drift from the original model even when accuracy is preserved. Because the two scores multiply, neither can be sacrificed to inflate the other.

## Data

The Hugging Face dataset `SprintML/unlearning` is public:

- `model.pt` (343,617,075 bytes): a weights-only PyTorch state dict for timm `vit_base_patch16_224`, with 100 classes and LayerScale init 1e-4.
- `forget_images.npz` (526,848,266 bytes): key `images`, a uint8 array of shape `[3500, 224, 224, 3]`, channels last, raw pixels in 0–255, not normalised. **Labels are not included.**
- `task_template.py`: model builder, the server's preprocessing, a prediction helper and an `unlearn()` stub. Running it saves the unchanged model as a baseline `submission.pt`, which scores about 0.
- `submission.py`: the code to submit your model.

The small files are copied unmodified into `upstream/`. The large files are not committed; `make data TASK=unlearning` downloads them into `data/unlearning/`. On JURECA they are already in the task folder (below).

**Preprocessing** (what the server does): float32 `[B, 3, 224, 224]`, `x = (pixels / 255 − 0.5) / 0.5`. Use `preprocess()` from the template and do not resize or normalise a second time. The server runs the forward pass in eval mode under float16 autocast; this does not change the weights you submit.

## On JURECA

- `/p/scratch/training2666/hibti1/Begg/unlearning/`: a uv project with a Python 3.12 `.venv` holding torch 2.15.0+cu132 and timm 1.0.31, plus `model.pt` and `forget_images.npz`.
- Run GPU work with `jsc/jureca_gpu.sbatch.template` and `TASK_DIR` set to this folder (see `jsc/README.md`, Submitting a job): one A100 40 GB per job (`gpu:1`, 8 CPUs) on reservation `cispahack`, which ends 2026-11-22 14:00. Compute nodes have no internet.
- A ViT-B/16 state dict is about 345 MB. Submitting it from a laptop means copying it there first.

## Submission

Submit only the unlearned model's state dict, matching the exact architecture, layer names and parameter shapes of the provided model:

- a `.pt` or `.pth` file that loads with `strict=True` into the architecture from `task_template.py`;
- float32 weights only (any other floating-point dtype is rejected), with no NaN or infinite values;
- at most 450 MB (a float32 ViT-B state dict is about 345 MB);
- if a method modifies or adds to the architecture, for example adapters or low-rank PEFT-style layers, those changes must be merged back into the original weights before submission.

`make submit` checks dtype, NaN/Inf and size locally, plus the strict load when torch and timm are installed.

## Scoring and leaderboard

- Scoring uses held-out sets that are not provided: the hidden altered copies of the forget images, a control set for the retain score, and a validation set for the utility gate. All come from the training distribution.
- Scoring data is split deterministically into a public and a held-out portion, based on a hash of each sample's identifier; an altered copy follows its source image. About 30% is used for the public leaderboard, and the remaining 70% for the final ranking.
- Leaderboard: http://34.147.229.55/leaderboard_page. Our team name there is **Begg**. It shows each team's best result; a lower score does not replace it.

## What can be scored where

| | Locally | Only on the server |
|---|---|---|
| Metric | Proxy only: no labels, no retain set, no gold model. The owner defines the proxy first. | forget score, retain score, utility gate |
| Signals | orig's predictions on the forget images (pseudo-labels), loss and confidence statistics, behavior under your own image alterations | |

## Budgets

- Submission limit: not stated. Treat it like model equality: one submission per 30 minutes, failures included. `make submit` enforces this.

## References

- Meghdad Kurmanji, Peter Triantafillou, Jamie Hayes, Eleni Triantafillou. "Towards Unbounded Machine Unlearning." NeurIPS 2023. https://arxiv.org/abs/2302.09880. Brief: `papers/briefs/2302.09880_scrub.md`
- Vikram S Chundawat, Ayush K Tarun, Murari Mandal, Mohan Kankanhalli. "Zero-Shot Machine Unlearning." IEEE Transactions on Information Forensics and Security 2023. https://arxiv.org/abs/2201.05629
- Chongyu Fan, Jiancheng Liu, Yihua Zhang, Eric Wong, Dennis Wei, Sijia Liu. "SalUn: Empowering Machine Unlearning via Gradient-based Weight Saliency in Both Image Classification and Generation." ICLR 2024. https://arxiv.org/abs/2310.12508
- Zhang, Lin, Bai, Mei. "Negative Preference Optimization: From Catastrophic Collapse to Effective Unlearning." COLM 2024. https://arxiv.org/abs/2404.05868

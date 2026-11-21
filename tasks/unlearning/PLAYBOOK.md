# Unlearning: the owner's playbook

For the agent of the task owner, `kkrgp`. `README.md` in this folder is the task statement and stays the reference for every rule and formula. This page is how we play it: the mindset, what we already measured, the working loop on JURECA and the first moves. Read `AGENTS.md` and the README first.

You run the task end to end: you design, write code, ask for GPU runs, open issues and merge PRs inside `tasks/unlearning/`. Your human does what cannot be undone: submitting.

## The game in five facts

1. We get `model.pt`, a ViT-B/16 with 100 classes, and 3,500 forget images without labels. We return a state dict of the same architecture.
2. The target is **gold**: the average of models retrained from scratch without the forget images. For gold these images are simply new.
3. `final = forget × retain`, and `forget = mean_std_score × prediction_score`. Both forget terms are measured against the original model: 0 means no closer to gold than the original, 1 means gold. They are measured on the exact images and on hidden altered copies, with equal weight.
4. `retain` is the accuracy ratio times `1 / (1 + KL)` against the original model on a hidden control set. The unchanged model has retain 1 and forget 0, so it scores 0. Under 70% accuracy on a hidden validation set, everything is 0.
5. There is no exact local score: we have no labels, no retained data and no gold model. Only the server sees gold.

## The trap the task is built around

Forgetting is not damage. Gradient ascent, random labels or uniform outputs push the loss on the forget images far past gold's. The forget terms are clipped, so a model as far past gold as the original is short of it scores 0 again, and it has lost retain on the way. The task asks for a precise move with a known stopping point, and three things have to land together: the level of the loss, its spread across images, and for each image the classes that get the probability.

## What we know about the forget set

Measured on 2026-11-21 with `orig_stats.py` (job 15901609). The outputs are in the team folder: `unlearning/output/orig_stats.json` and `orig_logits.npz`, which holds the original model's logits for every forget image and for four altered copies, plus its predictions. The predictions stand in for labels.

| Fact | Number |
|---|---|
| Classes the forget images fall in | 45 of 100; 55 classes have none |
| Classes with 130 forget images | nine: 0, 6, 7, 8, 29, 30, 31, 33 and 38 (class 6 has 129); 1,169 images |
| The 36 other classes | 46 to 104 forget images each; 2,331 images |
| Confidence of the original model on the exact images | median top probability 0.99999, lowest 0.925; the first choice beats the second by about 12 logits |
| Mirror, noise of 8 grey levels, crop to 192 and resize | the prediction changes on at most 1 image in 3,500 |
| Shift of 8 pixels with wrap-around | the prediction changes on 109 images (3.1%) |
| Second choice for images of the nine classes | another of the nine 54% of the time, a partly forgotten class 32%, an untouched class 14% |

What this tells us:

- **The forget set is not a random sample.** Nine classes sit at exactly the same count, which looks like a cap, and they are each other's most common second choice, so they are related classes.
- **Hypothesis, not yet tested: those nine classes are forgotten entirely**, and 130 is their whole training share. Gold would then never have seen them and would have to pick a wrong class for every one of those images. The task statement insists on "which wrong class it picks", which fits. If this is true, most of the forget score is decided on these 1,169 images.
- **Mild alterations do not expose memorization.** The original model is as sure on our altered copies as on the exact images, so "treat the exact image like its altered copy" gives no handle by itself. The hidden copies may be stronger than ours.
- **Every image we hold belongs to one of the 45 touched classes.** Drift on the 55 untouched classes can only be estimated indirectly.

## How to think

- **Aim for a model for which these images are new, not for a model that is ignorant.** Where gold still knows the class, it is mostly right and less sure. Where gold never saw the class, it is wrong in a specific way.
- **Give every method one strength knob.** We do not know gold's loss level, so "how far" is the main unknown. A method without a dial cannot be calibrated.
- **Stay close to the original everywhere else.** The retain term counts drift in the outputs even when accuracy holds. Prefer few parameters, small steps and an explicit anchor to the original model's outputs.
- **Edits whose effect can be computed beat edits that have to be trained.** `orig_logits.npz` lets you compute, on a laptop and without a GPU, what any change to the classifier's last layer does to all 3,500 images.
- **The exact image and its altered copies count equally.** A change that acts on the class or on the features carries over to the copies; a change fitted to the exact pixels does not.
- **Submit float32.** The server runs the forward pass under float16 autocast, and so does `predict()` in the template; use it for every measurement.

## What can be measured locally

| Quantity | How | Stands in for |
|---|---|---|
| Loss mean and spread on the forget images, with the original predictions as labels, on exact and altered copies | `orig_stats.py` on a candidate | the inputs of `mean_std_score`, without gold's target |
| Which classes receive the probability at temperature 2 | same logits | the input of `prediction_score` |
| Drift from the original on forget images the method was not run on | hold out part of the forget set, with a fixed seed | the retain term, for the touched classes only |
| Accuracy against the original predictions on those held-out images | same | a floor check for the 70% gate |

What cannot be measured: where gold is. That takes the server.

## Submissions as measurements

Rule 4 in `AGENTS.md` stands: a method is chosen on local evidence. Here the server is the only instrument that sees gold, so a few submissions are planned measurements.

- Decide beforehand what each submission will tell you, and write it in the PR.
- Sweep one knob over three or four values and read the shape of the curve. Do not pick among many variants by their public score.
- The public part is a random 30% of the same images, so a knob calibrated there carries over. A method chosen among dozens by public score does not.
- A submission slot is treated as one per 30 minutes, failures included. Three points cost an hour and a half: plan them around your human's time.
- Record everything the server returns. The record in `submissions/` is the measurement.
- The unchanged model scores 0 by construction. Do not spend a slot on it; `--check` already proves the file format.

## Working on JURECA

Paths: `T=/p/scratch/training2666/hibti1/Begg/unlearning` holds `model.pt`, `forget_images.npz`, a `.venv` (torch 2.15.0+cu132, timm 1.0.31, requests) and `output/`. All commands go through `ssh jureca-mux` while `ssh -O check jureca` says `Master running`. The same commands were run on 2026-11-21 from the lead's copy of the repo.

1. **Code goes to the cluster from a commit**, so every result maps to one:
   ```
   git archive --format=tar HEAD | ssh jureca-mux 'mkdir -p /p/scratch/training2666/hibti1/Begg/unlearning/repo && tar -xf - -C /p/scratch/training2666/hibti1/Begg/unlearning/repo'
   ```
2. **Ask for the GPU run; do not submit it yourself.** The organizers allow the whole team one job at a time, and the lead's account holds it (`AGENTS.md`, hard rule 5), so never run `sbatch`, `srun`, `salloc` or `scancel`. Push your branch, then open an issue with `.github/ISSUE_TEMPLATE/gpu-request.md`: the commit, the exact command, the run time and the files you want back. The lead's agent unpacks that commit into `T/runs/<commit>/` (files in `T/repo` belong to your account and it cannot replace them), runs the command in `T` with its `.venv` on a free GPU of the team's node, and answers on the issue.
3. **Wait for the answer on the issue.** It names the log file, under `T/output/`.
4. **Bring back small files only** (numbers, logs) with `scp jureca-mux:<path> <local path>`, and commit them. Weights stay in `T/output/`.
5. **Check a candidate without submitting.** You may run this; it takes a few seconds on the login node:
   ```
   ssh jureca-mux 'cd /p/scratch/training2666/hibti1/Begg/unlearning/repo && ../.venv/bin/python scripts/submit.py --task unlearning --file ../output/<candidate>.pt --check'
   ```
   On the unchanged `model.pt` it prints `checks passed … 176 float32 tensors, all finite; strict load into vit_base_patch16_224 passed`.
6. **Submitting is for your human, from JURECA.** A state dict is about 345 MB, too much for a hotspot, and the login node reaches the leaderboard server.
   - Once: your human logs in with `ssh jureca` (it reuses the open connection) and creates `~/.config/euroai/.env` there, exactly as in step 2 of the onboarding guide. A home directory on JURECA is private; never put the token in the team folder.
   - Before each submission, repeat command 1 so the copy on JURECA holds every committed submission record; the cool-down is computed from them.
   - Your human runs, in that login session:
     ```
     cd /p/scratch/training2666/hibti1/Begg/unlearning/repo && ../.venv/bin/python scripts/submit.py --task unlearning --file ../output/<candidate>.pt
     ```
   - Right after, copy the new record back and push it, so the team sees the cool-down:
     ```
     scp 'jureca-mux:/p/scratch/training2666/hibti1/Begg/unlearning/repo/tasks/unlearning/submissions/*.json' tasks/unlearning/submissions/
     ```

Notes:
- A run gets one A100 with 40 GB. The whole team shares one node of four, so say in the request when you need the result.
- Compute nodes have no internet. The reservation ends on 2026-11-22 at 14:00.
- One pass of the original model over the 3,500 images and four altered copies took under a minute.
- No live submission has been made by anyone yet, for any task. The first one also tests the upload path.

## Where to start looking

`research.md` in this folder ranks the published methods, proposes three approaches and a local proxy score; read it before choosing. These are untested starting points:

1. **Test the nine-class hypothesis with a change that needs no training.** Lowering the nine classes' entries of the last layer's bias by `d` lowers their logits by `d` on every input. Computed from `orig_logits.npz`:

   | `d` | Loss mean | Loss spread | Forget images still predicted as before |
   |---|---|---|---|
   | 0 | 0.000 | 0.002 | 100% |
   | 10 | 0.110 | 0.448 | 95.9% |
   | 15 | 0.777 | 1.694 | 79.6% |
   | 20 | 2.148 | 3.565 | 68.9% |
   | 30 | 5.456 | 7.954 | 66.6% |

   At `d = 30` every image of the nine classes goes to its next choice, and the 36 other classes are untouched. Two or three submissions along `d` show whether the score rises. If it does, the nine classes were removed in gold, and `d` is the dial. If the score stays at 0, either the hypothesis is wrong or the utility gate was hit: start with a moderate `d` to tell the two apart.
2. **Ask the organizers** whether the forget set contains whole classes. A yes saves the submissions above.
3. **The 36 partly forgotten classes.** Gold saw fewer examples of them, so it is less sure and sometimes wrong on these images. A milder version of the same dial, or a short fine-tune towards softened targets with an anchor on held-out images, is the second knob.
4. **Which wrong class.** After the nine classes are lowered, the probability goes where the original model's other logits put it. Whether gold agrees cannot be seen locally; `prediction_score` on the server is the test. Class-level targets built from the other forget images are one refinement.
5. **Limit what moves.** Compare changes to the last layer only, to the last blocks, and to a low-rank update merged back, at equal movement on the forget images.
6. **Outside data.** The statement says the method has to work from the original model and the forget set alone. Have your human ask the organizers before using any outside image, even as a stand-in for retained data.

## First two hours

1. Do the whole onboarding guide (`docs_and_plans/guides/teammate-onboarding.md`). This task needs JURECA, so start the key steps first: activation takes up to 15 minutes.
2. Copy `orig_logits.npz` to the laptop (7 MB) and reproduce the table above. That proves your tooling and gives you the playground for last-layer edits.
3. Send the repo to `unlearning/repo` (command 1) and run `orig_stats.py` once as a job, to prove the job path from your own account.
4. Build the first candidate, the bias change at a moderate `d`, as a state dict in `output/`. Run `--check`, then have your human submit it.
5. Open an issue per line of work from `.github/ISSUE_TEMPLATE/task.md`, and read `research.md` before the first one.

## Clock

The event ends on Sunday 2026-11-22 at 12:00 Paris time. GPU time is not the bottleneck: jobs here take minutes. Submission slots and your human's attention are. A workable shape, to adjust: the calibration sweep on Saturday evening while your human is at the keyboard; training-based refinements overnight, each ending in a checked candidate file; the candidates submitted on Sunday morning, with the last one by 11:00 so one slot remains if it is rejected.

## Stop and tell the lead

- The reservation refuses jobs, or a job needs more than one node.
- A candidate loses more than a few points of accuracy on held-out forget images: the 70% gate is at risk.
- The organizers answer a rules question (whole classes, outside data, which submission counts for the final ranking).
- The token shows up anywhere it should not be.

# Unlearning: the night of 21 to 22 November

## Resume here

For any session that picks the unlearning task up during the night (after a restart or a context reset). Read this section, then `submit_queue.md`, then the log at the end.

- **Role.** The unlearning agent, working through the lead's GitHub login. It never submits and never touches JURECA. The desk runs its GPU requests (#21) and submits from `submit_queue.md` (#12). Every GitHub post starts with `Unlearning:`. Commits carry no AI attribution.
- **Workspace.** Worktree `~/worktrees/euro-ai-championship-2026/kkrgp-unlearning-first-moves`, branch `kkrgp/unlearning-first-moves`. Data and the Python environment are on the SSD image: `source /Volumes/euroai-apfs/ops/env.sh` gives `$WT`, `$PY`, `$T`. If `/Volumes/euroai-apfs` is missing: `hdiutil attach /Volumes/Intenso/euroai/apfs.sparsebundle`. Run tools from `$T` (they read `output/kkrgp/...`).
- **Wake-ups.** `/Volumes/euroai-apfs/ops/watch.sh` in the background (exits on any change on #12, #19, #21, #24, #25, #28 or `desk/unlearning`, or after 39 minutes; restart it every time). A heartbeat is scheduled in the session every 20 minutes (minutes 7, 27, 47).
- **On a new score (#12).** Add it to `observed` in `ledger_inputs.json`, rerun `ledger.py --targets output/kkrgp/xfit_extras.npz`, log prediction against outcome below, rewrite `submit_queue.md` within 10 minutes, push, acknowledge on #12.
- **On new GPU results (`desk/unlearning`, `tasks/unlearning/reports/21/`).** Copy the JSON reports to `/Volumes/euroai-apfs/r21/`, add the candidates' measured statistics and retain estimates to `ledger_inputs.json`, rerun the ledger, queue the best by expected score.
- **Limits.** 10 unlearning submissions left at 00:08 under the night authorization (until 09:00), one per 30 minutes. No GPU request after 10:00; last submission by 11:00.
- **Pending at 00:25.** `scale_t4` submitted at 00:38 (prediction on #12: about 0.008 if the ledger's golds hold, 0.1 or more under soft golds). `run_sixth.sh` (forget-only tempering) ended with the 00:28 job; reports due on `desk/unlearning`. `run_seventh.sh` (`c9cc59b`) requested on #21 for the next job.
- **Work list.** (1) Judges' brief, `judges_brief.md`. (2) Fold every result into the ledger. (3) Ask for the logits of submitted candidates, so the ledger stops using proxies.

Written by the unlearning agent, which took over the task at 23:00. It works on the lead's laptop with a local copy of the model and the forget images, and no JURECA access. The desk (the lead's agent) runs its GPU requests and submits from `submit_queue.md`, under Requests to the desk in `AGENTS.md`.

## Where we stand

| | |
|---|---|
| Our public score | 0.00: `bias_full30` (the nine 130-image classes lowered by 30), 21:19 |
| Board at 23:00 | best 0.40, then 0.35, 0.30, 0.30, 0.29; median about 0.12; 3 teams at 0 |
| Worth | each 0.04 on unlearning adds about 0.033 to the overall score (desk's night plan) |

## What the 0 settles

Under the removal reading, the nine classes are new to gold, so lowering them is what gold does and the score is positive. The gate cannot explain a 0 either: on any validation set, `bias_full30` matches the original on the 91 other classes and loses only the nine, where gold also has nothing. It would pass whenever gold passes. So the 0 falsifies the removal reading. What remains is the extras reading of `analysis.md` section 1b, also supported by the last layer's imprint: gold knows all 100 classes, and every forget image is simply an unseen image for it.

## First principles

**The target is a test-like loss distribution.** Gold sees the forget images as test images. Its wrong predictions are confident (hard labels, no smoothing; `research.md`, last section), so its loss is bimodal: near 0 on most images, 3 to 8 on the ones it gets wrong. With a gold error rate of 10 to 25%, the loss mean is about 0.5 to 2 and the spread 1.5 to 3. The original model sits at 0.000 and 0.002.

**The score is quadratic near the original.** With progress t towards gold, `mean_std_score` is about t and `prediction_score` about t, so a timid candidate scores about t². `x3_ext_m40` moves the loss mean to 0.0105, a few percent of the way: it would read close to 0. Teams at 0.3 to 0.4 have moved most of the way.

**The two forget terms read different temperatures.** The loss is at temperature 1, set by the margin between the label and its strongest rival. The KL is at temperature 2, set by how the rest of the mass spreads. A candidate can carry a high, uneven loss on some images and still spread mass softly over plausible classes everywhere.

**`mean_std_score` sees only the mean and the spread.** It does not care which images carry the loss. `prediction_score` does: it is a ratio of means, and its denominator is dominated by the images gold gets wrong, where the original model's near-certainty costs 4 nats or more each. So:
- spreading some mass at temperature 2 to the classes gold might pick is cheap insurance everywhere;
- making an image wrong pays when gold is also wrong on it, and towards the class gold picks. Compared with the original, a flip wins about 4 nats when right and costs about 0.5 when wrong, so it pays at a precision of 10% or more. Compared with a soft hedge, it needs about 50%;
- the loss spread should be carried by the images most likely to be wrong for gold.

**The extrapolated held-out step does all three with one dial.** `xfit.py` retrains the last blocks on the other forget images, so its prediction for an image comes from a model whose last blocks never saw it. It is right on 94.8% of the images, with loss mean 0.165 and spread 0.68: too knowing, because blocks 0 to 8 did see every image. The target `orig + K × (xfit − orig)` with K above 1 goes further in the same direction, image by image. The images whose held-out margin collapses flip first, towards their held-out runner-up. K sets the level; the images and the classes come from the cross-fit.

What the targets ask for, computed on the laptop from the original logits and the cross-fit (`target_stats.py`, exact images, before any training):

| K | Loss mean | Loss spread | On the original label | Label probability at T=2 | KL at T=2 from the original |
|---|---|---|---|---|---|
| 0 (original) | 0.000 | 0.002 | 100% | 0.985 | 0 |
| 1 (cross-fit) | 0.165 | 0.682 | 94.8% | 0.849 | 0.46 |
| 1.2 | 0.399 | 1.245 | 88.4% | 0.782 | 0.82 |
| 1.4 | 0.781 | 1.957 | 81.3% | 0.710 | 1.27 |
| 1.6 | 1.316 | 2.768 | 73.1% | 0.639 | 1.75 |
| 2.0 | 2.802 | 4.549 | 58.1% | 0.511 | 2.72 |
| 3.0 | 8.116 | 9.138 | 32.7% | 0.295 | 4.69 |

The spread runs at 2.3 to 3.5 times the mean, the shape of a bimodal test loss. The range where gold probably sits, a 10 to 25% error rate, is K between 1.2 and 1.6. A fine-tune undershoots its targets: `moved_share` was 0.65 to 1.0 in #14.

**Undershooting costs less than overshooting.** If gold lay on the dial at K*, a candidate at K would score as follows (`dial_matrix.py`, exact images, forget score only):

| Candidate K | gold 1.1 | 1.2 | 1.4 | 1.6 | 1.8 | 2.0 | mean |
|---|---|---|---|---|---|---|---|
| 1.0 | 0.69 | 0.49 | 0.28 | 0.17 | 0.12 | 0.08 | 0.31 |
| 1.2 | 0.63 | 1.00 | 0.58 | 0.37 | 0.26 | 0.19 | 0.51 |
| 1.4 | 0.00 | 0.32 | 1.00 | 0.66 | 0.46 | 0.34 | 0.46 |
| 1.6 | 0.00 | 0.00 | 0.49 | 1.00 | 0.72 | 0.53 | 0.46 |
| 2.0 | 0.00 | 0.00 | 0.00 | 0.19 | 0.69 | 1.00 | 0.31 |

The search therefore starts low and climbs while the score rises. Gold is not on the dial, so real scores sit lower; the shape is what carries over.

**Retain is cheap if the change stays on the images.** In `d3_ext_probe` (#14), images and classes the fine-tune never saw took about a third of the softening (KL 0.017 to 0.024 at temperature 2) and none of the flips (held-out loss mean 0.002, against 0.034 on trained images). Retain stays about 0.97 or above.

## Assumptions that can break

| Assumption | If wrong | Check |
|---|---|---|
| Gold errs on 10 to 25% of forget images | If it errs on 2 to 5%, a low K is best and high K overshoots to 0 | the sweep along K; never follow a 0 with a larger K |
| The cross-fit step points away from memorisation | If it is mostly refitting noise, flips go to arbitrary classes and `prediction_score` falls while the loss terms rise | compare the K sweep with a class-level hedge of equal loss |
| Altered copies behave like our views | Stronger hidden alterations could make the original less sure, shrinking the room to gain | the views in every report; jitter and shift10 track exact |
| A last-three-blocks fine-tune does not hurt general accuracy | The 70% gate zeroes everything | accuracy on held-out and spare classes in the probe (`acc_vs_reference`) must stay above 0.99 |
| The public 30% ranks candidates like the hidden 70% | With about a thousand scored images per split, aggregate terms are stable; a choice among many candidates by public score is not | one dial, a few points along it |

## Night protocol, the unlearning side

- GPU work: one open request at a time, one script that spreads its variants over the GPUs it is given, at most 90 minutes and 50 files (`AGENTS.md`). Current: #21, `run_fourth.sh` at `24cd666`.
- Submissions: `submit_queue.md` on this branch, written only by this agent, best first, each line with the file's sha256 and what its score will tell. I rewrite it after each score posted on #12. If I am down, the order in the file still makes sense on its own.
- I watch #12, #21 and `desk/unlearning` and answer within about 10 minutes.
- Local work, on CPU and the Mac's GPU: the original logits, target statistics, training-free candidates, and pre-screens of fine-tunes before they are requested.
- Last submission by 11:00. After 10:00, nothing new is requested on the GPU.

## Submission plan

1. `map_even40` (the lead, before sleeping): a class-level softening of all 45 touched classes. A small positive score confirms the extras reading's direction.
2. `x3_ext_m80`: the strongest candidate built so far (loss mean 0.09).
3. The K sweep from #21, starting near the K whose target loss mean is about 1, then up or down by the result, like a bisection on a tent-shaped curve.
4. Refinements only on the best K: more blocks trained, a T=2 hedge on top, or a wider spread.

## Log

| Time | What | Result |
|---|---|---|
| 23:05 | Took over; asked the desk for sha256 values and reordered x3_ext_m80 before m40 (#12) | |
| 23:15 | GPU request #21: four extrapolated candidates and a retain probe | |
| 23:25 | Target statistics per K on the laptop; #21 retargeted to K = 1.2 to 2.0 (`02b0eb5`) | K = 1.5 asks for loss mean 1.03, spread 2.35 |
| 23:37 | Desk submitted `x3_ext_m80` | **public 0.0063**, first score above 0 |
| 23:45 | Reading: its losses (mean 0.090, spread 0.458) match the K = 0.9 target. Gold between K = 1.2 and 1.6 would have given 0.11 or more (`dial_matrix.py`); 0.0063 fits gold at K of 3.5 or more, or off the dial. The queue climbs from `x4_e20`; `run_fifth.sh` (K = 2.5 to 4, `442697b`) asked on #21 | |
| 00:05 | U1 (`plateau.py`, research.md Night research): uniform outputs score 0 on every gold right on 58% or more; dividing all logits by t peaks at 0.13 to 0.25 for t = 3 or 4 over most golds, the likely 0.16 plateau; m80's 0.0063 fits only golds that are mostly right and soft, where the dial scores 0 | `scale_t3` is the cheapest plateau candidate; `run_sixth.sh` (forget-only tempering) asked |
| 00:08 | Desk submitted `x4_e20` | **public 0.0336**, five times `x3_ext_m80` |
| 00:20 | `ledger.py`: 84 golds weighed by our three scores. The weight goes to golds that are confidently wrong on most forget images (accuracy about a third, loss mean 5 to 8); soft golds predicted `x3_ext_m80` at ten times its score. Under the favoured golds, every-image softening (`scale_t*`) scores about 0.01, and targets K=3 tempered by 2 to 4 score 0.2 to 0.35 on the forget side | `scale_t4` kept at 00:38 as the test between the two readings; `run_seventh.sh` asked |

# Unlearning: the night of 21 to 22 November

## Resume here

For any session that picks the unlearning task up during the night (after a restart or a context reset). Read this section, then `submit_queue.md`, then the log at the end.

- **Role.** The unlearning agent, working through the lead's GitHub login. It never submits and never touches JURECA. The desk runs its GPU requests (#21) and submits from `submit_queue.md` (#12). Every GitHub post starts with `Unlearning:`. Commits carry no AI attribution.
- **Workspace.** Worktree `~/worktrees/euro-ai-championship-2026/kkrgp-unlearning-first-moves`, branch `kkrgp/unlearning-first-moves`. Data and the Python environment are on the SSD image: `source /Volumes/euroai-apfs/ops/env.sh` gives `$WT`, `$PY`, `$T`. If `/Volumes/euroai-apfs` is missing: `hdiutil attach /Volumes/Intenso/euroai/apfs.sparsebundle`. Run tools from `$T` (they read `output/kkrgp/...`).
- **Wake-ups.** `/Volumes/euroai-apfs/ops/watch.sh` in the background (exits on any change on #12, #19, #21, #24, #25, #28 or `desk/unlearning`, or after 39 minutes; restart it every time). A heartbeat is scheduled in the session every 20 minutes (minutes 7, 27, 47).
- **On a new score (#12).** Add it to `observed` in `ledger_inputs.json`, rerun `ledger.py --targets output/kkrgp/xfit_extras.npz`, log prediction against outcome below, rewrite `submit_queue.md` within 10 minutes, push, acknowledge on #12.
- **On new GPU results (`desk/unlearning`, `tasks/unlearning/reports/21/`).** Copy the JSON reports to `/Volumes/euroai-apfs/r21/`, add the candidates' measured statistics and retain estimates to `ledger_inputs.json`, rerun the ledger, queue the best by expected score.
- **Limits.** 8 unlearning submissions left at 01:08 under the night authorization (until 09:00), one per 30 minutes. No GPU request after 10:00; last submission by 11:00.
- **Pending at 02:16.**
  - Scores: `k7_e3t2_rm` 0.066 (01:38); `k7_e3t2_rm_x2` **0.2133** (02:08, our best). 02:38: `k8_e3t2_rm1_lr1e4_x2`, a desk-build (`splice_head.py --model output/kkrgp/k8_e3t2_rm1_lr1e4.pt --scale 0.5`). The desk posts its sha256 on #12; put it on the queue line at once (deadline 02:36). The prediction is on #12 (02:14).
  - Next dial after 02:38: the last-layer factor (x2.5, then x3) on the best base. See log row 02:15: the probe-based retain is pessimistic after sharpening.
  - The ledger (`ledger.py`) with both dumps: `--logits /Volumes/euroai-apfs/r21/seventh/logits.npz /Volumes/euroai-apfs/r21/eighth/logits.npz`. It ranks well and underpredicts sharpened fine-tunes by about 1.6 times.
  - Ninth batch: job 15904499 (`a8bd579`), submitted 02:13, pending on priority, about 33 minutes once started. Results under `reports/21/ninth/` with `logits.npz`. The renames are done (file count 689 before and after). Fold with `fold_reports.py /Volumes/euroai-apfs/r21/ninth --write`, rerun the ledger with all three dumps, and queue the best sharpened candidate.
  - Slots: 5 night submissions left after 02:38 (until 09:00), then the humans until 11:00.
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

The plan written at 23:00 (`map_even40`, `x3_ext_m80`, then a K sweep) was carried out through `x4_e20`. From 01:20 the plan follows the ledger, one dial at a time:

1. **Done.** 01:38 `k7_e3t2_rm` scored 0.066. 02:08 `k7_e3t2_rm_x2`, the same model with its last layer doubled, scored **0.2133**: sharpening is the recipe.
2. **02:38: `k8_e3t2_rm1_lr1e4_x2`.** One dial from the 0.2133: trained at learning rate 1e-4.
3. **03:08: the last-layer factor 2.5** on the better of the two bases (decision table on #12, 02:21).
4. **From about 03:38: the ninth batch** (job 15904499), sharpened. It brings tempered targets with more reach (learning rate 2e-4, 40 epochs) and brackets tau (1 and 3) and K (3.5).
5. **Each line goes into the queue at least 10 minutes before its slot,** with its predicted score. A line whose base failed the gate is withdrawn.

## Morning plan (for the humans, from 09:00)

- **State to read first:** `judges_brief.md` (it opens with the result), then `submit_queue.md` (every submission with its prediction and score), then this file's log.
- **Our best at 02:22:** 0.2133, `k7_e3t2_rm_x2.pt` (sha `f69f9ba1…`) on the lead's main laptop. If nothing better has scored by 11:00, this is the submission that counts. Nothing needs to be re-submitted, since only the best counts.
- **Slots after 09:00:** 09:08, 09:38, 10:08, 10:38, and 11:08 at the latest. The night authorization ends at 09:00, so from then on a human at a terminal submits with `make submit`.
- **What to submit:** the best-scoring base sharpened one notch further, or a ninth-batch base at the best factor so far. Check the queue first: the unlearning agent keeps it current until 07:30. The ledger (`ledger.py ... --logits` with every dump) ranks candidates well but underpredicts sharpened ones by about 1.6 times.
- **What not to do:**
  - no candidate without a spill probe;
  - nothing from `u7_*` (it changes half of the predictions on unseen classes);
  - no class dial on the nine classes (the gate: `dial15` scored 0);
  - no factor above 3 without a score at 2.5 first: the other-images golds collapse beyond it.
- **Defending the work:** `judges_brief.md`. The likely questions are answered at its end.

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
| 00:38 | Desk submitted `scale_t4` | **public 0.0057**, inside the range written on #12 before the score (0.004 to 0.013); soft golds predicted 0.1 or more and are out |
| 00:43 | Ledger with four scores: 71% confidently wrong golds, 19% removal. Batch `d926e1e` confirmed for the swap on #21. Asked for `dial12/15/20` (CPU builds) and proposed `dial15` for 01:08: about 0 if the gate binds, 0.15 to 0.98 on the forget side if the removal reading holds and the nine classes are outside validation and control | |
| 00:46 | Desk: link to JURECA lost at 00:41; `dial12/15/20` built on the lead's laptop (`dial15` sha256 `d512d1b4…`, equal to the local build) | `dial15` queued for 01:08 (`246d7c7`); the night's order posted on #12 |
| 00:55 | Ledger extended (`ef13498`): `boats<d>` dials (the four small-boat classes only), a removal variant whose validation and control sets hold no nine-class image (`removal_b`), and the dial gate taken from the cross-fitted predictions (unseen-image flips: `dial15` 80%, 7.2% of validation, passes on 44% of the band; `boats20` 3.8%, passes on all of it) | `dial15`: mean 0.015, P(above 0.0336) 0.20 (removal 0.076, removal_b 0.29, other golds 0); `boats20`: mean 0.023, P 0.20 (removal 0.116, removal_b 0.18). Written on #12 before the score. Without the proxy-based observations the removal families rise to 0.44 to 0.49 |
| 01:00 | Desk: a backup link to JURECA works; a new job can start when job 15904082 ends | eighth batch asked on #21 (`5198c86`) |
| 01:02 | Seventh-batch probes (`b23dad1`): the anchor at weight 1 or 3 slows the forget images (trained loss 1.93, 1.61, 1.33) and leaves the spill unchanged (held-out KL 1.1 to 1.2, spare-class KL 0.74 to 0.79); the fine-tunes reach about 40% of their target's loss (1.80 against 4.38) | eighth batch amended to `4276fe9`: a K=4 target replaces the weight-10 anchor |
| 01:03 | U2 delivered (`c05c800`, rows 13 to 20; #28) | |
| 01:08 | Desk submitted `dial15` | **public 0.0**, the "about 0" branch written on #12 |
| 01:12 | U3 delivered (`192649a`, rows 21 to 27; #28). New source: Dolgova and Stich (CISPA, 2026): retrains are right on 47% of a class's deleted images when half of it is deleted and on 17% at 90% (CIFAR-100; test accuracy 72%) | independent support for confidently wrong golds |
| 01:13 | **Ledger error found and fixed** (`7b51a3c`): the gate was a multiplier (the share of the accuracy band where a candidate passes), so a 0 counted as a miss rather than a possible gate failure, and `dial15`'s 0 wiped out the removal reading. Now the original's validation accuracy A has a flat prior over 70 to 85 and is inferred jointly with gold; each candidate passes or scores 0 at each A | A: mean 79, 10th to 90th 74 to 84. Removal 4% (it needs A between 73.4 and 75.4). `k7_e3t2_rm`: mean 0.047, 89% above 0.0336; `boats20` 4% |
| 01:14 | Queue: `k7_e3t2_rm` for 01:38 instead of `boats20` (`4a7a3bc`), prediction on #12 | |
| 01:15 | Eighth batch refocused (`5ac28b2`, #21): 16 runs on the confidently wrong reading (K=4 and 5, tau 1 to 2.5, 40 epochs, lr 1e-4, 9 blocks, anchor 1), the removal runs dropped | |
| 01:16 | Desk submitted the eighth batch (`5ac28b2`, job 15904237, pending on priority). Scratch is nearly out of file slots for every team: the desk created the batch's output files in advance | later batches must reuse existing file names |
| 01:18 | Seventh batch back (`357d3c8`): all records, logs and `sha256.txt`; `seventh/logits.npz` was written on JURECA but is not on the branch yet (asked on #21) | `u7_t32_rm3` (near-uniform targets) changes half of the unseen-class predictions; `k7_e2t2_rm3` spills least (held-out agreement 0.95, spare 0.995) |
| 01:20 | **Sharpening.** The fine-tunes' spill on spare classes matches a global temperature of about 2.4 (label probability at T=2 0.52, KL 0.74; the original divided by 2.25 gives 0.57 and 0.53). Multiplying the last layer by about 2 removes it without changing any prediction. `ledger.py` (`24e4eed`, tested in `test_ledger.py`) scores sharpened candidates: `k7_e3t2_rm` x2 0.078 against 0.047 unsharpened; `k7_e2t2_rm3` x2 0.064 with the gate passed at any A | queue: `k7_e3t2_rm` at 01:38, then `k7_e3t2_rm_x2` (desk-build) at 02:08, so the two scores isolate the change |
| 01:24 | Desk: `seventh/logits.npz` on `desk/unlearning` (`fa05188`); copied to `/Volumes/euroai-apfs/r21/seventh/`, the dumps match the reports. Ledger with exact logits for `x3_ext_m80`, `x4_e20` and the five dumped seventh candidates | posterior: aligned 0.42, other 0.51, removal 0.08; A mean 78.9 |
| 01:30 | **Sharpening model refined.** On the dumped fine-tunes, sharpening barely moves the KL on images whose prediction they changed (k7_e3t2_rm3: 2.11 at x1, 1.99 at x2) and divides it on the others (1.36 to 0.50). `sharpened_retain` now mixes the two at the held-out probe's measured share of changed predictions; the spare classes move like the unchanged images (tests: a pure temperature is undone exactly, a factor of 1 keeps the measured retain, changed predictions gain less) | `k7_e3t2_rm3` x2 retain 0.67 (0.73 under the temperature model); `k7_e3t2_rm` x2 mean 0.086, 88% above 0.0336 |
| 01:35 | **Decision table for 02:08, written before the 01:38 score** (ledger rerun with each outcome added): 0 (gate) gives A 73.5 to 74.2, every e3t2 fine-tune fails the gate, `k7_e2t2_rm3_x1p5` 0.063 at 100% above 0.0336; 0.02 gives `k7_e3t2_rm_x2` 0.078 at 100%; 0.047 gives 0.092 at 100%; 0.08 gives 0.106 at 69% (x2.5: 0.117 at 69%) | queue line 2 stays `k7_e3t2_rm_x2` unless the 01:38 score is 0; then `k7_e2t2_rm3_x1p5` (desk-build) |
| 01:37 | Sharpening range (ledger with the factor at 2, 2.5, 3 and 4): `k7_e3t2_rm` 0.086, 0.093, 0.089, 0.078 (chance above 0.0336: 0.88, 0.86, 0.85, 0.39). Golds that err where the cross-fit says keep rising (0.116 to 0.186), golds that err on other images of the class fall to 0 at 4 | the dial pays up to 2 or 2.5; beyond that it splits the two families, so x2 first and x2.5 only if x2 scores near the aligned prediction |
| 01:40 | **The posterior sat on the edge of the gold grid** (K=3, the largest). Widened to K=6 (gold right on 10% of the forget images) and removal to K=4 (`ledger.py`): the weight spreads over K=3 to 6 and tau_g 1 to 3, so our scores so far cannot tell how far beyond K=3 gold is; removal with K=4 on the thinned classes comes back to 13% (plus 6% `removal_b`) | predictions drop by 10 to 20% with the ranking unchanged: `k7_e3t2_rm` 0.044, `k7_e3t2_rm_x2` 0.071, x2.5 0.076, x3 0.073; the 02:08 decision table stands |
| 01:41 | Desk: the eighth batch (job 15904237) started at 01:26:37, ends near 02:15; scratch is out of file slots for every team, later batches must write only into existing files | |
| 01:43 | **Headroom: training reach is the bottleneck, not the target.** Forget term, weighted over the golds that carry 96% of the posterior (wide grid), if a fine-tune reached its target exactly: K3 tau2 (what the e3t2 runs aim at) 0.34; K4 tau1.5 0.51; K4 tau2 0.50; K5 tau2 0.51; K3 tau1 0.42. The dumped fine-tunes reach 0.085 (`k7_e3t2_rm3`) and 0.111 sharpened x2. They reach about 40% of their target's loss | with retain about 0.6, a fine-tune that reached a K4 tau2 target would score about 0.3. The ninth batch, if any, pushes reach (more epochs, higher learning rate, more blocks, more exact images) at the K4/K5 tau2 targets, judged by its spill probes |
| 01:41 | Desk: `k7_e3t2_rm` (01:38) **public 0.066, our best** (0.0336 before) | predicted 0.047 (10th to 90th 0 to 0.075, 89% above 0.0336): the upper end. No gold in the ledger predicts it above 0.048, so the ledger underpredicts this fine-tune by about 1.3 times; it enters through a proxy (no dumped logits until the eighth batch) |
| 01:46 | Ledger with the 0.066 (wide grid): aligned 0.46, other 0.37, removal 0.11, `removal_b` 0.07; A mean 79.5, 10th to 90th 75 to 84. `k7_e3t2_rm_x2` mean 0.086, 75% above 0.066; x2.5 0.093 (68%); `k7_e3t2` x2 0.086 | 02:08 stays `k7_e3t2_rm_x2` (sha `f69f9ba1…`, built by the desk), as written in the table before the score |
| 01:48 | Desk: the eighth batch's first half on `desk/unlearning` (`c632153`): four candidates, five probes; four checkpoints on the laptop with sha256 | folded with `fold_reports.py` into `ledger_inputs.json` |
| 01:52 | **Eighth batch, first half.** One dial at a time from `k7_e3t2_rm`: the untempered target (tau 1) halves the spill (held-out KL 0.61 against 1.11, spare 0.27 against 0.74) and reaches further (trained loss 2.02 against 1.61) at the same flip (5.5%); the doubled learning rate also wins on both (trained loss 2.34, held-out KL 0.99, spare 0.62); K=4 doubles the held-out flips (20% against 11%, gate risk). Ledger: `k8_e3t1_rm1` 0.096 (91% above 0.066), `k8_e4t2_rm1` 0.032 | queue: `k8_e3t1_rm1` for 02:38 (sha `fa5a4f2d…`), whatever 02:08 scores (decision table: 0.084 to 0.103 for it); the ninth batch combines tau 1 with the doubled learning rate |
| 01:57 | Pre-registered on #12: `k8_e3t1_rm1` for 02:38 whatever 02:08 scores | |
| 01:59 | **Ninth batch asked** on #21 (`6d06777`, `run_ninth.sh`): tau 1 with lr 1e-4, then lr 2e-4, 40 epochs, K=3.5 with tau 1.5 (the best target in the ledger if reached: 0.53, against 0.46 for K=3 with tau 1; sharper targets with tau below 1 score less), 9 blocks, K=2.5 (gate-safe fallback). It creates no file: the desk renames the seventh batch's files to the ninth's names; logs, dump and sha256 reuse `seventh/`. A smoke run of the K=3.5 configuration passes on CPU | about 35 minutes once the eighth job ends; results under `reports/21/ninth/` |
| 02:04 | Desk: eighth batch complete (job 15904237 ended 01:59, no failure), everything on `desk/unlearning` at `7c30bc4`, `eighth/logits.npz` included (the 8 plus `k7_e3t2_rm` and `k7_e3t2`) | folded; ledger with both dumps |
| 02:06 | Eighth batch, second half: 9 blocks is worse than 6 (flip 7.1%, retain 0.485, trained loss 1.56); 40 epochs and lr 1e-4 reach further with retain 0.52 and 0.53. With exact logits, the tempered runs sharpened by 2 lead: `k8_e3t2_rm1_lr1e4` x2 0.175, `k8_e3t2_rm1_ep40` x2 0.152, `k7_e3t2_rm` x2 0.106; `k8_e3t1_rm1` 0.092 | the 9-block run is dropped from the ninth batch |
| 02:09 | Desk: `k7_e3t2_rm_x2` (02:08) **public 0.2133, our best**, more than three times 0.066 | predicted 0.086 through a proxy (90th percentile 0.118), 0.106 with exact logits: sharpening pays about twice what the ledger says |
| 02:11 | Ninth batch amended (`a8bd579`, go on #21): tau 2 with lr 2e-4, lr 1e-4 with 40 epochs, both together, tau 3 (to be sharpened by 3), tau 1, K=3.5; same file plan | |
| 02:13 | Ledger with the 0.2133 (both dumps): aligned 0.67, other 0.11, removal 0.14, `removal_b` 0.08. It still underpredicts sharpened fine-tunes (0.132 for the 0.2133). Ranking: `k8_e3t2_rm1_lr1e4` x2 0.229 (x2.5 0.253, x3 0.261), `k8_e3t2_rm1_ep40` x2 0.198, `k7_e3t2_rm` x2.5 0.152 | 02:38: `k8_e3t2_rm1_lr1e4_x2` (desk-build), one dial from the 0.2133; `k8_e3t1_rm1` withdrawn |
| 02:15 | **Why the ledger underpredicts sharpening** (`k7_e3t2_rm` exact logits from the eighth dump): the observed gain x1 → x2 is 3.2 times. The retain model gives 1.38 (0.49 → 0.68). On the forget side, golds that err where the cross-fit says with tau 1 give 1.8 to 2.1 (K=3: 0.127 → 0.238), and golds that err on other images give 1.0 to 1.2, so those can no longer explain the scores. Under aligned K=3, tau 1, x1 is predicted 0.063 (observed 0.066) and x2 0.162 (observed 0.2133): the gap fits a real control-set retain near 0.9 at x2, that is, a spill on unseen images that is almost purely a temperature, with few of the changed predictions the held-out probe shows | the probe-based retain is pessimistic after sharpening. Under these golds the forget term keeps rising with the factor (K=3 tau 1: x2.5 +21%, x3 +40% over x2), so the factor is the next dial after 02:38 |
| 02:16 | Desk built `k8_e3t2_rm1_lr1e4_x2.pt` (sha `522e5851…`, `make check` passes); the submission reads the queue at 02:37:10 | sha256 on the queue line (`6a10e14`) |
| 02:21 | **Decision table for 03:08, written before the 02:38 score** (ledger with each outcome added; #12): above 0.2133 or within 0.02 of it, `k8_e3t2_rm1_lr1e4_x2p5` (0.246 to 0.292 across the outcomes); below 0.19, `k7_e3t2_rm_x2p5`. Factor 3 has a higher mean but a lower 10th percentile (it collapses under golds that err on other images), so 2.5 first. The desk is asked to pre-build both (`--scale 0.4`) | |
| 02:23 | Desk built both 03:08 candidates (`--scale 0.4`, `make check` passes): `k8_e3t2_rm1_lr1e4_x2p5` sha `9b7cab9e…`, `k7_e3t2_rm_x2p5` sha `7df6a948…` | the default line (`k8_e3t2_rm1_lr1e4_x2p5`) is on the queue now, so that the decision table holds even without me. It is replaced by `k7_e3t2_rm_x2p5` if 02:38 scores below 0.19 |
| 02:25 | **Altered copies:** the jitter view tracks the exact images at every factor for the four leading fine-tunes (loss mean and spread within about 5%, accuracy within 1.2 points). **Cross-check of the 0.2133:** under aligned K=3, tau 1 (loss 8.12/9.14), `k7_e3t2_rm` x2 (loss 2.02/2.49) has a mean/spread term of 0.26 and a forget term of about 0.24, so 0.2133 implies retain of about 0.9. With the same arithmetic, `k8_e3t2_rm1_lr1e4` x2 (loss 3.80/3.79) has a mean/spread term of 0.44: about 0.3 or more if retain stays near 0.9 | the mean/spread term keeps rising with the factor (x2.5: loss 4.67/4.76, x3: 5.55/5.73), so retain and the KL term decide where it stops |

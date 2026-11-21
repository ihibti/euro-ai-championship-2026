# Watermark localization: find the watermarked rectangle by decoding crops

**For:** the second agent (Devin), working for the lead on `watermark_localization`.
**Written:** 2026-11-21 18:55 Europe/Paris. The event ends 2026-11-22 12:00.
**Read first:** `AGENTS.md`, then `tasks/watermark_localization/README.md` (metric, edit types), then this file.

## Why this job

- A quarter of the images are "outpainting" cases: the watermark was embedded on one axis-aligned rectangle, and the rest of the picture is clean.
- Our segmentation networks are weak there. On held-out real images the best one (StegaStamp) scores 0.68 on rectangles against 0.82 on the other kinds, and early in training a network scores 0.48 on them, which is what an all-zeros mask gets.
- A rectangle decodes its message almost perfectly when you decode exactly its crop. So the box can be found by search, and checked, with no training.
- Each image counts the same. A found rectangle moves an image from about 0.48 to about 1.0.

This job is independent of the training work. The lead's other agent (Claude) trains the networks and assembles submissions. Do not duplicate that.

## What we measured (build on it, do not redo it)

- Images are 512 x 512. Methods: PixelSeal (256 bits), TrustMark Q (100 bits), StegaStamp (100 bits). Method and message are known for every train and test image.
- Per method, train has 170 images: 85 "hole" (full frame watermarked, a part restored), 43 exact rectangles, 42 object-shaped regions. Test has 340 per method, same mix expected.
- "Hole" images are recognised by decoding the full frame: bit accuracy at or above 0.89 (PixelSeal), 0.75 (TrustMark), 0.76 (StegaStamp). **Skip them in this job.**
- Rectangles: sides 9 to 501 px, aspect ratio at most 16:1 (PixelSeal, StegaStamp) and 2:1 (TrustMark), area 0.3% to 38% of the image.
- The embedder ran on the crop itself (the crop is resized to the embedder's working resolution and the change is resized back). The organizers' wrappers in `upstream/wmloc/watermarks.py` are the exact embedders.
- Bit accuracy when decoding the true bounding box of an exact rectangle, quartiles over the 43 images:
  - TrustMark: 0.52, 0.985, 1.0, 1.0, 1.0
  - StegaStamp: 0.45, 0.665, 0.83, 0.99, 1.0
  - PixelSeal: 0.473, 0.512, 0.645, 0.891, 1.0
- Decoding the full frame of a rectangle image gives chance level (about 0.5).
- Object-shaped regions do not decode from their bounding box (median about 0.5).
- A pixel-exact rectangle is not always a rectangle on the 128 x 128 scoring grid (corner cells can fall under 50%). Take exact rectangles from the pixel masks `release/train/masks/<name>.png`, not from the grid.

## Goal

For every image that is not a "hole" image, decide whether it holds a watermarked rectangle and, if so, return its box.

**Deliverable 1, files on JURECA:** `output/rect/<split>_<method>.npz` for split in `train`, `test` and the three methods, with:
- `names`: every image of that split and method, in `metadata.json` order;
- `boxes`: int array `[N, 4]` as `y0, y1, x0, x1` in pixels (`y1`, `x1` exclusive), `-1` where no search ran;
- `acc`: bit accuracy of the best box against the known message; `score`: mean over bits of `(2 * bit - 1) * logit`;
- `accepted`: bool, your decision that the box is a real watermarked rectangle.

**Deliverable 2, numbers on train (put them in the PR and in the report section below):**
- On the 43 exact rectangles per method: median IoU between found and true box, share with IoU at or above 0.9, and the mean official per-image mIoU of the rectangle mask on the 128 grid. All-zeros scores about 0.48 on these images.
- On the 42 object-shaped images per method: how many are wrongly accepted.
- The acceptance threshold per method, chosen to maximise mean per-image mIoU over all 85 non-hole train images when rejected images get an all-zeros mask.

Use `mean_miou` from `tasks/watermark_localization/evidence_features.py` for the metric, and `to_grid` from `synth.py` to turn a pixel mask into the 128 grid.

## A method to start from

1. **Tolerance first (20 minutes).** On train rectangles, decode boxes that are shifted or rescaled copies of the true box. Plot or tabulate score against IoU with the true box. This tells how coarse the search grid may be and whether the score has one clear peak. Do TrustMark first.
2. **Coarse search.** Candidate boxes on a ladder of sizes (for example 32 to 512 px, factor 1.25) and the allowed aspect ratios, centres on a stride of a quarter of the box side. Decode in batches. Keep the best few.
3. **Refine.** Move one edge at a time with shrinking steps (16, 8, 4, 2, 1 px) while the score rises.
4. **Decide.** Accept when bit accuracy passes the per-method threshold.

Details that matter:
- Reproduce what the decoder saw. Resize each crop to the decoder's working resolution with the same call the wrapper uses (`_resize` / `_prep`: PixelSeal 256 and TrustMark 245 with antialiased bilinear, StegaStamp 400 with antialiased bicubic and a clamp), then call `logits`.
- Use the soft `score` for the search and `acc` for the decision. The score is smoother.
- Optional seed for the search: the networks' probability maps, `output/seg/<method>_<tag>/train.npz` and `test.npz` (`names`, `probs` on the 128 grid). Read only. The bounding box of the predicted blob can replace the coarse search when it exists.
- All 1,530 images are packed in `covers/dataset_512.npy` (uint8, `[1530, 512, 512, 3]`), described by `covers/dataset_512.json` (`name`, `split`, `method`, and for train `kind`, `held_out`). Faster than opening PNGs. The `kind` there is grid-based, see the note above.
- Full-frame bit accuracy per image is already in `output/evidence/<split>_<method>.npz` (`names`, `crop512_acc[:, 0]`).

## Where and how to run

Everything heavy runs on JURECA. Paths below are relative to the task folder `W=/p/scratch/training2666/hibti1/Begg/Hackathon_Watermark_Localisation_Task`.

- Check the connection first: `ssh -O check jureca`. If it does not say "Master running", stop and tell the lead. Never retry SSH in a loop.
- Your code on the cluster lives in `$W/devin/repo`. Send it with, from your worktree:
  `git archive --format=tar HEAD | ssh jureca-mux "tar -xf - -C /p/scratch/training2666/hibti1/Begg/Hackathon_Watermark_Localisation_Task/devin/repo"`
- Run on a GPU with the helper, which attaches to one of our running jobs and starts in `$W` with the Python environment active:
  `ssh jureca-mux "/p/scratch/training2666/hibti1/Begg/Hackathon_Watermark_Localisation_Task/devin/run_gpu.sh 'python devin/repo/tasks/watermark_localization/rect/search.py --method trustmark --split train'"`
- The GPU is shared with a training run. Keep to these limits:
  - call `torch.cuda.set_per_process_memory_fraction(0.2)` at start, and decode at most 128 crops per batch;
  - one process at a time, each run under 15 minutes; write results as you go so a run can resume;
  - a training job ends around 20:00 and another starts; if a run dies then, start it again.
- If the helper prints "no running job to attach to", stop and tell the lead.

## Rules for this job

- Work in your own worktree and branch: `~/worktrees/euro-ai-championship-2026/devin-rect-search`, branch `devin/rect-search` from `main`. One PR to `main` at the end; the lead merges.
- New code goes in `tasks/watermark_localization/rect/`. Do not edit existing files in `tasks/watermark_localization/`.
- On JURECA, write only under `$W/devin/` and `$W/output/rect/`. Do not touch `$W/repo`, `$W/next`, `$W/output/seg`, `$W/covers` or `$W/.venv`.
- Never run `sbatch` or `scancel`. Never submit to the leaderboard. Never read or print secrets.
- No inline comments in code, plain commit messages, `make lint && make test` before committing.

## Report

When you have train numbers for a method, append them here under a dated heading and tell the lead. Stop and report early if:
- the tolerance test shows no usable peak for a method (say so, with the table);
- the search needs more than about 20 minutes of GPU per method on test.

## If this is finished: second job

Ask the lead before starting it.

- Object-shaped masks follow a real object in the picture. A network mask that roughly covers the object could be snapped to the object's outline.
- Idea: cut each image into segments with a public segmentation model, then mark a segment when most of it lies inside the network's predicted mask.
- Open points to settle with the lead first: whether outside pretrained models are allowed, where to install it (never in the shared `.venv`), and which GPU to use.

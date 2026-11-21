# Task 1: Localize image watermarks

Source: the organizers' task description "Localize Image Watermark" (European AI Championship Paris 2026). Supervisors: Adam Dziedzic and Franziska Boenisch. Developed by Simon Hector, Michel Meintz, Louis Kerner and Maitri Shah. Rewritten here, with numbers and formulas copied exactly.

Owner: TBD. Server task id: `34-watermark-localisation-image`.

## Goal

Find which parts of an image carry a watermark after image alterations have overwritten or added the watermark in parts of the image.

Every image was watermarked with one of three public post-hoc methods, **PixelSeal**, **TrustMark** or **StegaStamp**, in a rectangular subset of the original image. The aspect ratio of this subset does not exceed **16:1** for PixelSeal and StegaStamp, and **2:1** for TrustMark.

For every test image, return a binary mask:
- 1 = watermarked region
- 0 = clean (unwatermarked) region

## Why it is hard

- Some watermarking models, such as Watermark Anything, are trained for localization: they say which pixels are watermarked. PixelSeal, TrustMark and StegaStamp are post-hoc models, trained only to decode one message from the whole image.
- Edits range from large (half of the image) to small and irregular (a thin object, a face), so a solution has to be adaptive and general.
- The three watermarks behave very differently; there is likely no one-size-fits-all solution.
- Without localization, later edits are attributed to the original watermark owner. Third parties could then frame the owner by adding undesirable content.

## Edit types

- **Outpainting:** a watermarked image is extended to a larger image. The full watermark remains present but covers only part of the image.
- **Object removal or alteration:** an undesired object in a watermarked image is modified or edited out. A hole in the watermark remains.
- **Object addition:** something unwatermarked is added to a watermarked image and covers the watermark in that region. Again a hole forms.
- **Object transplantation:** an object from a watermarked image is added to an unwatermarked image. The watermark is present only on this object.

## Metric

Images are 512 x 512. Masks are evaluated on a **128 x 128 grid**, where one cell is 4 x 4 pixels. A ground-truth cell is 1 if at least 50% of its pixels are watermarked, otherwise 0. The starter kit contains the exact downsampling function.

Per image, for c in {0, 1}:

```
IoU_c      = |P_c ∩ G_c| / |P_c ∪ G_c|        (if P_c ∪ G_c is empty, IoU_c = 1)
mIoU_image = (IoU_0 + IoU_1) / 2
Score      = mean of mIoU_image over all test images
```

- P is the predicted mask and G the ground-truth mask. P_c and G_c are the sets of cells labelled c in each.
- Both false positives and false negatives enlarge the union, so both are penalized.
- If an image has no watermarked region and you correctly predict none, the union for class 1 is empty, and that class counts as a perfect 1.
- Every image counts equally, whatever the size of its edit.
- The score is in [0, 1]; higher is better.

## Data

1,530 PNG images, split equally across the three methods.

| Split | Images | Contents |
|---|---|---|
| Train | 510 (170 per method) | image, ground-truth mask (512 x 512 and 128 x 128), watermarking method, embedded message (bit string) |
| Test | 1,020 (340 per method) | image, watermarking method and embedded message |

- **Hugging Face:** `SprintML/Hackathon_Watermark_Localisation_Task`, **gated, with manual approval**. The team lead's account has access. The task PDF links to `SprintML/Hackathon_Watermark_Localisation_Task_Paris`, which redirects to the same dataset.
- **Files:** `train.zip` (190,830,424 bytes), `test.zip` (378,735,114 bytes), `starter_kit.zip` (10,057 bytes), `README.txt`.
- **Download:** `make data TASK=watermark_localization`, with `HF_TOKEN` from an approved account. The zips unpack to `release/<split>/`:
  - `images/<name>.png`: 512 x 512 RGB;
  - `metadata.json`: `[{"image_name", "method", "message"}, ...]`. Train items also have `full_watermark`: true if the whole image was watermarked and a part restored to the clean original;
  - train only: `masks/<name>.png` (512 x 512, 255 = watermarked) and `masks_128.npz` (`names`, and `masks` as N x 128 x 128 uint8 on the official grid).
- **JURECA:** the organizers' setup script downloaded the data into `/p/scratch/training2666/hibti1/Begg/Hackathon_Watermark_Localisation_Task/`. It builds **no `.venv`** for this task, because it reads only a top-level `requirements.txt` and this dataset has none (the kit's `requirements.txt` is inside `starter_kit.zip`). We built the environment ourselves; see On JURECA below.

## Starter kit

Vendored unmodified in `upstream/`; its own instructions are in `upstream/README.txt`.

- `wmloc/watermarks.py`: one interface for the three methods, `load_watermark("pixelseal" | "trustmark" | "stegastamp", models_dir, device)`. It exposes `embed(image, bits)`, `logits(images)`, `decode(images)` and `features(images)` (intermediate extractor feature maps). These are exactly the embedders used to build the dataset. Message lengths: PixelSeal 256 bits, TrustMark (variant Q) 100, StegaStamp 100. Each decoder resizes its input to its own working resolution: 256, 245 and 400. The models directory can be set with `WMLOC_MODELS`.
- `wmloc/data.py`: `unzip(zip, out_dir)` and `LocalisationDataset(root, split)`. Train items also carry `mask`, `mask128` and `full_watermark`.
- `wmloc/utils.py`: `bit_accuracy`, the official `downsample_mask` (512 to 128; a cell is 1 if at least 50% of its pixels are), the official `miou_image` and `score`.
- `submission_template.py`: writes an all-zeros baseline and prints the local train score. Its `BASE_URL` is a placeholder (`http://SERVER_IP`); the server is http://34.147.229.55, as for the other tasks. Submit with `make submit`, not with this script.
- `download_models.py`: downloads the PixelSeal checkpoint from `dl.fbaipublicfiles.com`, clones `facebookresearch/videoseal` at a pinned commit, and fetches the StegaStamp ONNX export from GitHub (WAVES), adding a copy with a logits output (needs `onnx`). TrustMark (`pip install trustmark==0.8.0`) downloads its weights on first use.

## Environment

The kit was tested on Python 3.11 with torch 2.5.1 and CUDA 12.4 wheels (`upstream/requirements.txt`, including `onnxruntime-gpu==1.23.2` and `trustmark==0.8.0`).

## On JURECA

`/p/scratch/training2666/hibti1/Begg/Hackathon_Watermark_Localisation_Task/`, set up and checked on 2026-11-21:

- `.venv`: Python 3.11 with the kit's `requirements.txt`, built by us.
- The kit is unpacked at the folder root: `wmloc/`, `submission_template.py`, `download_models.py`. `load_watermark` finds `models/` there by default; set `WMLOC_MODELS` only when running from elsewhere.
- `models/`: videoseal at the pinned commit, the PixelSeal checkpoint and the StegaStamp ONNX. All three wrappers were loaded once on the login node, so the lazily fetched TrustMark weights are cached and jobs can run offline.
- Data: `release/train/` and `release/test/`.
- Baseline: the official all-zeros prediction scores **0.2518** on the 510 train images (recorded in `results/`). A valid test file is at `output/baseline_allzeros.pt`; it has not been submitted.
- Run GPU work with `jsc/jureca_gpu.sbatch.template` and `TASK_DIR` set to this folder (see `jsc/README.md`, Submitting a job): one A100 40 GB per job (`gpu:1`, 8 CPUs) on reservation `cispahack`, which ends 2026-11-22 14:00. Compute nodes have no internet.

## Submission

A `.pt` file written with:

```
torch.save({"image_names": names, "masks": masks}, "submission.pt")
```

- `names[i]` is the image name of `masks[i]`: the file name **without `.png`**.
- `masks` is a tensor of shape `[1020, 128, 128]` with entries 0 or 1, dtype `bool` or `uint8`. 1 = watermarked.
- Every test image appears exactly once: no duplicates, no other entries.
- The server loads the file with `torch.load(..., weights_only=True)`. Anything other than plain tensors, lists, strings and dicts is rejected.
- Size: the task PDF says at most 20 MB; the kit's README says 25 MB. We enforce **20 MB**. A bool tensor is about 17 MB.

Submissions that break these rules are rejected automatically. `make submit` checks them all locally first. That includes the names, against `release/test/metadata.json` when the test split has been downloaded.

## Scoring and leaderboard

- Public leaderboard: 30% of the test images. The final ranking uses the other 70%.
- Leaderboard: http://34.147.229.55/leaderboard_page. Our team name there is **Begg**. It shows each team's best result; a lower score does not replace it.

## What can be scored where

| | Locally | Only on the server |
|---|---|---|
| Exact metric | the official per-image mIoU on the 510 training images: `make score TASK=watermark_localization ANSWER=<file>` (`wmloc/utils.py` semantics) | the 1,020 test images |
| Other signals | extractor bit accuracy against the known embedded message, intermediate feature maps | |

Test images come with their watermarking method and embedded message. A local decode can therefore be checked against the known message at test time too.

## Budgets

- Submission limit: not stated. Treat it like model equality: one submission per 30 minutes, failures included. `make submit` enforces this.

## References

- Tomáš Souček et al. "Pixel Seal: Adversarial-only training for invisible image and video watermarking." https://arxiv.org/abs/2512.16874. Brief: `papers/briefs/2512.16874_pixel-seal.md`
- Tu Bui, Shruti Agarwal, John Collomosse. "TrustMark: Robust Watermarking and Watermark Removal for Arbitrary Resolution Images." ICCV 2025. https://openaccess.thecvf.com/content/ICCV2025/html/Bui_TrustMark_Robust_Watermarking_and_Watermark_Removal_for_Arbitrary_Resolution_Images_ICCV_2025_paper.html
- Matthew Tancik, Ben Mildenhall, Ren Ng. "StegaStamp: Invisible Hyperlinks in Physical Photographs." CVPR 2020. https://www.computer.org/csdl/proceedings-article/cvpr/2020/716800c114/1m3nYiyKXK0
- Tom Sander, Pierre Fernandez, Alain Durmus, Teddy Furon, Matthijs Douze. "Watermark Anything with Localized Messages." ICLR 2025. https://openreview.net/forum?id=IkZVDzdC8M

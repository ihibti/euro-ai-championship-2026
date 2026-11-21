# Task 1: Localize image watermarks

Source: the organizers' task description "Localize Image Watermark" (European AI Championship Paris 2026). Supervisors: Adam Dziedzic and Franziska Boenisch. Developed by Simon Hector, Michel Meintz, Louis Kerner and Maitri Shah. Rewritten here, with numbers and formulas copied exactly.

Owner: TBD. Server task id: unknown until the starter kit is readable (see `upstream/PENDING.md`).

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

- **Hugging Face:** `SprintML/Hackathon_Watermark_Localisation_Task`, **gated, with manual approval**. The task PDF links to `SprintML/Hackathon_Watermark_Localisation_Task_Paris`, which redirects to the same dataset.
- **Files:** `train.zip` (190,830,424 bytes), `test.zip` (378,735,114 bytes), `starter_kit.zip` (10,057 bytes), `README.txt`.
- **Download:** `make data TASK=watermark_localization`, with `HF_TOKEN` set from an approved account.
- **JURECA:** the organizers' setup script puts the data in `/p/scratch/training2666/<lead>/<team>/Hackathon_Watermark_Localisation_Task/` (with `.venv`, `requirements.txt` and `output/`), if the lead's `HF_TOKEN` is approved for the gated dataset.

**Starter kit**, as described in the task PDF:
- code to unzip and load the dataset into PyTorch datasets;
- unified PyTorch wrappers for the PixelSeal, TrustMark and StegaStamp extractors and embedders, all with the same interface: `decode(images) -> bits` and access to intermediate feature maps;
- helper functions: bit accuracy, mask downsampling to 128 x 128, and local mIoU evaluation on the train split;
- `submission_template.py`.

## Submission

A `.pt` file containing a dictionary:

```
{"image_names": names, "masks": masks}
```

`names[i]` corresponds to `masks[i]`, and `masks` is a tensor of shape `[num_images, 128, 128]`.

- Each image name must appear exactly once.
- Masks must have the correct shape and entries (0 or 1).
- No duplicates or invalid entries.
- File size must not exceed 20 MB.

Submissions that break these rules are rejected automatically. `make submit` checks them locally first.

## Scoring and leaderboard

- Public leaderboard: 30% of the test images. The final ranking uses the other 70%.
- Leaderboard: http://34.147.229.55/leaderboard_page. Our team name there is **Begg**. It shows each team's best result; a lower score does not replace it.

## What can be scored where

| | Locally | Only on the server |
|---|---|---|
| Exact metric | mIoU on the 510 training images with ground truth (starter-kit evaluator) | the 1,020 test images |
| Other signals | extractor bit accuracy against the known embedded message, intermediate feature maps | |

Test images come with their watermarking method and embedded message. A local decode can therefore be checked against the known message at test time too.

## Budgets

- Submission limit: not stated. Treat it like model equality: one submission per 30 minutes, failures included. `make submit` enforces this.

## References

- Tomáš Souček et al. "Pixel Seal: Adversarial-only training for invisible image and video watermarking." https://arxiv.org/abs/2512.16874. Brief: `papers/briefs/2512.16874_pixel-seal.md`
- Tu Bui, Shruti Agarwal, John Collomosse. "TrustMark: Robust Watermarking and Watermark Removal for Arbitrary Resolution Images." ICCV 2025. https://openaccess.thecvf.com/content/ICCV2025/html/Bui_TrustMark_Robust_Watermarking_and_Watermark_Removal_for_Arbitrary_Resolution_Images_ICCV_2025_paper.html
- Matthew Tancik, Ben Mildenhall, Ren Ng. "StegaStamp: Invisible Hyperlinks in Physical Photographs." CVPR 2020. https://www.computer.org/csdl/proceedings-article/cvpr/2020/716800c114/1m3nYiyKXK0
- Tom Sander, Pierre Fernandez, Alain Durmus, Teddy Furon, Matthijs Douze. "Watermark Anything with Localized Messages." ICLR 2025. https://openreview.net/forum?id=IkZVDzdC8M

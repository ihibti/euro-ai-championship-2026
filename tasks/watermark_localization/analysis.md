# Watermark localization: where the held-out points are

Measured on 2026-11-21 at about 20:30, CPU only, on the candidates that `assemble.py` writes to `output/candidates/` (copied from JURECA into `data/watermark_localization/output/candidates/`) and on the train truth:

```
uv run --extra sci python tasks/watermark_localization/points_map.py --candidate c5
```

The held-out fifth is the 111 train images with `crc32(name) % 5 == 0`; no network trains on them. "At the test mix" reweights them to the test split: 340 images per method and, inside a method, the train shares of 85 holes, 43 rectangles and 42 objects per 170.

## 1. The train score measures memorisation

| Candidate | Train score (result record) | Trained-on images | Held-out | Held-out at the test mix | Public |
|---|---|---|---|---|---|
| c4a | 0.8216 | 0.8630 | 0.6727 | 0.6753 | 0.6876 |
| c5 | 0.9135 | 0.9666 | 0.7227 | 0.7273 | no submission record yet |

- Per method, held-out: c4a 0.5808 (PixelSeal), 0.6650 (TrustMark), 0.8068 (StegaStamp); c5 0.6529, 0.7016, 0.8432.
- c4a's held-out estimate at the test mix is 0.012 under its public score, which is inside the noise. Compare candidates on that estimate, never on the train score.
- Standard error: 0.017 on the held-out fifth, about 0.011 at the public split's 306 images. A difference under about 0.03 between two candidates means nothing unless it holds image by image (the paired interval of section 5).

## 2. Where the remaining gap is

c5, held-out at the test mix:

| Kind | Share of the remaining gap | Gain if every image of the kind were perfect |
|---|---|---|
| Holes (full frame watermarked, a part restored) | 51.6% | +0.141 |
| Rectangles | 25.8% | +0.070 |
| Objects | 22.6% | +0.062 |

- The largest pools: TrustMark holes 22.4% (held-out score 0.634, 20 images), PixelSeal holes 20.8% (0.660, 21 images), PixelSeal rectangles 14.0% (0.549, 7 images). StegaStamp holes hold 8.4% (0.862).
- c4a had the same shape: holes 50.3% (+0.163), rectangles 25.9% (+0.084), objects 23.8% (+0.077).

## 3. Hole images: the networks fit the training holes, not new ones

IoU of the clean (restored) region, median over hole images; held-out, with the share of held-out images under 0.1 in brackets, then trained-on:

| Method | c4a held-out | c4a trained-on | c5 held-out | c5 trained-on |
|---|---|---|---|---|
| PixelSeal | 0.01 (57%) | 0.56 | 0.43 (29%) | 0.94 |
| TrustMark | 0.22 (40%) | 0.83 | 0.25 (35%) | 0.93 |
| StegaStamp | 0.79 (0%) | 0.98 | 0.78 (0%) | 0.98 |

- From c4a to c5, TrustMark gained 0.10 on trained-on holes and 0.03 on held-out ones. More steps on the same four fifths mostly fit those images; a gain on holes has to be read on held-out images.
- Which images are hole images is solved (section 4). The open problem is where the clean region lies inside them. In the truth, 54% of hole images have two or more clean regions, up to 24.

## 4. The full-frame decode rule finds the hole images

On all 510 train images; the rule involves no training:

| Method | Holes found | False alarms | Missed | Test images flagged |
|---|---|---|---|---|
| PixelSeal | 82 of 85 | 2 | 3 | 49.4% |
| TrustMark | 84 of 85 | 3 | 1 | 51.8% |
| StegaStamp | 84 of 85 | 1 | 1 | 50.9% |

The test split therefore holds the same half of hole images per method as the train split.

## 5. Dead ends

Changes to c5's binary masks, on the held-out fifth, with a 90% paired bootstrap interval:

| Rule | Change |
|---|---|
| The decode flag replaces a mask under 50% marked by all ones | −0.0120 (−0.0249 to +0.0000) |
| Drop specks under 4, 16 or 64 cells, in both classes | −0.0002, −0.0005, −0.0072 |
| Keep the largest blob | −0.0185 (−0.0298 to −0.0082) |
| Snap a non-hole blob to its bounding box when it fills at least 80% of it | −0.0013 (−0.0038 to +0.0011) |

The masks are already clean, so the points are not in post-processing of binary masks. Keeping one blob loses because 54% of hole images and 39% of object images have two or more regions in the truth.

## 6. Kinds in the training logs

`kind_of` in `train_seg.py` calls an image a rectangle only when its whole bounding box is set on the 128 grid. Corner cells of an exact rectangle can fall under 50%, so it files 15 (PixelSeal, TrustMark) and 19 (StegaStamp) of the 43 exact rectangles per method as objects, and the `held_out_rect` and `held_out_object` numbers in `log.jsonl` mix the two kinds. `points_map.py` counts a rectangle when only corner cells are missing and nothing lies outside the box; that gives exactly 85, 43 and 42 per method.

## 7. What follows

- Hole localisation for TrustMark and PixelSeal is the largest pool: 43% of c5's remaining gap.
- The rectangle search can add at most +0.07 at the test mix, and only if it found every box. PixelSeal rectangles are where it pays most.
- Prefer a new candidate only when its paired held-out gain has an interval above 0.

## Limits

- One held-out fold of 111 images, with 4 to 21 images per method and kind.
- The test mix assumes the train shares of rectangles and objects; the decode rule confirms the share of holes.

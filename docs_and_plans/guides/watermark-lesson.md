# Watermark localisation: the lesson

For the lead, before the jury at 14:00. Written by the desk at 11:10 on 22 November from the code (`tasks/watermark_localization/`), the result records and the training logs on the cluster. Read sections 1 to 4, learn section 5, rehearse section 7 aloud.

## 1. The pitch (60 seconds, say this first)

> Each image carries one of three invisible watermarks on part of the picture, and we return which part. The tools that ship with these watermarks only answer for a whole image, so we trained one segmentation network per watermark. We had 510 labelled images, far too few, so we manufactured training data: we embedded watermarks ourselves into clean pictures, in shapes we chose, which gives exact masks for free. We judged every network on a fifth of the real images that none of them trained on. That held-out score, 0.73, predicted our public score, 0.74. The score on the training images, 0.92, did not. Our weak points are PixelSeal, the most invisible of the three, and rectangular watermarks.

## 2. The task in plain words

- **Input:** a 512 × 512 image, the name of its watermark method (PixelSeal, TrustMark or StegaStamp) and the hidden message.
- **Output:** a mask on a 128 × 128 grid: 1 where the watermark is, 0 where the picture is clean.
- **Why a watermark is only on a part:** the image was edited. Three kinds, measured on the training set:
  - **holes** (half of the images): the whole picture was watermarked, then a part was replaced by clean content;
  - **rectangles** (a quarter): the watermark sits only on a rectangle;
  - **objects** (a quarter): the watermark sits only on an object-shaped part.
- **Score:** IoU, intersection over union: the overlap between our mask and the true one, divided by their union. It is computed for the watermarked cells and for the clean cells, and the two are averaged (mIoU), then averaged over images.
- **Two reference points:**
  - answering "nothing is watermarked" everywhere scores **0.25**;
  - answering "all or nothing" per image, by decoding the message from the whole image, scores **0.45**.

## 3. Our solution

```
image ──> which method? (given) ──> that method's network ──> 128 × 128 probabilities ──> cut at 0.5 ──> mask
```

One network per method, because the three watermarks look nothing alike.

| Method | Network | Starts from | Held-out score (c5) |
|---|---|---|---|
| StegaStamp | plain convolutional network | random weights | 0.84 |
| TrustMark | plain convolutional network with extra fine-detail input channels | random weights | 0.70 |
| PixelSeal | a small network on top of PixelSeal's own extractor | the extractor's trained weights | 0.65 |

- **Why PixelSeal is built differently.** PixelSeal is designed to be invisible, so a network starting from nothing barely finds it. Its extractor, the network that reads the hidden message, already reacts to the watermark. We reuse its internal layers and train a small network on top that turns those reactions into a mask. Reusing a trained network for a new job is called **transfer learning**.
- **Where the training images come from.** A background process keeps making new examples: take a clean picture (a "cover"), draw a region (hole, rectangle or object, in the dataset's proportions), embed the watermark there with the organizers' own embedder. The mask is known exactly. Each batch of 16 holds 14 such synthetic images and 2 real ones.
- **A useful trick for covers.** An image carrying one method's watermark is a clean cover for the two other methods, so nearly half of the synthetic images are built on the dataset's own pictures.

## 4. How the training went (from the logs)

**StegaStamp, first network, 93 minutes on one GPU:**

| Step | Minutes | Loss | Held-out | Real images it trained on | Synthetic |
|---|---|---|---|---|---|
| 0 | 0 | | 0.17 | 0.27 | 0.27 |
| 3,000 | 6 | 0.51 | 0.53 | 0.51 | 0.34 |
| 7,000 | 14 | 0.32 | 0.72 | 0.78 | 0.76 |
| 15,000 | 29 | 0.19 | 0.70 | 0.91 | 0.84 |
| 23,000 | 44 | 0.13 | 0.80 | 0.97 | 0.87 |
| 31,000 | 59 | 0.11 | 0.82 | 0.98 | 0.89 |
| 49,000 | 93 | 0.09 | 0.83 | 0.99 | 0.90 |

How to read it:
- **The loss falls fast, then slowly.** That is normal: easy patterns first.
- **Held-out climbs, then flattens near 0.83** after about 30,000 steps.
- **The score on the real images it trained on keeps climbing to 0.99.** From there, more steps mostly memorise those 139 images. This is **overfitting**, and it is why we never quote the training score.
- **Held-out wobbles** (0.72, then 0.70, then 0.80). With 31 held-out StegaStamp images, one reading is uncertain by 0.02 to 0.03.

**PixelSeal, on its extractor, 49 minutes:** held-out 0.26 → 0.42 (1,000 steps) → 0.58 (3,000) → 0.62 (7,000) → 0.645 (22,000). It started with the watermark amplified three times, lowered to its real strength over the first third of the run. Making the task easy first and real afterwards is called a **curriculum**.

**The last runs (StegaStamp and TrustMark trained further):** held-out moved inside the noise (StegaStamp 0.86 to 0.88, TrustMark 0.68 to 0.72). The networks had reached their plateau. More of the same training no longer helps; a new idea is needed.

## 5. The improvement path

| Candidate | What changed | Training images | Held-out | Public |
|---|---|---|---|---|
| baseline | all zeros | 0.2518 | | |
| whole-image decode | all or nothing per image | 0.4529 | | 0.4421 |
| c3 | a network for StegaStamp only | 0.5942 | | |
| c4a | a network for each method | 0.8216 | 0.6727 | 0.6876 |
| c5 | the same designs trained much longer (PixelSeal: 2,000 steps to 18,000) | 0.9135 | 0.7227 | 0.7251 |
| c6 | StegaStamp and TrustMark trained further | 0.9248 | 0.7331 | **0.7444** |

- **Look at the two middle columns.** From c4a to c6 the training score rose by 0.10 and the held-out score by 0.06. The public score followed the held-out column every time, within 0.02.
- **c6 against c5:** +0.0104 on the same 111 held-out images, with a 90% interval from +0.0015 to +0.0201. Small, but above zero image by image (a **paired comparison**), so we kept it.
- **What we tried and dropped:** four ways of cleaning up the masks after the network (removing specks, keeping the largest blob, snapping to a box, forcing all-ones). All four lowered the held-out score or left it unchanged.

## 6. Vocabulary

| Word | Meaning, in our task |
|---|---|
| Weights | the adjustable numbers of a network; training sets them |
| Loss | one number for how wrong a prediction is; ours is a per-cell error plus one minus the IoU |
| Gradient descent | move every weight a little in the direction that lowers the loss, thousands of times |
| Learning rate | the size of that move; ours rises for 500 steps, then decays smoothly to zero |
| Step, batch | one update of the weights; the 16 images used for it |
| Segmentation | a prediction per pixel or cell, here "watermarked or not" |
| Convolutional network | a network that looks at small neighbourhoods of pixels, layer after layer |
| Held-out (validation) set | real images never used for training: 111 of 510, chosen by a hash of the file name |
| Overfitting | scoring well on training images by memorising them: 0.92 there, 0.73 held-out |
| Synthetic data | training images we made ourselves, with masks known by construction |
| Transfer learning, backbone | starting from a network trained for another job; the reused part is the backbone |
| Fine-tuning | continuing to train reused weights gently; our backbone moves at 30% of the learning rate |
| Curriculum | easy examples first: PixelSeal's watermark amplified three times at the start |
| IoU | overlap divided by union of two masks; 1 is perfect |
| Paired comparison | compare two candidates image by image on the same images, to see past the noise |

## 7. Honest limits

- **PixelSeal is our weakest method** (0.65 held-out), and it is a third of the score.
- **Rectangles are our weakest shape.** For PixelSeal rectangles the held-out score is 0.55. The real fix is a search over boxes using the extractor, which we planned and did not build.
- **The held-out set is small:** 111 images, 31 to 42 per method. A difference under about 0.03 between two candidates means nothing unless it is paired.
- **We stopped at 23:36.** The night went to the two tasks that stood near zero.
- **Rank:** 12th of 44 on this task (0.74; the best is 0.82).

## 8. Fifteen questions the jury may ask

1. **Why not use the watermark's own detector?** It answers for the whole image: one message, one yes or no. We need an answer per cell.
2. **Why three networks?** The three watermarks are different signals. One network per method lets each specialise, and the method is given with every image.
3. **You had 510 images. How did you train a network?** With synthetic data: we embed watermarks ourselves, so the masks are exact and the supply is unlimited. Real images are 2 of every 16.
4. **How do you know the synthetic images resemble the real ones?** We measured the real training set first (half holes, a quarter rectangles, a quarter objects) and generate in those proportions. The proof is the held-out score on real images.
5. **What is your loss?** Per-cell cross-entropy plus one minus a soft IoU, so that training pushes on the metric we are scored with.
6. **How did you choose when to stop training?** By the held-out score. It flattened while the training score kept rising, which is the sign of overfitting.
7. **Why is the training score 0.92 and the public score 0.74?** The 0.92 includes images the networks trained on and memorised. On images they never saw we measured 0.73, and the board gave 0.74.
8. **How was the held-out set chosen?** A fixed fifth, by a hash of the file name, decided before training. No network ever trained on it.
9. **Why is PixelSeal harder?** It is built to be imperceptible, so there is less signal in the pixels. That is why we put that network on PixelSeal's own extractor.
10. **What is transfer learning here?** The extractor was trained to read the hidden message, so its layers already respond to the watermark. We keep them, let them move slowly, and train a small part on top.
11. **Did you tune on the public board?** No. Candidates were compared on the held-out images; the board confirmed the held-out number each time.
12. **How sure are you that c6 beats c5?** +0.0104 paired, 90% interval above zero. It is a small gain and we say so.
13. **What did not work?** Cleaning the masks after the network: four rules, none helped. And training longer at the end: the networks had plateaued.
14. **What would you do next?** A box search for rectangles using the extractor's decoding score, and a better PixelSeal network. Holes on TrustMark and PixelSeal are the largest pool of lost points.
15. **What did you learn?** That the number to trust is the one on data the model never saw, and that when data is scarce you can often manufacture it if you own the process that creates it.

## 9. Three facts, settled this morning from the files

- **Only the PixelSeal network sits on its watermark's extractor.** The StegaStamp and TrustMark networks are plain networks trained from random weights. (A TrustMark network on its extractor was trained too, and not used in the final candidate.)
- **"snap3"** in `pixelseal_x1_snap3` is the third frozen copy of the PixelSeal run, taken at 19:41 on the 21st at step 18,000 while the run went on. It is the same PixelSeal network in c5 and in c6.
- **The two held-out numbers for c5** are two averages of the same results: 0.7326 is the mean of the three methods' scores; 0.7227 is the mean over the 111 images, where PixelSeal, the weakest, has the most images (42, against 38 and 31).

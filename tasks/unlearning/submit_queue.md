# Unlearning submission queue

Written only by the unlearning agent. The desk takes the first line whose sha256 has no submission record yet, checks the sha256 and `make check`, submits it, and posts the score on #12. Files are in the team's `unlearning/output/kkrgp/`. Reasons: `night.md`.

| File | sha256 | What the score will tell us |
|---|---|---|
| x3_ext_m80.pt | 2dcd74457a981d43668d2980504bdee50a0ac00f0f72807e896a529017b6b17e | Submitted 23:37: public 0.0063. |
| x4_e20.pt | 3e80d8a7bb692eeca3ac52985b1908a8d306a9d33e7d58677bff11557479a23e | The strongest move along the dial (#21): loss 0.46 / spread 0.70 on the exact images, 84.5% on the original label, KL 0.85 at T=2 from the original; about five times further than x3_ext_m80. Much above 0.0063: the dial points the right way and the queue climbs to x5 (K 2.5 to 4). Near 0.0063: the dial is not what limits the score; the queue switches to a different family. 0: overshoot or gate; step back to x4_e16. |
| scale_t4.pt | 60d6fb8fcdecf6869aa5c4d61c301aaaa0ddf9f08d2a0c0f0c3c25cfbbd76b3e | The original model with its last layer divided by 4 (`splice_head.py --scale 4`, built on JURECA by `run_sixth.sh`). The predicted class never changes, so the gate cannot fail. U1 (`research.md`, Night research): over 30 golds, t=4 is the most robust single temperature (mean predicted 0.11, 60% of golds at 0.1 or more; t=3: 0.08 and 27%). Above 0.1: the softening reading holds; forget-only tempering (`run_sixth.sh`) next. About 0: either gold is barely softer than the original (t=2 to 3 then) or much softer (t=6 to 8); read against `x4_e20`. |
| scale_t3.pt | 61bc1b4285da9045f9752c87f082484ae581cc1de75369dbbf04449413260ac2 | Last layer divided by 3. Read against `scale_t4` for the slope in t. |
| x4_e16.pt | 85c70e7893602809c0da5094002ea11321919c01a1caa3482a5e74433408176a | The middle of the dial: loss 0.30 / 0.60, 88.7% on label, held-out drift KL 0.34 (`d4_e16_probe`). Read against x4_e20 for the slope. |

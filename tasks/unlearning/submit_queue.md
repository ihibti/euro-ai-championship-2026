# Unlearning submission queue

Written only by the unlearning agent. The desk takes the first line whose sha256 has no submission record yet, checks the sha256 and `make check`, submits it, and posts the score on #12. Files are in the team's `unlearning/output/kkrgp/`. Reasons: `night.md`.

| File | sha256 | What the score will tell us |
|---|---|---|
| x3_ext_m80.pt | 2dcd74457a981d43668d2980504bdee50a0ac00f0f72807e896a529017b6b17e | Submitted 23:37: public 0.0063. |
| x4_e20.pt | 3e80d8a7bb692eeca3ac52985b1908a8d306a9d33e7d58677bff11557479a23e | The strongest move along the dial (#21): loss 0.46 / spread 0.70 on the exact images, 84.5% on the original label, KL 0.85 at T=2 from the original; about five times further than x3_ext_m80. Much above 0.0063: the dial points the right way and the queue climbs to x5 (K 2.5 to 4). Near 0.0063: the dial is not what limits the score; the queue switches to a different family. 0: overshoot or gate; step back to x4_e16. |
| scale_t3.pt | f77346a81a70ccfae92e21f4ebb190cb0ef7c656accc47efb19c2f8365f33e98 | The original model with its last layer divided by 3 (`splice_head.py --scale 3` at `58ab3b3`; CPU, seconds). The predicted class never changes, so the gate cannot fail. U1 (`research.md`, Night research): dividing all logits by one temperature peaks at 0.13 to 0.25 for t = 3 or 4 over most golds, the likely 0.16 plateau. Above 0.1: the softening reading holds; t=4 next, and forget-only tempering (`run_sixth.sh`). About 0: gold is barely softer than the original; t=2 next. The sha256 is the laptop's build (torch 2.15.0); if JURECA's build differs, the desk's sha256 replaces it. |
| x4_e16.pt | 85c70e7893602809c0da5094002ea11321919c01a1caa3482a5e74433408176a | The middle of the dial: loss 0.30 / 0.60, 88.7% on label, held-out drift KL 0.34 (`d4_e16_probe`). Read against x4_e20 for the slope. |

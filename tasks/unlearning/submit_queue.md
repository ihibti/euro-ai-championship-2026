# Unlearning submission queue

Written only by the unlearning agent. The desk takes the first line whose sha256 has no submission record yet, checks the sha256 and `make check`, submits it, and posts the score on #12. Files are in the team's `unlearning/output/kkrgp/`. Reasons: `night.md`.

| File | sha256 | What the score will tell us |
|---|---|---|
| x3_ext_m80.pt | 2dcd74457a981d43668d2980504bdee50a0ac00f0f72807e896a529017b6b17e | The strongest per-image fine-tune built so far (loss mean 0.09, label probability at T=2 0.90). Above 0: soft per-image targets from the cross-fit move towards gold, and the K dial of #21 is the next step. Exactly 0: the cross-fit direction is wrong or the gate was hit; stop the K sweep and write the queue again. |

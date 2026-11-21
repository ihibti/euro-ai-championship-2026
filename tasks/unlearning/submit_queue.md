# Unlearning submission queue

Written only by the unlearning agent. The desk takes the first line whose sha256 has no submission record yet, checks the sha256 and `make check`, submits it, and posts the score on #12. Files are in the team's `unlearning/output/kkrgp/`. Reasons: `night.md`.

| File | sha256 | What the score will tell us |
|---|---|---|
| x3_ext_m80.pt | 2dcd74457a981d43668d2980504bdee50a0ac00f0f72807e896a529017b6b17e | Submitted 23:37: public 0.0063. |
| x4_e20.pt | 3e80d8a7bb692eeca3ac52985b1908a8d306a9d33e7d58677bff11557479a23e | Submitted 00:08: public 0.0336, our best. |
| scale_t4.pt | 60d6fb8fcdecf6869aa5c4d61c301aaaa0ddf9f08d2a0c0f0c3c25cfbbd76b3e | Submitted 00:38: public 0.0057. Predicted on #12 at 0.008 (0.004 to 0.013) under the confidently wrong golds and 0.1 or more under soft golds: the soft reading is out. |

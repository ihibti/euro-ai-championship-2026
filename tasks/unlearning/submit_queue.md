# Unlearning submission queue

Written only by the unlearning agent. The desk takes the first line whose sha256 has no submission record yet, checks the sha256 and `make check`, submits it, and posts the score on #12. Files are in the team's `unlearning/output/kkrgp/`. Reasons: `night.md`.

| File | sha256 | What the score will tell us |
|---|---|---|
| x3_ext_m80.pt | 2dcd74457a981d43668d2980504bdee50a0ac00f0f72807e896a529017b6b17e | Submitted 23:37: public 0.0063. |
| x4_e20.pt | 3e80d8a7bb692eeca3ac52985b1908a8d306a9d33e7d58677bff11557479a23e | Submitted 00:08: public 0.0336, our best. |
| scale_t4.pt | 60d6fb8fcdecf6869aa5c4d61c301aaaa0ddf9f08d2a0c0f0c3c25cfbbd76b3e | Submitted 00:38: public 0.0057. Predicted on #12 at 0.008 (0.004 to 0.013) under the confidently wrong golds and 0.1 or more under soft golds: the soft reading is out. |
| dial15.pt | d512d1b496b7c3fdfb2262e0fc5c8cd49beb049ef64ac53e6ef6ae32278c9c78 | Submitted 01:08: public 0.0. Predicted on #12: 0 under the confidently wrong golds, 0.076 under removal with the gate, 0.29 under removal without nine-class validation images. Removal now needs the original's validation accuracy between 73.4 and 75.4 (ledger with the joint gate, `ledger.py` 01:15): 4% of the weight. |
| k7_e3t2_rm.pt | desk copy from JURECA, sha256 checked on copy: please post it on #12 | Seventh batch (cancelled run, `7ea0106`): the last 6 blocks fine-tuned 25 epochs towards `orig + 3 × step` tempered by 2, mixup anchor weight 1. Ledger at the commit of this line: mean 0.047, 10th to 90th 0 to 0.075, 89% chance above 0.0336. The 0 end is the gate: it changes about 6% of validation predictions (probe `d7_e3t2_rm_probe`), so it needs A ≥ 74.5. Above 0.034: the confidently wrong reading holds and moving further pays; about 0: the gate binds at about 6% of changed predictions; 0.01 to 0.03: the spill costs more than the probes say. |

# What a partial answer is worth inside one family of 11

Exact within-family MCC (55 pairs), scored as the organizers' scorer scores it.
 With the families right and the three families alike, the task score is 0.1 + 0.9 x MCC.

- M1: S x7, fp8, three distinct LoRA (the references' make-up, two more disguises)
- M2: S x5, fp8, LoRA L1 x2, L2 x2, L3 (two adapters served twice, once disguised)
- M3: S x6, fp8, a second quantization, three distinct LoRA

| answer | M1 | M2 | M3 |
|---|---|---|---|
| all one model (the 01:30 submission) | 0.000 | 0.000 | 0.000 |
| (a) base with fp8 as one model, every LoRA alone | 0.772 | 0.665 | 0.601 |
| (a) and the LoRA twins paired | 0.772 | 0.790 | 0.601 |
| (a) and fp8 found (alone) | 1.000 | 0.892 | 1.000 |
| (b) one LoRA missed (in the base) | 0.571 | 0.491 | 0.445 |
| (b) two LoRA missed | 0.370 | 0.431 | 0.289 |
| (b) every LoRA missed but one | 0.370 | 0.249 | 0.289 |
| (c) one base endpoint set apart | 0.538 | 0.436 | 0.359 |
| (c) two base endpoints set apart | 0.359 | 0.239 | 0.175 |
| (b) and (c): one LoRA missed, one base endpoint apart | 0.323 | 0.270 | 0.193 |
| LoRA found but merged into one model | 0.692 | 0.579 | 0.539 |
| the true answer | 1.000 | 1.000 | 1.000 |

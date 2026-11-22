# What a partial answer is worth inside one family of 11

Exact within-family MCC (55 pairs), not clipped at 0: the scorer clips only the pooled MCC over the 165 within-family pairs, so a negative family subtracts.
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

## k endpoints split off as singletons, j of them truly changed

Family of 11 with c changed endpoints, each its own model, and 11 - c copies of one model. The answer keeps 11 - k endpoints as one model and sets k apart, j of them changed. Cells: within-family MCC.

| k | j | c = 3 | c = 4 | c = 5 | c = 6 |
|---|---|---|---|---|---|
| 0 | 0 | 0.000 | 0.000 | 0.000 | 0.000 |
| 1 | 0 | -0.180 | -0.212 | -0.241 | -0.267 |
| 1 | 1 | 0.480 | 0.370 | 0.289 | 0.222 |
| 2 | 0 | -0.254 | -0.295 | -0.328 | -0.351 |
| 2 | 1 | 0.204 | 0.099 | 0.016 | -0.054 |
| 2 | 2 | 0.740 | 0.571 | 0.445 | 0.342 |
| 3 | 0 | -0.310 | -0.351 | -0.379 | -0.386 |
| 3 | 1 | 0.054 | -0.052 | -0.134 | -0.197 |
| 3 | 2 | 0.491 | 0.323 | 0.193 | 0.086 |
| 3 | 3 | 1.000 | 0.772 | 0.601 | 0.463 |
| 4 | 0 | -0.351 | -0.387 | -0.397 | -0.370 |
| 4 | 1 | -0.052 | -0.155 | -0.229 | -0.273 |
| 4 | 2 | 0.323 | 0.153 | 0.023 | -0.079 |
| 4 | 3 | 0.772 | 0.538 | 0.359 | 0.212 |
| 4 | 4 | - | 1.000 | 0.779 | 0.600 |
| 5 | 0 | -0.379 | -0.397 | -0.375 | -0.289 |
| 5 | 1 | -0.134 | -0.229 | -0.283 | -0.289 |
| 5 | 2 | 0.193 | 0.023 | -0.100 | -0.183 |
| 5 | 3 | 0.601 | 0.359 | 0.175 | 0.029 |
| 5 | 4 | - | 0.779 | 0.542 | 0.346 |
| 5 | 5 | - | - | 1.000 | 0.770 |

## Expected within-family MCC of splitting k endpoints, each changed with probability p

j is binomial(k, p), the splits independent; the values above for each j. A first guide to the cut: all-in-one is 0.

c = 4 changed endpoints:

| k | p = 0.3 | p = 0.4 | p = 0.5 | p = 0.6 | p = 0.7 |
|---|---|---|---|---|---|
| 1 | -0.037 | 0.021 | 0.079 | 0.138 | 0.196 |
| 2 | -0.052 | 0.033 | 0.118 | 0.206 | 0.295 |
| 3 | -0.061 | 0.044 | 0.154 | 0.269 | 0.388 |
| 4 | -0.068 | 0.057 | 0.191 | 0.334 | 0.487 |
| 5 | -0.073 | 0.060 | 0.193 | 0.310 | 0.387 |

c = 5 changed endpoints:

| k | p = 0.3 | p = 0.4 | p = 0.5 | p = 0.6 | p = 0.7 |
|---|---|---|---|---|---|
| 1 | -0.082 | -0.029 | 0.024 | 0.077 | 0.130 |
| 2 | -0.114 | -0.039 | 0.037 | 0.115 | 0.195 |
| 3 | -0.136 | -0.045 | 0.050 | 0.151 | 0.256 |
| 4 | -0.150 | -0.048 | 0.065 | 0.188 | 0.320 |
| 5 | -0.155 | -0.045 | 0.083 | 0.230 | 0.395 |


# Night-probe simulator: share of p < 0.05

20 items x 4 options, base logits N(0, 2.0^2), 150 replicates per cell, 1,000 sampled permutations per test. Same-model rows should read 0.05 or less.

| scenario | mode, n=5 | mmd, n=5 | mode, n=10 | mmd, n=10 |
|---|---|---|---|---|
| same model: identical | 0.040 | 0.080 | 0.027 | 0.033 |
| same model: temperature x1.6 (flatter) | 0.067 | 0.133 | 0.020 | 0.187 |
| same model: temperature /1.6 (sharper) | 0.027 | 0.087 | 0.027 | 0.253 |
| same model: bias +1 on option 0 | 0.127 | 0.133 | 0.200 | 0.373 |
| different: new mode on 2 items | 0.367 | 0.293 | 0.680 | 0.640 |
| different: new mode on 5 items | 0.760 | 0.740 | 0.940 | 0.960 |
| different: noise sd 0.5 on all items | 0.100 | 0.167 | 0.193 | 0.360 |
| different: noise sd 1.0 on all items | 0.380 | 0.533 | 0.473 | 0.907 |

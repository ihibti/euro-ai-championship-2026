# p3_repeat: answers per endpoint to call a LoRA copy, plain rate 0.1

Plain rate p0 = 0.1. Exact one-sided binomial test per endpoint with p0 known. Power is the chance that one LoRA endpoint is called; false calls are counted over 33 endpoints as if none were LoRA (the worst case).

## E[false calls] <= 1: per-endpoint alpha 0.0303

| answers n | call at k >= c | actual alpha | expected false calls in 33 | power, p1 = 0.6 | power, p1 = 0.35 |
|---|---|---|---|---|---|
| 4 | 3 | 0.0037 | 0.12 | 0.475 | 0.126 |
| 5 | 3 | 0.0086 | 0.28 | 0.683 | 0.235 |
| 6 | 3 | 0.0159 | 0.52 | 0.821 | 0.353 |
| 8 | 4 | 0.0050 | 0.17 | 0.826 | 0.294 |
| 10 | 4 | 0.0128 | 0.42 | 0.945 | 0.486 |
| 12 | 4 | 0.0256 | 0.85 | 0.985 | 0.653 |
| 15 | 5 | 0.0127 | 0.42 | 0.991 | 0.648 |
| 20 | 6 | 0.0113 | 0.37 | 0.998 | 0.755 |
| 25 | 7 | 0.0095 | 0.31 | 1.000 | 0.827 |
| 30 | 7 | 0.0258 | 0.85 | 1.000 | 0.941 |

## P(2+ false calls) <= 0.10: per-endpoint alpha 0.0162

| answers n | call at k >= c | actual alpha | expected false calls in 33 | power, p1 = 0.6 | power, p1 = 0.35 |
|---|---|---|---|---|---|
| 4 | 3 | 0.0037 | 0.12 | 0.475 | 0.126 |
| 5 | 3 | 0.0086 | 0.28 | 0.683 | 0.235 |
| 6 | 3 | 0.0159 | 0.52 | 0.821 | 0.353 |
| 8 | 4 | 0.0050 | 0.17 | 0.826 | 0.294 |
| 10 | 4 | 0.0128 | 0.42 | 0.945 | 0.486 |
| 12 | 5 | 0.0043 | 0.14 | 0.943 | 0.417 |
| 15 | 5 | 0.0127 | 0.42 | 0.991 | 0.648 |
| 20 | 6 | 0.0113 | 0.37 | 0.998 | 0.755 |
| 25 | 7 | 0.0095 | 0.31 | 1.000 | 0.827 |
| 30 | 8 | 0.0078 | 0.26 | 1.000 | 0.876 |

# p3_repeat: answers per endpoint to call a LoRA copy, plain rate 0.2

Plain rate p0 = 0.2. Exact one-sided binomial test per endpoint with p0 known. Power is the chance that one LoRA endpoint is called; false calls are counted over 33 endpoints as if none were LoRA (the worst case).

## E[false calls] <= 1: per-endpoint alpha 0.0303

| answers n | call at k >= c | actual alpha | expected false calls in 33 | power, p1 = 0.6 | power, p1 = 0.35 |
|---|---|---|---|---|---|
| 4 | 3 | 0.0272 | 0.90 | 0.475 | 0.126 |
| 5 | 4 | 0.0067 | 0.22 | 0.337 | 0.054 |
| 6 | 4 | 0.0170 | 0.56 | 0.544 | 0.117 |
| 8 | 5 | 0.0104 | 0.34 | 0.594 | 0.106 |
| 10 | 6 | 0.0064 | 0.21 | 0.633 | 0.095 |
| 12 | 6 | 0.0194 | 0.64 | 0.842 | 0.213 |
| 15 | 7 | 0.0181 | 0.60 | 0.905 | 0.245 |
| 20 | 9 | 0.0100 | 0.33 | 0.943 | 0.238 |
| 25 | 10 | 0.0173 | 0.57 | 0.987 | 0.370 |
| 30 | 11 | 0.0256 | 0.85 | 0.997 | 0.492 |

## P(2+ false calls) <= 0.10: per-endpoint alpha 0.0162

| answers n | call at k >= c | actual alpha | expected false calls in 33 | power, p1 = 0.6 | power, p1 = 0.35 |
|---|---|---|---|---|---|
| 4 | 4 | 0.0016 | 0.05 | 0.130 | 0.015 |
| 5 | 4 | 0.0067 | 0.22 | 0.337 | 0.054 |
| 6 | 5 | 0.0016 | 0.05 | 0.233 | 0.022 |
| 8 | 5 | 0.0104 | 0.34 | 0.594 | 0.106 |
| 10 | 6 | 0.0064 | 0.21 | 0.633 | 0.095 |
| 12 | 7 | 0.0039 | 0.13 | 0.665 | 0.085 |
| 15 | 8 | 0.0042 | 0.14 | 0.787 | 0.113 |
| 20 | 9 | 0.0100 | 0.33 | 0.943 | 0.238 |
| 25 | 11 | 0.0056 | 0.18 | 0.966 | 0.229 |
| 30 | 12 | 0.0095 | 0.31 | 0.992 | 0.345 |

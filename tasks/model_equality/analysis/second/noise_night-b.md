# Noise floor on the reference endpoints, `night-b.jsonl`

## Statistic: night p3_repeat (p3_repeat)

Answers per endpoint (minimum over these prompts): ref-01 10, ref-02 10, ref-03 10, ref-04 10, ref-05 10, ref-06 10, ref-07 10, ref-08 10, ref-09 10.

| rows | count | median z | share p<0.05 | median z at half size |
|---|---|---|---|---|
| 0 zero point: two halves of one endpoint | 27 | -0.69 | 0.00 | -0.69 |
| 1 same model, disguised | 10 | -0.28 | 0.00 | -0.06 |
| 2 fp8 copy against S | 5 | -0.12 | 0.00 | -0.55 |
| 3 LoRA copy against S | 15 | 3.56 | 0.80 | 1.38 |
| 4 LoRA against LoRA | 3 | 0.00 | 0.00 | 0.23 |
| 5 LoRA against fp8 | 3 | 4.03 | 1.00 | 1.50 |

| a | b | type | z | p | z at half size |
|---|---|---|---|---|---|
| ref-01 | ref-02 | 2 fp8 copy against S | -0.12 | 0.4340 | -0.06 |
| ref-01 | ref-03 | 1 same model, disguised | -0.17 | 0.5505 | 0.11 |
| ref-01 | ref-04 | 3 LoRA copy against S | 0.85 | 0.2070 | 0.45 |
| ref-01 | ref-05 | 1 same model, disguised | -0.41 | 0.6610 | -0.19 |
| ref-01 | ref-06 | 1 same model, disguised | 0.33 | 0.3705 | 0.07 |
| ref-01 | ref-07 | 3 LoRA copy against S | 1.17 | 0.2020 | 0.59 |
| ref-01 | ref-08 | 1 same model, disguised | 1.56 | 0.1270 | 0.29 |
| ref-01 | ref-09 | 3 LoRA copy against S | 1.18 | 0.1990 | 0.54 |
| ref-02 | ref-03 | 2 fp8 copy against S | 0.69 | 0.2795 | 0.14 |
| ref-02 | ref-04 | 5 LoRA against fp8 | 5.11 | 0.0045 | 1.50 |
| ref-02 | ref-05 | 2 fp8 copy against S | 0.81 | 0.3630 | -0.70 |
| ref-02 | ref-06 | 2 fp8 copy against S | -1.04 | 0.8885 | -0.55 |
| ref-02 | ref-07 | 5 LoRA against fp8 | 4.03 | 0.0090 | 1.50 |
| ref-02 | ref-08 | 2 fp8 copy against S | -1.09 | 0.9130 | -0.65 |
| ref-02 | ref-09 | 5 LoRA against fp8 | 3.97 | 0.0080 | 1.47 |
| ref-03 | ref-04 | 3 LoRA copy against S | 7.35 | 0.0005 | 3.48 |
| ref-03 | ref-05 | 1 same model, disguised | 1.12 | 0.1235 | 0.14 |
| ref-03 | ref-06 | 1 same model, disguised | -1.53 | 0.8885 | -0.63 |
| ref-03 | ref-07 | 3 LoRA copy against S | 5.93 | 0.0005 | 3.04 |
| ref-03 | ref-08 | 1 same model, disguised | -1.48 | 0.9525 | -1.15 |
| ref-03 | ref-09 | 3 LoRA copy against S | 6.04 | 0.0015 | 3.04 |
| ref-04 | ref-05 | 3 LoRA copy against S | 4.91 | 0.0025 | 1.72 |
| ref-04 | ref-06 | 3 LoRA copy against S | 2.25 | 0.0360 | 0.91 |
| ref-04 | ref-07 | 4 LoRA against LoRA | -0.01 | 0.7140 | 0.23 |
| ref-04 | ref-08 | 3 LoRA copy against S | 3.87 | 0.0025 | 1.21 |
| ref-04 | ref-09 | 4 LoRA against LoRA | 0.00 | 0.7015 | 0.32 |
| ref-05 | ref-06 | 1 same model, disguised | -1.67 | 0.9665 | -0.93 |
| ref-05 | ref-07 | 3 LoRA copy against S | 3.56 | 0.0095 | 1.58 |
| ref-05 | ref-08 | 1 same model, disguised | -0.39 | 0.7090 | -0.46 |
| ref-05 | ref-09 | 3 LoRA copy against S | 3.49 | 0.0110 | 1.38 |
| ref-06 | ref-07 | 3 LoRA copy against S | 2.73 | 0.0405 | 1.20 |
| ref-06 | ref-08 | 1 same model, disguised | 1.52 | 0.1860 | 0.14 |
| ref-06 | ref-09 | 3 LoRA copy against S | 2.69 | 0.0425 | 1.12 |
| ref-07 | ref-08 | 3 LoRA copy against S | 4.25 | 0.0040 | 1.91 |
| ref-07 | ref-09 | 4 LoRA against LoRA | 0.84 | 0.3640 | 0.06 |
| ref-08 | ref-09 | 3 LoRA copy against S | 4.20 | 0.0030 | 1.91 |

## Statistic: night p4_nextword (p4_nextword)

Answers per endpoint (minimum over these prompts): ref-01 10, ref-02 10, ref-03 10, ref-04 10, ref-05 10, ref-06 10, ref-07 10, ref-08 10, ref-09 10.

| rows | count | median z | share p<0.05 | median z at half size |
|---|---|---|---|---|
| 0 zero point: two halves of one endpoint | 27 | 0.00 | 0.00 | 0.00 |
| 1 same model, disguised | 10 | -1.40 | 0.00 | 0.12 |
| 2 fp8 copy against S | 5 | -1.30 | 0.00 | 0.16 |
| 3 LoRA copy against S | 15 | -0.20 | 0.00 | 0.49 |
| 4 LoRA against LoRA | 3 | 0.69 | 0.00 | -0.25 |
| 5 LoRA against fp8 | 3 | -0.28 | 0.00 | -0.23 |

| a | b | type | z | p | z at half size |
|---|---|---|---|---|---|
| ref-01 | ref-02 | 2 fp8 copy against S | -1.70 | 0.9430 | 0.16 |
| ref-01 | ref-03 | 1 same model, disguised | -2.23 | 0.9905 | 0.30 |
| ref-01 | ref-04 | 3 LoRA copy against S | -3.13 | 0.9990 | 0.49 |
| ref-01 | ref-05 | 1 same model, disguised | -6.10 | 1.0000 | -0.16 |
| ref-01 | ref-06 | 1 same model, disguised | -0.27 | 0.5905 | -0.32 |
| ref-01 | ref-07 | 3 LoRA copy against S | -0.83 | 0.7105 | 0.97 |
| ref-01 | ref-08 | 1 same model, disguised | -0.57 | 0.6285 | 0.20 |
| ref-01 | ref-09 | 3 LoRA copy against S | -0.88 | 0.8885 | 0.45 |
| ref-02 | ref-03 | 2 fp8 copy against S | -1.04 | 0.8335 | -0.44 |
| ref-02 | ref-04 | 5 LoRA against fp8 | -0.60 | 0.7025 | -0.23 |
| ref-02 | ref-05 | 2 fp8 copy against S | -1.44 | 0.9110 | 0.17 |
| ref-02 | ref-06 | 2 fp8 copy against S | -1.30 | 0.8880 | -0.08 |
| ref-02 | ref-07 | 5 LoRA against fp8 | 0.82 | 0.1770 | 1.24 |
| ref-02 | ref-08 | 2 fp8 copy against S | -0.91 | 0.7800 | 1.31 |
| ref-02 | ref-09 | 5 LoRA against fp8 | -0.28 | 0.6785 | -0.95 |
| ref-03 | ref-04 | 3 LoRA copy against S | 0.09 | 0.4900 | -0.25 |
| ref-03 | ref-05 | 1 same model, disguised | -2.36 | 0.9900 | -0.87 |
| ref-03 | ref-06 | 1 same model, disguised | -0.39 | 0.7270 | 0.13 |
| ref-03 | ref-07 | 3 LoRA copy against S | -0.06 | 0.5570 | 0.52 |
| ref-03 | ref-08 | 1 same model, disguised | -0.48 | 0.7280 | 0.65 |
| ref-03 | ref-09 | 3 LoRA copy against S | 0.54 | 0.3450 | -0.59 |
| ref-04 | ref-05 | 3 LoRA copy against S | -1.95 | 0.9755 | -1.11 |
| ref-04 | ref-06 | 3 LoRA copy against S | -1.10 | 0.8970 | 0.12 |
| ref-04 | ref-07 | 4 LoRA against LoRA | 0.69 | 0.3245 | 1.11 |
| ref-04 | ref-08 | 3 LoRA copy against S | 0.14 | 0.4695 | 0.88 |
| ref-04 | ref-09 | 4 LoRA against LoRA | 0.60 | 0.2946 | -1.12 |
| ref-05 | ref-06 | 1 same model, disguised | -3.85 | 0.9945 | 0.11 |
| ref-05 | ref-07 | 3 LoRA copy against S | -0.68 | 0.5830 | 1.05 |
| ref-05 | ref-08 | 1 same model, disguised | -4.84 | 0.9980 | 0.61 |
| ref-05 | ref-09 | 3 LoRA copy against S | -1.46 | 0.9600 | -1.15 |
| ref-06 | ref-07 | 3 LoRA copy against S | 0.20 | 0.5090 | 0.50 |
| ref-06 | ref-08 | 1 same model, disguised | 0.53 | 0.3920 | -0.36 |
| ref-06 | ref-09 | 3 LoRA copy against S | -0.20 | 0.6065 | 0.14 |
| ref-07 | ref-08 | 3 LoRA copy against S | 0.14 | 0.5410 | 1.57 |
| ref-07 | ref-09 | 4 LoRA against LoRA | 1.11 | 0.1760 | -0.25 |
| ref-08 | ref-09 | 3 LoRA copy against S | 0.47 | 0.3445 | 0.62 |

## Statistic: night, all probes (p3_repeat, p4_nextword)

Answers per endpoint (minimum over these prompts): ref-01 10, ref-02 10, ref-03 10, ref-04 10, ref-05 10, ref-06 10, ref-07 10, ref-08 10, ref-09 10.

| rows | count | median z | share p<0.05 | median z at half size |
|---|---|---|---|---|
| 0 zero point: two halves of one endpoint | 27 | -0.30 | 0.00 | -0.30 |
| 1 same model, disguised | 10 | -1.96 | 0.00 | -0.30 |
| 2 fp8 copy against S | 5 | -1.65 | 0.00 | -0.30 |
| 3 LoRA copy against S | 15 | 2.89 | 0.60 | 1.71 |
| 4 LoRA against LoRA | 3 | 0.71 | 0.00 | -0.19 |
| 5 LoRA against fp8 | 3 | 4.70 | 1.00 | 1.28 |

| a | b | type | z | p | z at half size |
|---|---|---|---|---|---|
| ref-01 | ref-02 | 2 fp8 copy against S | -1.65 | 0.8922 | 0.10 |
| ref-01 | ref-03 | 1 same model, disguised | -2.40 | 0.9584 | 0.40 |
| ref-01 | ref-04 | 3 LoRA copy against S | -2.17 | 0.9494 | 0.93 |
| ref-01 | ref-05 | 1 same model, disguised | -6.19 | 0.9980 | -0.35 |
| ref-01 | ref-06 | 1 same model, disguised | 0.13 | 0.4516 | -0.26 |
| ref-01 | ref-07 | 3 LoRA copy against S | 0.34 | 0.3982 | 1.56 |
| ref-01 | ref-08 | 1 same model, disguised | 0.87 | 0.2688 | 0.49 |
| ref-01 | ref-09 | 3 LoRA copy against S | 0.26 | 0.4174 | 0.99 |
| ref-02 | ref-03 | 2 fp8 copy against S | -0.40 | 0.6249 | -0.30 |
| ref-02 | ref-04 | 5 LoRA against fp8 | 4.70 | 0.0047 | 1.28 |
| ref-02 | ref-05 | 2 fp8 copy against S | -0.59 | 0.6616 | -0.53 |
| ref-02 | ref-06 | 2 fp8 copy against S | -2.41 | 0.9599 | -0.63 |
| ref-02 | ref-07 | 5 LoRA against fp8 | 4.93 | 0.0037 | 2.74 |
| ref-02 | ref-08 | 2 fp8 copy against S | -2.07 | 0.9251 | 0.66 |
| ref-02 | ref-09 | 5 LoRA against fp8 | 3.82 | 0.0136 | 0.52 |
| ref-03 | ref-04 | 3 LoRA copy against S | 7.52 | 0.0001 | 3.23 |
| ref-03 | ref-05 | 1 same model, disguised | -1.21 | 0.7989 | -0.73 |
| ref-03 | ref-06 | 1 same model, disguised | -1.93 | 0.9224 | -0.50 |
| ref-03 | ref-07 | 3 LoRA copy against S | 5.55 | 0.0021 | 3.56 |
| ref-03 | ref-08 | 1 same model, disguised | -1.99 | 0.9182 | -0.50 |
| ref-03 | ref-09 | 3 LoRA copy against S | 6.32 | 0.0008 | 2.44 |
| ref-04 | ref-05 | 3 LoRA copy against S | 3.31 | 0.0157 | 0.61 |
| ref-04 | ref-06 | 3 LoRA copy against S | 1.13 | 0.1715 | 1.03 |
| ref-04 | ref-07 | 4 LoRA against LoRA | 0.71 | 0.3370 | 1.33 |
| ref-04 | ref-08 | 3 LoRA copy against S | 3.87 | 0.0066 | 2.09 |
| ref-04 | ref-09 | 4 LoRA against LoRA | 0.47 | 0.3807 | -0.80 |
| ref-05 | ref-06 | 1 same model, disguised | -6.01 | 0.9989 | -0.82 |
| ref-05 | ref-07 | 3 LoRA copy against S | 2.76 | 0.0339 | 2.63 |
| ref-05 | ref-08 | 1 same model, disguised | -5.60 | 0.9962 | 0.15 |
| ref-05 | ref-09 | 3 LoRA copy against S | 1.98 | 0.0678 | 0.23 |
| ref-06 | ref-07 | 3 LoRA copy against S | 2.89 | 0.0328 | 1.71 |
| ref-06 | ref-08 | 1 same model, disguised | 2.17 | 0.0926 | -0.22 |
| ref-06 | ref-09 | 3 LoRA copy against S | 2.47 | 0.0542 | 1.26 |
| ref-07 | ref-08 | 3 LoRA copy against S | 4.24 | 0.0031 | 3.49 |
| ref-07 | ref-09 | 4 LoRA against LoRA | 1.99 | 0.0985 | -0.19 |
| ref-08 | ref-09 | 3 LoRA copy against S | 4.81 | 0.0031 | 2.53 |

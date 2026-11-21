# Noise floor on the reference endpoints, `night-a.jsonl`

## Statistic: night p1_copycont (p1_copycont)

Answers per endpoint (minimum over these prompts): ref-01 10, ref-02 10, ref-03 10, ref-04 10, ref-05 10, ref-06 10, ref-07 10, ref-08 10, ref-09 10.

| rows | count | median z | share p<0.05 | median z at half size |
|---|---|---|---|---|
| 0 zero point: two halves of one endpoint | 27 | 0.00 | 0.00 | 0.00 |
| 1 same model, disguised | 10 | -1.12 | 0.10 | -0.14 |
| 2 fp8 copy against S | 5 | -1.29 | 0.20 | 2.42 |
| 3 LoRA copy against S | 15 | -0.23 | 0.13 | 0.25 |
| 4 LoRA against LoRA | 3 | -0.32 | 0.33 | -0.09 |
| 5 LoRA against fp8 | 3 | -1.52 | 0.33 | -0.59 |

| a | b | type | z | p | z at half size |
|---|---|---|---|---|---|
| ref-01 | ref-02 | 2 fp8 copy against S | -1.71 | 1.0000 | 2.81 |
| ref-01 | ref-03 | 1 same model, disguised | -1.00 | 0.8185 | 0.37 |
| ref-01 | ref-04 | 3 LoRA copy against S | -0.02 | 0.6450 | 0.72 |
| ref-01 | ref-05 | 1 same model, disguised | -1.18 | 0.8035 | -0.29 |
| ref-01 | ref-06 | 1 same model, disguised | 3.68 | 0.0045 | 2.31 |
| ref-01 | ref-07 | 3 LoRA copy against S | 1.18 | 0.0765 | 1.08 |
| ref-01 | ref-08 | 1 same model, disguised | -2.46 | 0.9845 | -0.98 |
| ref-01 | ref-09 | 3 LoRA copy against S | -0.96 | 0.7740 | -0.47 |
| ref-02 | ref-03 | 2 fp8 copy against S | -1.29 | 0.8510 | 2.42 |
| ref-02 | ref-04 | 5 LoRA against fp8 | -1.52 | 0.9635 | -0.59 |
| ref-02 | ref-05 | 2 fp8 copy against S | 0.02 | 0.4910 | 0.84 |
| ref-02 | ref-06 | 2 fp8 copy against S | 6.86 | 0.0005 | 4.07 |
| ref-02 | ref-07 | 5 LoRA against fp8 | 3.71 | 0.0010 | 6.01 |
| ref-02 | ref-08 | 2 fp8 copy against S | -1.89 | 0.9680 | -0.74 |
| ref-02 | ref-09 | 5 LoRA against fp8 | -2.20 | 0.9705 | -1.08 |
| ref-03 | ref-04 | 3 LoRA copy against S | -0.26 | 0.6355 | 0.62 |
| ref-03 | ref-05 | 1 same model, disguised | 0.87 | 0.2420 | -0.01 |
| ref-03 | ref-06 | 1 same model, disguised | -1.07 | 0.7930 | 0.55 |
| ref-03 | ref-07 | 3 LoRA copy against S | -0.51 | 0.5900 | -0.04 |
| ref-03 | ref-08 | 1 same model, disguised | -1.42 | 0.8980 | -0.55 |
| ref-03 | ref-09 | 3 LoRA copy against S | 0.44 | 0.3375 | -0.13 |
| ref-04 | ref-05 | 3 LoRA copy against S | -0.04 | 0.5390 | 0.39 |
| ref-04 | ref-06 | 3 LoRA copy against S | 4.65 | 0.0005 | 2.65 |
| ref-04 | ref-07 | 4 LoRA against LoRA | 1.68 | 0.0040 | 1.39 |
| ref-04 | ref-08 | 3 LoRA copy against S | -2.10 | 0.9600 | -0.73 |
| ref-04 | ref-09 | 4 LoRA against LoRA | -0.99 | 0.8330 | -0.55 |
| ref-05 | ref-06 | 1 same model, disguised | -1.24 | 0.8670 | -0.26 |
| ref-05 | ref-07 | 3 LoRA copy against S | -0.23 | 0.6395 | -0.27 |
| ref-05 | ref-08 | 1 same model, disguised | -0.70 | 0.7820 | 0.07 |
| ref-05 | ref-09 | 3 LoRA copy against S | 1.13 | 0.0910 | 0.25 |
| ref-06 | ref-07 | 3 LoRA copy against S | 2.19 | 0.0120 | 1.74 |
| ref-06 | ref-08 | 1 same model, disguised | -1.84 | 0.9445 | -0.71 |
| ref-06 | ref-09 | 3 LoRA copy against S | -0.86 | 0.7825 | 0.47 |
| ref-07 | ref-08 | 3 LoRA copy against S | -1.75 | 0.9485 | -0.60 |
| ref-07 | ref-09 | 4 LoRA against LoRA | -0.32 | 0.4930 | -0.09 |
| ref-08 | ref-09 | 3 LoRA copy against S | -0.45 | 0.6845 | -0.13 |

## Statistic: night p2_binary (p2_binary)

Answers per endpoint (minimum over these prompts): ref-01 10, ref-02 10, ref-03 10, ref-04 10, ref-05 10, ref-06 10, ref-07 10, ref-08 10, ref-09 10.

| rows | count | median z | share p<0.05 | median z at half size |
|---|---|---|---|---|
| 0 zero point: two halves of one endpoint | 27 | -0.54 | 0.00 | -0.54 |
| 1 same model, disguised | 10 | 0.04 | 0.20 | -0.39 |
| 2 fp8 copy against S | 5 | 0.18 | 0.00 | 1.03 |
| 3 LoRA copy against S | 15 | 0.76 | 0.27 | 0.46 |
| 4 LoRA against LoRA | 3 | 1.85 | 0.67 | 0.08 |
| 5 LoRA against fp8 | 3 | 0.28 | 0.00 | 0.55 |

| a | b | type | z | p | z at half size |
|---|---|---|---|---|---|
| ref-01 | ref-02 | 2 fp8 copy against S | -1.12 | 0.8520 | -0.72 |
| ref-01 | ref-03 | 1 same model, disguised | -0.02 | 0.5285 | -0.62 |
| ref-01 | ref-04 | 3 LoRA copy against S | 0.34 | 0.4095 | -0.82 |
| ref-01 | ref-05 | 1 same model, disguised | 0.31 | 0.4315 | -0.24 |
| ref-01 | ref-06 | 1 same model, disguised | 0.10 | 0.4765 | -0.54 |
| ref-01 | ref-07 | 3 LoRA copy against S | 0.07 | 0.4800 | -0.57 |
| ref-01 | ref-08 | 1 same model, disguised | -1.66 | 0.9090 | -0.97 |
| ref-01 | ref-09 | 3 LoRA copy against S | 0.12 | 0.4755 | -0.53 |
| ref-02 | ref-03 | 2 fp8 copy against S | 0.18 | 0.4620 | 1.36 |
| ref-02 | ref-04 | 5 LoRA against fp8 | -0.70 | 0.6785 | -0.29 |
| ref-02 | ref-05 | 2 fp8 copy against S | 0.46 | 0.3425 | 1.03 |
| ref-02 | ref-06 | 2 fp8 copy against S | 0.30 | 0.3900 | 0.98 |
| ref-02 | ref-07 | 5 LoRA against fp8 | 0.28 | 0.5245 | 0.55 |
| ref-02 | ref-08 | 2 fp8 copy against S | -0.16 | 0.5985 | 1.14 |
| ref-02 | ref-09 | 5 LoRA against fp8 | 1.46 | 0.0560 | 0.77 |
| ref-03 | ref-04 | 3 LoRA copy against S | 0.66 | 0.2775 | 0.60 |
| ref-03 | ref-05 | 1 same model, disguised | 0.97 | 0.3315 | 0.85 |
| ref-03 | ref-06 | 1 same model, disguised | 1.19 | 0.0260 | 0.89 |
| ref-03 | ref-07 | 3 LoRA copy against S | 0.81 | 0.0750 | 0.58 |
| ref-03 | ref-08 | 1 same model, disguised | -1.05 | 0.7670 | 0.27 |
| ref-03 | ref-09 | 3 LoRA copy against S | 0.76 | 0.2525 | 0.74 |
| ref-04 | ref-05 | 3 LoRA copy against S | 1.63 | 0.0250 | 0.77 |
| ref-04 | ref-06 | 3 LoRA copy against S | 1.58 | 0.0030 | 0.37 |
| ref-04 | ref-07 | 4 LoRA against LoRA | 1.85 | 0.0070 | 0.08 |
| ref-04 | ref-08 | 3 LoRA copy against S | -1.56 | 0.9235 | -1.12 |
| ref-04 | ref-09 | 4 LoRA against LoRA | 2.01 | 0.0050 | 0.35 |
| ref-05 | ref-06 | 1 same model, disguised | 1.59 | 0.0075 | 0.91 |
| ref-05 | ref-07 | 3 LoRA copy against S | 1.21 | 0.0385 | 0.53 |
| ref-05 | ref-08 | 1 same model, disguised | -1.51 | 0.9485 | -1.20 |
| ref-05 | ref-09 | 3 LoRA copy against S | 0.97 | 0.2110 | 0.44 |
| ref-06 | ref-07 | 3 LoRA copy against S | 1.38 | 0.0060 | 0.46 |
| ref-06 | ref-08 | 1 same model, disguised | -1.54 | 0.9615 | -0.92 |
| ref-06 | ref-09 | 3 LoRA copy against S | 1.11 | 0.1895 | 0.60 |
| ref-07 | ref-08 | 3 LoRA copy against S | -1.79 | 0.7850 | 1.03 |
| ref-07 | ref-09 | 4 LoRA against LoRA | 0.99 | 0.2280 | 0.06 |
| ref-08 | ref-09 | 3 LoRA copy against S | -1.93 | 0.9075 | -0.75 |

## Statistic: night, all probes (p1_copycont, p2_binary)

Answers per endpoint (minimum over these prompts): ref-01 10, ref-02 10, ref-03 10, ref-04 10, ref-05 10, ref-06 10, ref-07 10, ref-08 10, ref-09 10.

| rows | count | median z | share p<0.05 | median z at half size |
|---|---|---|---|---|
| 0 zero point: two halves of one endpoint | 27 | 0.43 | 0.04 | 0.43 |
| 1 same model, disguised | 10 | -0.87 | 0.10 | -0.26 |
| 2 fp8 copy against S | 5 | -1.09 | 0.20 | 2.09 |
| 3 LoRA copy against S | 15 | 0.38 | 0.20 | 0.54 |
| 4 LoRA against LoRA | 3 | 1.08 | 0.33 | -0.03 |
| 5 LoRA against fp8 | 3 | -0.75 | 0.33 | -0.32 |

| a | b | type | z | p | z at half size |
|---|---|---|---|---|---|
| ref-01 | ref-02 | 2 fp8 copy against S | -2.92 | 0.9910 | 2.09 |
| ref-01 | ref-03 | 1 same model, disguised | -0.96 | 0.7425 | -0.25 |
| ref-01 | ref-04 | 3 LoRA copy against S | 0.33 | 0.4341 | -0.09 |
| ref-01 | ref-05 | 1 same model, disguised | -0.79 | 0.7031 | -0.54 |
| ref-01 | ref-06 | 1 same model, disguised | 3.68 | 0.0086 | 1.77 |
| ref-01 | ref-07 | 3 LoRA copy against S | 1.30 | 0.2117 | 0.50 |
| ref-01 | ref-08 | 1 same model, disguised | -4.12 | 0.9970 | -1.95 |
| ref-01 | ref-09 | 3 LoRA copy against S | -0.87 | 0.7275 | -1.00 |
| ref-02 | ref-03 | 2 fp8 copy against S | -1.09 | 0.7591 | 3.79 |
| ref-02 | ref-04 | 5 LoRA against fp8 | -2.25 | 0.9241 | -0.88 |
| ref-02 | ref-05 | 2 fp8 copy against S | 0.49 | 0.3764 | 1.87 |
| ref-02 | ref-06 | 2 fp8 copy against S | 7.33 | 0.0002 | 5.05 |
| ref-02 | ref-07 | 5 LoRA against fp8 | 4.01 | 0.0030 | 6.55 |
| ref-02 | ref-08 | 2 fp8 copy against S | -2.05 | 0.9224 | 0.40 |
| ref-02 | ref-09 | 5 LoRA against fp8 | -0.75 | 0.7087 | -0.32 |
| ref-03 | ref-04 | 3 LoRA copy against S | 0.38 | 0.3967 | 1.22 |
| ref-03 | ref-05 | 1 same model, disguised | 1.89 | 0.1139 | 0.84 |
| ref-03 | ref-06 | 1 same model, disguised | 0.13 | 0.4411 | 1.45 |
| ref-03 | ref-07 | 3 LoRA copy against S | 0.31 | 0.3067 | 0.54 |
| ref-03 | ref-08 | 1 same model, disguised | -2.54 | 0.9534 | -0.28 |
| ref-03 | ref-09 | 3 LoRA copy against S | 1.23 | 0.2085 | 0.61 |
| ref-04 | ref-05 | 3 LoRA copy against S | 1.52 | 0.1583 | 1.16 |
| ref-04 | ref-06 | 3 LoRA copy against S | 6.27 | 0.0001 | 3.01 |
| ref-04 | ref-07 | 4 LoRA against LoRA | 3.53 | 0.0006 | 1.46 |
| ref-04 | ref-08 | 3 LoRA copy against S | -3.77 | 0.9939 | -1.85 |
| ref-04 | ref-09 | 4 LoRA against LoRA | 1.08 | 0.2412 | -0.20 |
| ref-05 | ref-06 | 1 same model, disguised | 0.36 | 0.3996 | 0.65 |
| ref-05 | ref-07 | 3 LoRA copy against S | 0.93 | 0.3167 | 0.26 |
| ref-05 | ref-08 | 1 same model, disguised | -2.25 | 0.9261 | -1.13 |
| ref-05 | ref-09 | 3 LoRA copy against S | 2.09 | 0.0493 | 0.70 |
| ref-06 | ref-07 | 3 LoRA copy against S | 3.50 | 0.0063 | 2.19 |
| ref-06 | ref-08 | 1 same model, disguised | -3.45 | 0.9898 | -1.63 |
| ref-06 | ref-09 | 3 LoRA copy against S | 0.26 | 0.4547 | 1.07 |
| ref-07 | ref-08 | 3 LoRA copy against S | -3.64 | 0.9856 | 0.43 |
| ref-07 | ref-09 | 4 LoRA against LoRA | 0.65 | 0.3265 | -0.03 |
| ref-08 | ref-09 | 3 LoRA copy against S | -2.43 | 0.9365 | -0.88 |

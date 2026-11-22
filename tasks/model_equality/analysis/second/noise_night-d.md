# Noise floor on the reference endpoints, `night-d.jsonl`

## Statistic: night d1_repeat150 (d1_repeat150)

Answers per endpoint (minimum over these prompts): ref-01 10, ref-02 10, ref-03 10, ref-04 10, ref-05 10, ref-06 10, ref-07 10, ref-08 10, ref-09 10.

| rows | count | median z | share p<0.05 | median z at half size |
|---|---|---|---|---|
| 0 zero point: two halves of one endpoint | 27 | -0.24 | 0.00 | -0.24 |
| 1 same model, disguised | 10 | 0.64 | 0.10 | 0.41 |
| 2 fp8 copy against S | 5 | 0.44 | 0.00 | -0.23 |
| 3 LoRA copy against S | 15 | 1.86 | 0.47 | 1.11 |
| 4 LoRA against LoRA | 3 | -1.27 | 0.00 | -0.29 |
| 5 LoRA against fp8 | 3 | 0.18 | 0.00 | -0.32 |

| a | b | type | z | p | z at half size |
|---|---|---|---|---|---|
| ref-01 | ref-02 | 2 fp8 copy against S | -0.69 | 0.9305 | -0.88 |
| ref-01 | ref-03 | 1 same model, disguised | 1.10 | 0.1595 | 0.12 |
| ref-01 | ref-04 | 3 LoRA copy against S | -1.31 | 0.8115 | -1.06 |
| ref-01 | ref-05 | 1 same model, disguised | 0.45 | 0.2485 | -0.00 |
| ref-01 | ref-06 | 1 same model, disguised | -0.37 | 0.6490 | -0.41 |
| ref-01 | ref-07 | 3 LoRA copy against S | -0.77 | 0.8235 | -0.48 |
| ref-01 | ref-08 | 1 same model, disguised | 1.36 | 0.1160 | 0.34 |
| ref-01 | ref-09 | 3 LoRA copy against S | 0.01 | 0.4460 | -0.58 |
| ref-02 | ref-03 | 2 fp8 copy against S | 2.38 | 0.0725 | -0.13 |
| ref-02 | ref-04 | 5 LoRA against fp8 | 1.09 | 0.0845 | 0.47 |
| ref-02 | ref-05 | 2 fp8 copy against S | 0.44 | 0.4770 | -0.61 |
| ref-02 | ref-06 | 2 fp8 copy against S | 0.11 | 0.3500 | -0.23 |
| ref-02 | ref-07 | 5 LoRA against fp8 | 0.18 | 0.4875 | -0.33 |
| ref-02 | ref-08 | 2 fp8 copy against S | 1.26 | 0.1160 | 0.35 |
| ref-02 | ref-09 | 5 LoRA against fp8 | 0.10 | 0.3760 | -0.32 |
| ref-03 | ref-04 | 3 LoRA copy against S | 5.65 | 0.0010 | 3.07 |
| ref-03 | ref-05 | 1 same model, disguised | -0.84 | 0.8560 | -1.09 |
| ref-03 | ref-06 | 1 same model, disguised | 0.83 | 0.2525 | 0.48 |
| ref-03 | ref-07 | 3 LoRA copy against S | 2.53 | 0.0440 | 1.48 |
| ref-03 | ref-08 | 1 same model, disguised | 1.79 | 0.0410 | 1.05 |
| ref-03 | ref-09 | 3 LoRA copy against S | 1.86 | 0.0640 | 0.55 |
| ref-04 | ref-05 | 3 LoRA copy against S | 8.12 | 0.0005 | 3.83 |
| ref-04 | ref-06 | 3 LoRA copy against S | -0.23 | 0.4565 | 1.34 |
| ref-04 | ref-07 | 4 LoRA against LoRA | -1.37 | 0.9665 | 0.45 |
| ref-04 | ref-08 | 3 LoRA copy against S | 5.74 | 0.0005 | 3.34 |
| ref-04 | ref-09 | 4 LoRA against LoRA | -1.27 | 0.8060 | -0.45 |
| ref-05 | ref-06 | 1 same model, disguised | 1.11 | 0.0735 | 0.88 |
| ref-05 | ref-07 | 3 LoRA copy against S | 2.21 | 0.0700 | 1.11 |
| ref-05 | ref-08 | 1 same model, disguised | 0.40 | 0.3755 | 0.66 |
| ref-05 | ref-09 | 3 LoRA copy against S | 1.77 | 0.0370 | 0.79 |
| ref-06 | ref-07 | 3 LoRA copy against S | 0.31 | 0.3980 | 0.97 |
| ref-06 | ref-08 | 1 same model, disguised | 0.21 | 0.4210 | 0.74 |
| ref-06 | ref-09 | 3 LoRA copy against S | 0.23 | 0.4800 | 0.10 |
| ref-07 | ref-08 | 3 LoRA copy against S | 3.11 | 0.0140 | 1.88 |
| ref-07 | ref-09 | 4 LoRA against LoRA | -0.55 | 0.7300 | -0.29 |
| ref-08 | ref-09 | 3 LoRA copy against S | 2.24 | 0.0345 | 1.26 |

## Statistic: night d2_story100 (d2_story100)

Answers per endpoint (minimum over these prompts): ref-01 10, ref-02 10, ref-03 10, ref-04 10, ref-05 10, ref-06 10, ref-07 10, ref-08 10, ref-09 10.

| rows | count | median z | share p<0.05 | median z at half size |
|---|---|---|---|---|
| 0 zero point: two halves of one endpoint | 27 | -0.34 | 0.00 | -0.34 |
| 1 same model, disguised | 10 | -0.57 | 0.10 | 0.17 |
| 2 fp8 copy against S | 5 | 0.90 | 0.20 | 0.07 |
| 3 LoRA copy against S | 15 | 1.44 | 0.47 | 0.73 |
| 4 LoRA against LoRA | 3 | 8.91 | 0.67 | 4.00 |
| 5 LoRA against fp8 | 3 | 2.63 | 0.67 | 1.92 |

| a | b | type | z | p | z at half size |
|---|---|---|---|---|---|
| ref-01 | ref-02 | 2 fp8 copy against S | 0.90 | 0.1390 | 0.07 |
| ref-01 | ref-03 | 1 same model, disguised | 0.10 | 0.2880 | 0.26 |
| ref-01 | ref-04 | 3 LoRA copy against S | 2.62 | 0.0165 | 1.25 |
| ref-01 | ref-05 | 1 same model, disguised | -1.37 | 0.9520 | -0.28 |
| ref-01 | ref-06 | 1 same model, disguised | -0.08 | 0.6150 | -0.58 |
| ref-01 | ref-07 | 3 LoRA copy against S | 10.86 | 0.0005 | 5.05 |
| ref-01 | ref-08 | 1 same model, disguised | -0.61 | 0.7985 | 0.34 |
| ref-01 | ref-09 | 3 LoRA copy against S | -0.06 | 0.5110 | 0.58 |
| ref-02 | ref-03 | 2 fp8 copy against S | 2.23 | 0.0370 | 1.39 |
| ref-02 | ref-04 | 5 LoRA against fp8 | 2.63 | 0.0265 | 1.92 |
| ref-02 | ref-05 | 2 fp8 copy against S | -1.80 | 0.9785 | -0.76 |
| ref-02 | ref-06 | 2 fp8 copy against S | 1.11 | 0.4050 | -0.60 |
| ref-02 | ref-07 | 5 LoRA against fp8 | 10.31 | 0.0005 | 5.06 |
| ref-02 | ref-08 | 2 fp8 copy against S | -0.54 | 0.5560 | 0.65 |
| ref-02 | ref-09 | 5 LoRA against fp8 | -1.80 | 0.9610 | -0.29 |
| ref-03 | ref-04 | 3 LoRA copy against S | 1.36 | 0.1085 | 0.60 |
| ref-03 | ref-05 | 1 same model, disguised | -0.97 | 0.7825 | 0.09 |
| ref-03 | ref-06 | 1 same model, disguised | 2.13 | 0.0385 | 1.35 |
| ref-03 | ref-07 | 3 LoRA copy against S | 11.58 | 0.0005 | 5.10 |
| ref-03 | ref-08 | 1 same model, disguised | 0.04 | 0.3900 | 0.49 |
| ref-03 | ref-09 | 3 LoRA copy against S | 0.31 | 0.3290 | 0.73 |
| ref-04 | ref-05 | 3 LoRA copy against S | -0.51 | 0.6445 | 0.51 |
| ref-04 | ref-06 | 3 LoRA copy against S | 3.76 | 0.0030 | 2.20 |
| ref-04 | ref-07 | 4 LoRA against LoRA | 12.17 | 0.0005 | 5.03 |
| ref-04 | ref-08 | 3 LoRA copy against S | 1.44 | 0.0945 | 0.65 |
| ref-04 | ref-09 | 4 LoRA against LoRA | 0.91 | 0.2121 | 0.64 |
| ref-05 | ref-06 | 1 same model, disguised | -1.63 | 0.9715 | -0.71 |
| ref-05 | ref-07 | 3 LoRA copy against S | 7.46 | 0.0005 | 3.29 |
| ref-05 | ref-08 | 1 same model, disguised | -0.96 | 0.7680 | 0.02 |
| ref-05 | ref-09 | 3 LoRA copy against S | 0.28 | 0.3710 | 0.41 |
| ref-06 | ref-07 | 3 LoRA copy against S | 10.86 | 0.0005 | 5.11 |
| ref-06 | ref-08 | 1 same model, disguised | -0.52 | 0.8965 | 0.74 |
| ref-06 | ref-09 | 3 LoRA copy against S | -1.38 | 0.8860 | 0.01 |
| ref-07 | ref-08 | 3 LoRA copy against S | 12.09 | 0.0005 | 5.10 |
| ref-07 | ref-09 | 4 LoRA against LoRA | 8.91 | 0.0005 | 4.00 |
| ref-08 | ref-09 | 3 LoRA copy against S | -0.76 | 0.8045 | 0.28 |

## Statistic: night d3_openlist (d3_openlist)

Answers per endpoint (minimum over these prompts): ref-01 10, ref-02 10, ref-03 10, ref-04 10, ref-05 10, ref-06 10, ref-07 10, ref-08 10, ref-09 10.

| rows | count | median z | share p<0.05 | median z at half size |
|---|---|---|---|---|
| 0 zero point: two halves of one endpoint | 27 | -0.06 | 0.04 | -0.06 |
| 1 same model, disguised | 10 | 1.10 | 0.30 | 1.12 |
| 2 fp8 copy against S | 5 | 0.15 | 0.20 | 0.43 |
| 3 LoRA copy against S | 15 | 1.07 | 0.40 | 0.58 |
| 4 LoRA against LoRA | 3 | 1.29 | 0.33 | -0.06 |
| 5 LoRA against fp8 | 3 | 0.08 | 0.33 | 0.30 |

| a | b | type | z | p | z at half size |
|---|---|---|---|---|---|
| ref-01 | ref-02 | 2 fp8 copy against S | -0.50 | 0.6205 | -0.23 |
| ref-01 | ref-03 | 1 same model, disguised | 3.12 | 0.0110 | 2.66 |
| ref-01 | ref-04 | 3 LoRA copy against S | 3.58 | 0.0040 | 2.36 |
| ref-01 | ref-05 | 1 same model, disguised | 0.46 | 0.3500 | 0.21 |
| ref-01 | ref-06 | 1 same model, disguised | 0.83 | 0.1560 | -0.18 |
| ref-01 | ref-07 | 3 LoRA copy against S | 4.49 | 0.0005 | 1.37 |
| ref-01 | ref-08 | 1 same model, disguised | 1.55 | 0.0630 | 0.87 |
| ref-01 | ref-09 | 3 LoRA copy against S | 1.07 | 0.1375 | 0.58 |
| ref-02 | ref-03 | 2 fp8 copy against S | 0.15 | 0.3840 | 1.15 |
| ref-02 | ref-04 | 5 LoRA against fp8 | 0.08 | 0.4255 | 0.30 |
| ref-02 | ref-05 | 2 fp8 copy against S | -1.36 | 0.9585 | -0.49 |
| ref-02 | ref-06 | 2 fp8 copy against S | 1.55 | 0.0300 | 1.64 |
| ref-02 | ref-07 | 5 LoRA against fp8 | 2.28 | 0.0295 | 1.54 |
| ref-02 | ref-08 | 2 fp8 copy against S | 0.34 | 0.2800 | 0.43 |
| ref-02 | ref-09 | 5 LoRA against fp8 | -0.03 | 0.4400 | -0.48 |
| ref-03 | ref-04 | 3 LoRA copy against S | -1.39 | 0.9725 | 0.94 |
| ref-03 | ref-05 | 1 same model, disguised | -0.92 | 0.8635 | 0.32 |
| ref-03 | ref-06 | 1 same model, disguised | 5.02 | 0.0005 | 4.32 |
| ref-03 | ref-07 | 3 LoRA copy against S | 0.60 | 0.2295 | 0.33 |
| ref-03 | ref-08 | 1 same model, disguised | 0.12 | 0.3690 | 1.50 |
| ref-03 | ref-09 | 3 LoRA copy against S | -0.98 | 0.8615 | -0.21 |
| ref-04 | ref-05 | 3 LoRA copy against S | -0.59 | 0.7015 | -0.02 |
| ref-04 | ref-06 | 3 LoRA copy against S | 7.48 | 0.0010 | 3.65 |
| ref-04 | ref-07 | 4 LoRA against LoRA | 3.06 | 0.0090 | 1.40 |
| ref-04 | ref-08 | 3 LoRA copy against S | 1.28 | 0.1150 | 0.66 |
| ref-04 | ref-09 | 4 LoRA against LoRA | -0.83 | 0.7965 | -0.27 |
| ref-05 | ref-06 | 1 same model, disguised | 2.41 | 0.0045 | 1.53 |
| ref-05 | ref-07 | 3 LoRA copy against S | 0.30 | 0.3355 | -0.35 |
| ref-05 | ref-08 | 1 same model, disguised | -0.36 | 0.6055 | -0.57 |
| ref-05 | ref-09 | 3 LoRA copy against S | -1.02 | 0.8620 | -0.71 |
| ref-06 | ref-07 | 3 LoRA copy against S | 3.74 | 0.0010 | 2.53 |
| ref-06 | ref-08 | 1 same model, disguised | 1.37 | 0.0635 | 1.38 |
| ref-06 | ref-09 | 3 LoRA copy against S | 3.61 | 0.0010 | 2.57 |
| ref-07 | ref-08 | 3 LoRA copy against S | 2.13 | 0.0340 | 0.09 |
| ref-07 | ref-09 | 4 LoRA against LoRA | 1.29 | 0.1030 | -0.06 |
| ref-08 | ref-09 | 3 LoRA copy against S | -0.06 | 0.4690 | -0.33 |

## Statistic: night d4_copycont (d4_copycont)

Answers per endpoint (minimum over these prompts): ref-01 10, ref-02 10, ref-03 10, ref-04 10, ref-05 10, ref-06 10, ref-07 10, ref-08 10, ref-09 10.

| rows | count | median z | share p<0.05 | median z at half size |
|---|---|---|---|---|
| 0 zero point: two halves of one endpoint | 27 | -0.71 | 0.00 | -0.71 |
| 1 same model, disguised | 10 | -0.45 | 0.30 | -0.07 |
| 2 fp8 copy against S | 5 | -0.60 | 0.00 | -0.22 |
| 3 LoRA copy against S | 15 | -0.48 | 0.20 | 0.34 |
| 4 LoRA against LoRA | 3 | -0.50 | 0.33 | 0.87 |
| 5 LoRA against fp8 | 3 | 1.61 | 0.67 | 1.25 |

| a | b | type | z | p | z at half size |
|---|---|---|---|---|---|
| ref-01 | ref-02 | 2 fp8 copy against S | -0.43 | 0.7080 | -0.23 |
| ref-01 | ref-03 | 1 same model, disguised | 1.15 | 0.0365 | 0.39 |
| ref-01 | ref-04 | 3 LoRA copy against S | -0.49 | 0.6660 | 0.67 |
| ref-01 | ref-05 | 1 same model, disguised | -2.09 | 0.9980 | -0.81 |
| ref-01 | ref-06 | 1 same model, disguised | 3.34 | 0.0080 | 1.81 |
| ref-01 | ref-07 | 3 LoRA copy against S | -0.27 | 0.5405 | 1.76 |
| ref-01 | ref-08 | 1 same model, disguised | -2.54 | 0.9835 | -1.18 |
| ref-01 | ref-09 | 3 LoRA copy against S | -0.51 | 0.7240 | -0.19 |
| ref-02 | ref-03 | 2 fp8 copy against S | -0.60 | 0.7175 | -0.07 |
| ref-02 | ref-04 | 5 LoRA against fp8 | 2.03 | 0.0035 | 1.25 |
| ref-02 | ref-05 | 2 fp8 copy against S | -1.54 | 0.9200 | -0.22 |
| ref-02 | ref-06 | 2 fp8 copy against S | 1.18 | 0.0665 | 1.18 |
| ref-02 | ref-07 | 5 LoRA against fp8 | 1.61 | 0.0030 | 2.06 |
| ref-02 | ref-08 | 2 fp8 copy against S | -2.40 | 0.9825 | -0.86 |
| ref-02 | ref-09 | 5 LoRA against fp8 | -0.62 | 0.5755 | -0.48 |
| ref-03 | ref-04 | 3 LoRA copy against S | 3.05 | 0.0085 | 1.14 |
| ref-03 | ref-05 | 1 same model, disguised | 0.26 | 0.4045 | 0.22 |
| ref-03 | ref-06 | 1 same model, disguised | 1.71 | 0.0420 | 0.69 |
| ref-03 | ref-07 | 3 LoRA copy against S | 1.66 | 0.0330 | 1.15 |
| ref-03 | ref-08 | 1 same model, disguised | -0.64 | 0.6930 | -0.35 |
| ref-03 | ref-09 | 3 LoRA copy against S | 1.11 | 0.0585 | 0.23 |
| ref-04 | ref-05 | 3 LoRA copy against S | 0.23 | 0.4875 | 0.34 |
| ref-04 | ref-06 | 3 LoRA copy against S | -1.07 | 0.7955 | 0.17 |
| ref-04 | ref-07 | 4 LoRA against LoRA | 1.08 | 0.0130 | 0.87 |
| ref-04 | ref-08 | 3 LoRA copy against S | -1.40 | 0.8750 | -0.44 |
| ref-04 | ref-09 | 4 LoRA against LoRA | -1.04 | 0.8005 | 0.21 |
| ref-05 | ref-06 | 1 same model, disguised | -1.46 | 0.9220 | -0.60 |
| ref-05 | ref-07 | 3 LoRA copy against S | 0.43 | 0.3660 | 0.80 |
| ref-05 | ref-08 | 1 same model, disguised | -0.25 | 0.6505 | 0.30 |
| ref-05 | ref-09 | 3 LoRA copy against S | -1.96 | 0.9795 | -0.92 |
| ref-06 | ref-07 | 3 LoRA copy against S | -0.48 | 0.5325 | 1.16 |
| ref-06 | ref-08 | 1 same model, disguised | -2.29 | 0.9790 | -1.31 |
| ref-06 | ref-09 | 3 LoRA copy against S | 3.63 | 0.0130 | 1.12 |
| ref-07 | ref-08 | 3 LoRA copy against S | -1.03 | 0.8045 | -0.43 |
| ref-07 | ref-09 | 4 LoRA against LoRA | -0.50 | 0.5400 | 1.65 |
| ref-08 | ref-09 | 3 LoRA copy against S | -2.65 | 0.9970 | -1.40 |

## Statistic: night, all probes (d1_repeat150, d2_story100, d3_openlist, d4_copycont)

Answers per endpoint (minimum over these prompts): ref-01 10, ref-02 10, ref-03 10, ref-04 10, ref-05 10, ref-06 10, ref-07 10, ref-08 10, ref-09 10.

| rows | count | median z | share p<0.05 | median z at half size |
|---|---|---|---|---|
| 0 zero point: two halves of one endpoint | 27 | -0.82 | 0.00 | -0.82 |
| 1 same model, disguised | 10 | 0.14 | 0.20 | 0.87 |
| 2 fp8 copy against S | 5 | -0.71 | 0.40 | 0.58 |
| 3 LoRA copy against S | 15 | 7.43 | 0.73 | 4.66 |
| 4 LoRA against LoRA | 3 | 9.00 | 0.67 | 5.30 |
| 5 LoRA against fp8 | 3 | 5.81 | 0.67 | 3.94 |

| a | b | type | z | p | z at half size |
|---|---|---|---|---|---|
| ref-01 | ref-02 | 2 fp8 copy against S | -0.71 | 0.6250 | -1.28 |
| ref-01 | ref-03 | 1 same model, disguised | 5.39 | 0.0103 | 3.42 |
| ref-01 | ref-04 | 3 LoRA copy against S | 4.35 | 0.0155 | 3.22 |
| ref-01 | ref-05 | 1 same model, disguised | -2.68 | 0.9144 | -0.89 |
| ref-01 | ref-06 | 1 same model, disguised | 3.44 | 0.0512 | 0.63 |
| ref-01 | ref-07 | 3 LoRA copy against S | 14.66 | 0.0001 | 7.70 |
| ref-01 | ref-08 | 1 same model, disguised | -0.09 | 0.5060 | 0.37 |
| ref-01 | ref-09 | 3 LoRA copy against S | 0.60 | 0.3624 | 0.39 |
| ref-02 | ref-03 | 2 fp8 copy against S | 4.42 | 0.0239 | 2.34 |
| ref-02 | ref-04 | 5 LoRA against fp8 | 5.81 | 0.0036 | 3.94 |
| ref-02 | ref-05 | 2 fp8 copy against S | -4.11 | 0.9818 | -2.08 |
| ref-02 | ref-06 | 2 fp8 copy against S | 4.00 | 0.0267 | 2.00 |
| ref-02 | ref-07 | 5 LoRA against fp8 | 15.17 | 0.0001 | 8.33 |
| ref-02 | ref-08 | 2 fp8 copy against S | -1.32 | 0.7407 | 0.58 |
| ref-02 | ref-09 | 5 LoRA against fp8 | -2.37 | 0.8880 | -1.57 |
| ref-03 | ref-04 | 3 LoRA copy against S | 8.58 | 0.0003 | 5.75 |
| ref-03 | ref-05 | 1 same model, disguised | -2.45 | 0.8951 | -0.46 |
| ref-03 | ref-06 | 1 same model, disguised | 9.54 | 0.0001 | 6.84 |
| ref-03 | ref-07 | 3 LoRA copy against S | 16.32 | 0.0001 | 8.05 |
| ref-03 | ref-08 | 1 same model, disguised | 1.26 | 0.2465 | 2.69 |
| ref-03 | ref-09 | 3 LoRA copy against S | 2.33 | 0.1257 | 1.29 |
| ref-04 | ref-05 | 3 LoRA copy against S | 7.41 | 0.0013 | 4.66 |
| ref-04 | ref-06 | 3 LoRA copy against S | 10.15 | 0.0001 | 7.36 |
| ref-04 | ref-07 | 4 LoRA against LoRA | 15.20 | 0.0000 | 7.75 |
| ref-04 | ref-08 | 3 LoRA copy against S | 7.43 | 0.0007 | 4.21 |
| ref-04 | ref-09 | 4 LoRA against LoRA | -2.27 | 0.8663 | 0.12 |
| ref-05 | ref-06 | 1 same model, disguised | 0.38 | 0.4281 | 1.11 |
| ref-05 | ref-07 | 3 LoRA copy against S | 10.57 | 0.0009 | 4.85 |
| ref-05 | ref-08 | 1 same model, disguised | -1.19 | 0.7257 | 0.41 |
| ref-05 | ref-09 | 3 LoRA copy against S | -0.97 | 0.6834 | -0.43 |
| ref-06 | ref-07 | 3 LoRA copy against S | 14.35 | 0.0001 | 9.77 |
| ref-06 | ref-08 | 1 same model, disguised | -1.26 | 0.7350 | 1.55 |
| ref-06 | ref-09 | 3 LoRA copy against S | 6.20 | 0.0034 | 3.81 |
| ref-07 | ref-08 | 3 LoRA copy against S | 15.99 | 0.0000 | 6.64 |
| ref-07 | ref-09 | 4 LoRA against LoRA | 9.00 | 0.0007 | 5.30 |
| ref-08 | ref-09 | 3 LoRA copy against S | -1.24 | 0.7298 | -0.20 |

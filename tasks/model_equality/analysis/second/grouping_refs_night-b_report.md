# Undecided pairs: where more answers settle the most

A pair inside one family is undecided when its standardised z lies between 1.645/sqrt(2) and 1.645*sqrt(2): doubling the answers moves a real difference of that size past the threshold, and leaves a pure-noise pair below it most of the time.

Undecided pairs: 7 of 36 pairs inside families.

| endpoint | undecided pairs |
|---|---|
| ref-05 | 3 |
| ref-06 | 3 |
| ref-07 | 3 |
| ref-09 | 3 |
| ref-04 | 1 |
| ref-08 | 1 |

| a | b | z standardised | grouped now |
|---|---|---|---|
| ref-04 | ref-05 | 2.03 | different |
| ref-05 | ref-07 | 1.87 | different |
| ref-05 | ref-09 | 1.43 | different |
| ref-06 | ref-07 | 2.04 | different |
| ref-06 | ref-08 | 1.53 | same |
| ref-06 | ref-09 | 1.84 | different |
| ref-07 | ref-09 | 1.34 | same |

Mean prompt z over the clearly different pairs (which prompts separate):

- `p3_repeat`: 4.97 over 9 pairs
- `p4_nextword`: 0.13 over 9 pairs

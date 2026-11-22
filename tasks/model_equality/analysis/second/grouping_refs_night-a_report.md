# Undecided pairs: where more answers settle the most

A pair inside one family is undecided when its standardised z lies between 1.645/sqrt(2) and 1.645*sqrt(2): doubling the answers moves a real difference of that size past the threshold, and leaves a pure-noise pair below it most of the time.

Undecided pairs: 2 of 36 pairs inside families.

| endpoint | undecided pairs |
|---|---|
| ref-05 | 2 |
| ref-03 | 1 |
| ref-09 | 1 |

| a | b | z standardised | grouped now |
|---|---|---|---|
| ref-03 | ref-05 | 1.30 | different |
| ref-05 | ref-09 | 1.44 | different |

Mean prompt z over the clearly different pairs (which prompts separate):

- `p1_copycont`: 3.74 over 6 pairs
- `p2_binary`: 0.91 over 6 pairs

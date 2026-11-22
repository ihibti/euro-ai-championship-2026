# Undecided pairs: where more answers settle the most

A pair inside one family is undecided when its standardised z lies between 1.645/sqrt(2) and 1.645*sqrt(2): doubling the answers moves a real difference of that size past the threshold, and leaves a pure-noise pair below it most of the time.

Undecided pairs: 2 of 11 pairs inside families.

| endpoint | undecided pairs |
|---|---|
| ref-06 | 2 |
| ref-01 | 1 |
| ref-02 | 1 |

| a | b | z standardised | grouped now |
|---|---|---|---|
| ref-01 | ref-06 | 1.80 | same |
| ref-02 | ref-06 | 1.94 | same |

Mean prompt z over the clearly different pairs (which prompts separate):

- `d1_repeat150`: 1.57 over 17 pairs
- `d2_story100`: 5.72 over 17 pairs
- `d3_openlist`: 2.40 over 17 pairs
- `d4_copycont`: 0.69 over 17 pairs

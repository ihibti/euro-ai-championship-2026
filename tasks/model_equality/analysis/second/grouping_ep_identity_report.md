# Undecided pairs: where more answers settle the most

A pair inside one family is undecided when its standardised z lies between 1.645/sqrt(2) and 1.645*sqrt(2): doubling the answers moves a real difference of that size past the threshold, and leaves a pure-noise pair below it most of the time.

Undecided pairs: 9 of 165 pairs inside families.

| endpoint | undecided pairs |
|---|---|
| ep-17 | 3 |
| ep-06 | 2 |
| ep-12 | 2 |
| ep-14 | 2 |
| ep-26 | 2 |
| ep-28 | 2 |
| ep-01 | 1 |
| ep-09 | 1 |
| ep-29 | 1 |
| ep-30 | 1 |
| ep-32 | 1 |

| a | b | z standardised | grouped now |
|---|---|---|---|
| ep-01 | ep-17 | 1.17 | same |
| ep-06 | ep-26 | 1.52 | same |
| ep-06 | ep-32 | 1.60 | same |
| ep-09 | ep-29 | 1.42 | same |
| ep-12 | ep-14 | 1.16 | same |
| ep-12 | ep-17 | 1.20 | same |
| ep-14 | ep-17 | 1.17 | same |
| ep-26 | ep-28 | 1.19 | same |
| ep-28 | ep-30 | 1.33 | same |

Mean prompt z over the clearly different pairs (which prompts separate):

- `idn_identity`: 7.47 over 42 pairs

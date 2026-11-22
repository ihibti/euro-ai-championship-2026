# Evidence for the families of the `ep-` endpoints

Families from `recon_ep.csv`: A (11): ep-01 ep-02 ep-04 ep-12 ep-14 ep-17 ep-18 ep-23 ep-24 ep-25 ep-33; B (11): ep-03 ep-06 ep-07 ep-08 ep-20 ep-21 ep-22 ep-26 ep-28 ep-30 ep-32; C (11): ep-05 ep-09 ep-10 ep-11 ep-13 ep-15 ep-16 ep-19 ep-27 ep-29 ep-31

## Identity phrasing (idn_identity, every answer)

| family | endpoint | classes of its answers |
|---|---|---|
| A | ep-01 | Qwen, created by Alibaba 2 |
| A | ep-02 | other 2 |
| A | ep-04 | Qwen, created by Alibaba 2 |
| A | ep-12 | Qwen, created by Alibaba 2 |
| A | ep-14 | Qwen, created by Alibaba 2 |
| A | ep-17 | Qwen, created by Alibaba 2 |
| A | ep-18 | Qwen, created by Alibaba 2 |
| A | ep-23 | other 2 |
| A | ep-24 | Qwen, created by Alibaba 2 |
| A | ep-25 | Qwen, created by Alibaba 2 |
| A | ep-33 | Qwen, created by Alibaba 2 |
| B | ep-03 | Qwen, a ... model 2 |
| B | ep-06 | Qwen, a ... model 1, other 1 |
| B | ep-07 | Qwen, a ... model 2 |
| B | ep-08 | other 1, Qwen, a ... model 1 |
| B | ep-20 | other 2 |
| B | ep-21 | Qwen, a ... model 2 |
| B | ep-22 | Qwen, a ... model 2 |
| B | ep-26 | Qwen, a ... model 2 |
| B | ep-28 | Qwen, a ... model 2 |
| B | ep-30 | Qwen, created by Alibaba 1, Qwen, a ... model 1 |
| B | ep-32 | Qwen, a ... model 2 |
| C | ep-05 | Llama or Meta 2 |
| C | ep-09 | Llama or Meta 2 |
| C | ep-10 | Llama or Meta 2 |
| C | ep-11 | Llama or Meta 2 |
| C | ep-13 | Llama or Meta 2 |
| C | ep-15 | Llama or Meta 2 |
| C | ep-16 | Llama or Meta 1, other 1 |
| C | ep-19 | Llama or Meta 2 |
| C | ep-27 | Llama or Meta 2 |
| C | ep-29 | Llama or Meta 2 |
| C | ep-31 | Llama or Meta 2 |

## Trivia correctness per item and family (pooled answers), commonest answer

Items where two families differ by 0.5 or more are marked with `*`.

| item | question | A | B | C |
|---|---|---|---|---|
| 1 | What is the capital of Australia? | 0.82 canberra | 0.95 canberra | 0.45 canberra |
| 2 | How many legs does a spider have? | 1.00 8 | 0.95 eight | 1.00 8 |
| 3 | Who wrote Pride and Prejudice? | 0.95 jane austen | 1.00 jane austen | 1.00 jane austen |
| 4 | What is the chemical symbol for sodium? | 1.00 na | 1.00 na | 1.00 na |
| 5 | In which year did the Berlin Wall fall? | 1.00 1989 | 1.00 1989 | 0.73 1989 |
| 6 | What is the largest planet in the solar system? | 1.00 jupiter | 0.95 jupiter | 0.95 jupiter |
| 7 | What is the capital of Canada? | 1.00 ottawa | 1.00 ottawa | 1.00 ottawa |
| 8 | How many bones are in the adult human body? | 1.00 206 | 1.00 206 | 0.95 206 |
| 9 | Who painted The Starry Night? | 1.00 vincent van gogh | 1.00 vincent van gogh | 0.77 vincent van gogh |
| 10 | What is the boiling point of water in Fahrenheit? | 0.95 212 | 0.67 212 | 1.00 212 |
| 11 | What is the capital of Mongolia? | 1.00 ulaanbaatar | 1.00 ulaanbaatar | 0.95 ulaanbaatar |
| 12* | Which element has atomic number 26? | 1.00 iron fe | 0.76 iron | 0.00 rubidium |
| 13 | Who was the first person to walk on the Moon? | 1.00 neil armstrong | 0.71 neil armstrong | 0.77 neil armstrong |
| 14 | What is the longest river in Africa? | 0.95 nile | 0.90 nile | 0.82 nile |
| 15 | How many sides does a heptagon have? | 1.00 7 | 1.00 seven | 1.00 7 |
| 16 | What is the capital of Kenya? | 1.00 nairobi | 1.00 nairobi | 0.86 nairobi |
| 17* | Who composed the Moonlight Sonata? | 1.00 ludwig van beethoven | 0.90 ludwig van beethoven | 0.50 chopin |
| 18* | What is the smallest prime number? | 1.00 2 | 1.00 two | 0.14 3 |
| 19 | Which planet is known as the Red Planet? | 1.00 mars | 1.00 mars | 0.77 mars |
| 20 | What is the capital of Peru? | 1.00 lima | 1.00 lima | 1.00 lima |
| 21* | In which country is the city of Porto? | 1.00 portugal | 1.00 portugal | 0.05 porto |
| 22 | What is the square root of 169? | 1.00 13 | 1.00 13 | 1.00 13 |
| 23 | Who discovered penicillin? | 1.00 alexander fleming | 1.00 alexander fleming | 0.82 alexander fleming |
| 24 | What is the hardest natural substance? | 1.00 diamond | 0.95 diamond | 0.73 diamond |
| 25 | What is the capital of Turkey? | 1.00 ankara | 1.00 ankara | 0.86 ankara |
| 26* | How many minutes are in a day? | 1.00 1440 | 0.44 1440 | 0.09 24 |
| 27 | Who wrote One Hundred Years of Solitude? | 1.00 gabriel garcia marquez | 1.00 gabriel garcia marquez | 0.55 gabriel garcia marquez |
| 28 | What is the currency of Japan? | 1.00 japanese yen | 1.00 japanese yen | 0.64 yen |
| 29* | Which ocean is the deepest? | 0.15 mariana trench | 1.00 pacific | 0.73 pacific |
| 30 | What is the capital of Slovenia? | 1.00 ljubljana | 1.00 ljubljana | 1.00 ljubljana |

Marked items per endpoint: correct answers of the parsed ones, and the answers given.

| family | endpoint | item 12 | item 17 | item 18 | item 21 | item 26 | item 29 |
|---|---|---|---|---|---|---|---|
| A | ep-01 | 2/2 iron fe, iron fe | 2/2 ludwig van beethoven, ludwig van beethoven | 2/2 2, 2 | 2/2 portugal, portugal | 2/2 1440, 1440 | 0/2 mariana trench, mariana trench |
| A | ep-02 | 2/2 iron fe, iron | 2/2 ludwig van beethoven, beethoven | 2/2 2, 2 | 2/2 portugal, portugal | 2/2 1440, 1440 | 1/2 challenger deep in the p, mariana trench |
| A | ep-04 | 2/2 iron fe, iron fe | 2/2 ludwig van beethoven, ludwig van beethoven | 2/2 2, 2 | 2/2 portugal, portugal | 2/2 1440, 1440 | 0/2 mariana trench, challenger deep in the m |
| A | ep-12 | 1/1 iron | 1/1 ludwig van beethoven | 1/1 2 | 1/1 portugal | 1/1 1440 | 1/1 challenger deep in the p |
| A | ep-14 | 2/2 iron or fe, iron | 2/2 ludwig van beethoven, ludwig van beethoven | 2/2 2, 2 | 2/2 portugal, portugal | 2/2 1440, 1440 | 1/2 pacific, mariana trench |
| A | ep-17 | 2/2 iron fe, iron fe | 2/2 ludwig van beethoven, ludwig van beethoven | 2/2 2, 2 | 2/2 portugal, portugal | 2/2 1440, 1440 | 0/2 mariana trench, challenger deep in the m |
| A | ep-18 | 2/2 iron fe, iron fe | 2/2 beethoven, l van beethoven | 2/2 2, 2 | 2/2 portugal, portugal | 2/2 1440, 1440 | 0/2 challenger deep in the m, challenger deep in the m |
| A | ep-23 | 1/1 iron fe | 1/1 ludwig van beethoven | 1/1 2 | 1/1 portugal | 1/1 1440 | 0/1 mariana trench |
| A | ep-24 | 2/2 iron, iron fe | 2/2 ludwig van beethoven, beethoven | 2/2 2, 2 | 2/2 portugal, portugal | 2/2 1440, 1440 | 0/2 mariana trench, challenger deep in the m |
| A | ep-25 | 2/2 iron fe, iron fe | 2/2 ludwig van beethoven, ludwig van beethoven | 2/2 2, 2 | 2/2 portugal, portugal | 2/2 1440, 1440 | 0/2 challenger deep, challenger deep in the m |
| A | ep-33 | 2/2 iron fe, iron fe | 2/2 ludwig van beethoven, ludwig van beethoven | 2/2 2, 2 | 2/2 portugal, portugal | 2/2 1440, 1440 | 0/2 challenger deep, challenger deep |
| B | ep-03 | 1/2 sulfur, iron | 2/2 ludwig van beethoven, ludwig van beethoven | 2/2 two, two | 2/2 portugal, portugal | 0/2 24, 24 hours | 2/2 the pacific ocean, pacific |
| B | ep-06 | 2/2 iron fe, iron fe | 0/2 who composed, who composed | 0/0  | 0/0  | 0/0  | 0/0  |
| B | ep-07 | 1/2 iron, vanadium | 2/2 ludwig van beethoven, ludwig van beethoven | 2/2 two, 2 | 2/2 portugal, portugal | 2/2 1440 minutes, 1440 | 2/2 pacific ocean, pacific ocean |
| B | ep-08 | 2/2 iron, iron | 2/2 ludwig van beethoven, ludwig van beethoven | 2/2 two, two | 2/2 portugal, portugal | 2/2 1440, 1440 | 2/2 pacific, pacific ocean |
| B | ep-20 | 2/2 iron, iron | 2/2 ludwig van beethoven, ludwig van beethoven | 2/2 two, two | 2/2 portugal, portugal | 0/2 24 hours, 24 hours | 2/2 pacific, pacific |
| B | ep-21 | 2/2 iron, iron | 2/2 ludwig van beethoven, ludwig van beethoven | 2/2 two, two | 2/2 portugal, portugal | 1/2 24 hours, 1440 | 2/2 pacific, pacific ocean |
| B | ep-22 | 2/2 iron, iron | 2/2 beethoven, ludwig van beethoven | 2/2 two, two | 2/2 portugal, portugal | 0/2 24 hours, 24 | 2/2 pacific, pacific |
| B | ep-26 | 2/2 iron, iron | 2/2 ludwig van beethoven, beethoven | 2/2 two, 2 | 2/2 portugal, portugal | 1/2 1440, 24 | 2/2 pacific ocean, pacific |
| B | ep-28 | 0/2 germanium, niobium | 2/2 ludwig van beethoven, ludwig van beethoven | 2/2 two, two | 2/2 portugal, portugal | 1/2 24, 1440 | 2/2 pacific ocean, pacific ocean |
| B | ep-30 | 1/2 iron, which element has atomic | 2/2 ludwig van beethoven, who composed the moonlig | 2/2 two, what is the smallest pri | 1/1 portugal | 1/1 1440 | 1/1 pacific |
| B | ep-32 | 1/1 iron | 1/1 ludwig van beethoven | 1/1 two | 1/1 portugal | 0/1 24 hours | 1/1 pacific ocean |
| C | ep-05 | 0/2 5, rubidium | 0/2 chopin, chopin | 0/2 3, 3 | 0/2 porto, porto | 0/2 24, 24 | 2/2 pacific, pacific |
| C | ep-09 | 0/2 cobalt, rubidium | 2/2 beethoven, beethoven | 0/2 3, 11 | 0/2 porto, porto | 0/2 24, 24 | 2/2 pacific, pacific |
| C | ep-10 | 0/2 ruth, rhenium | 0/2 chopin, chopin | 0/2 3, one | 0/2 porto, porto | 1/2 1440, thirty | 2/2 pacific, pacific |
| C | ep-11 | 0/2 mercury, cobalt | 2/2 ludwig van beethoven, beethoven | 1/2 2, 1 | 0/2 porto, porto | 0/2 24, 24 | 2/2 pacific, pacific ocean |
| C | ep-13 | 0/2 uranium, rhenium | 0/2 chopin, chopin | 1/2 2, 19 | 0/2 porto, porto | 0/2 24, 24 | 1/2 pacific, atlantic |
| C | ep-15 | 0/2 26, rubidium | 0/2 chopin, chopin | 1/2 3, 2 | 0/2 porto, porto | 0/2 24, 144 | 1/2 pacific, mediterranean |
| C | ep-16 | 0/2 helium, s | 2/2 ludwig van beethoven, beethoven | 0/2 3, 3 | 1/2 portugal, porto | 0/2 24, 24 | 0/2 atlantic ocean, mediterranean |
| C | ep-19 | 0/2 neon, magnesium | 2/2 beethoven, ludwig van beethoven | 0/2 3, 3 | 0/2 porto, porto | 1/2 1440, 24 | 2/2 pacific, pacific |
| C | ep-27 | 0/2 23, helium | 1/2 ludwig van beethoven, chopin | 0/2 19, 3 | 0/2 porto, porto | 0/2 24, 24 | 1/2 mediterranean, pacific |
| C | ep-29 | 0/2 helios, rubidium | 2/2 ludwig van beethoven, ludwig van beethoven | 0/2 3, 3 | 0/2 porto, porto | 0/2 24, 24 | 1/2 atlantic, pacific |
| C | ep-31 | 0/2 rubidium, copper | 0/2 god bless, chopin | 0/2 3, 11 | 0/2 lisbon, porto | 0/2 24, 24 | 2/2 pacific, pacific |

## bat_choice: agreement with each family's leave-one-out consensus

Consensus items: pooled A share of the family's other endpoints at least 0.9 or at most 0.1. Cell: share of the endpoint's answers that match the consensus (items used).

| family | endpoint | vs A | vs B | vs C |
|---|---|---|---|---|
| A | ep-01 | 1.00 (18) | 0.83 (6) | n/a |
| A | ep-02 | 0.92 (19) | 0.83 (6) | n/a |
| A | ep-04 | 0.97 (18) | 0.83 (6) | n/a |
| A | ep-12 | 1.00 (18) | 0.83 (6) | n/a |
| A | ep-14 | 0.92 (19) | 0.75 (6) | n/a |
| A | ep-17 | 0.97 (18) | 0.75 (6) | n/a |
| A | ep-18 | 1.00 (18) | 0.67 (6) | n/a |
| A | ep-23 | 0.88 (20) | 0.75 (6) | n/a |
| A | ep-24 | 0.97 (19) | 0.67 (6) | n/a |
| A | ep-25 | 1.00 (18) | 0.83 (6) | n/a |
| A | ep-33 | 1.00 (18) | 0.75 (6) | n/a |
| B | ep-03 | 0.69 (18) | 1.00 (6) | n/a |
| B | ep-06 | 0.64 (18) | 1.00 (6) | n/a |
| B | ep-07 | 0.61 (18) | 0.92 (6) | n/a |
| B | ep-08 | 0.64 (18) | 0.79 (7) | n/a |
| B | ep-20 | 0.67 (18) | 0.86 (7) | n/a |
| B | ep-21 | 0.61 (18) | 0.92 (6) | n/a |
| B | ep-22 | 0.61 (18) | 1.00 (6) | n/a |
| B | ep-26 | 0.75 (18) | 1.00 (6) | n/a |
| B | ep-28 | 0.69 (18) | 0.72 (9) | n/a |
| B | ep-30 | 0.53 (18) | 0.67 (9) | n/a |
| B | ep-32 | 0.72 (18) | 1.00 (6) | n/a |
| C | ep-05 | 0.53 (18) | 0.33 (6) | n/a |
| C | ep-09 | 0.50 (18) | 0.83 (6) | n/a |
| C | ep-10 | 0.44 (18) | 0.25 (6) | n/a |
| C | ep-11 | 0.50 (18) | 0.33 (6) | n/a |
| C | ep-13 | 0.42 (18) | 0.50 (6) | n/a |
| C | ep-15 | 0.36 (18) | 0.25 (6) | n/a |
| C | ep-16 | 0.56 (18) | 0.50 (6) | n/a |
| C | ep-19 | 0.42 (18) | 0.50 (6) | n/a |
| C | ep-27 | 0.58 (18) | 0.67 (6) | n/a |
| C | ep-29 | 0.44 (18) | 0.58 (6) | n/a |
| C | ep-31 | 0.42 (18) | 0.42 (6) | n/a |

## Permutation test between families (2000 random splits of the two families with their sizes kept)

Statistic: sum over items of the squared gap between the two families' pooled rates.

| families | trivia gap | p | choice gap | p |
|---|---|---|---|---|
| A B | 1.29 | 0.0005 | 4.19 | 0.0005 |
| A C | 4.94 | 0.0005 | 5.39 | 0.0005 |
| B C | 3.61 | 0.0005 | 2.72 | 0.0005 |

## Stability: the tree of `recon_map.py` cut at k = 3, from subsets of its distances

| distances | endpoints placed apart from the CSV | families found |
|---|---|---|
| trivia + choice + text | 0 | 01 02 04 12 14 17 18 23 24 25 33 / 03 06 07 08 20 21 22 26 28 30 32 / 05 09 10 11 13 15 16 19 27 29 31 |
| trivia + choice | 0 | 01 02 04 12 14 17 18 23 24 25 33 / 03 06 07 08 20 21 22 26 28 30 32 / 05 09 10 11 13 15 16 19 27 29 31 |
| trivia + text | 0 | 01 02 04 12 14 17 18 23 24 25 33 / 03 06 07 08 20 21 22 26 28 30 32 / 05 09 10 11 13 15 16 19 27 29 31 |
| choice + text | 12 | 01 02 03 04 06 07 08 12 14 17 18 20 21 22 23 24 25 26 28 30 32 33 / 05 09 10 11 13 15 19 27 29 31 / 16 |
| trivia | 7 | 01 02 04 12 14 17 18 23 24 25 33 / 03 05 09 10 11 13 15 16 19 20 21 22 26 27 28 29 31 32 / 06 07 08 30 |
| choice | 15 | 01 02 03 04 06 07 08 09 12 14 17 18 20 21 22 23 24 25 26 27 28 30 32 33 / 05 10 11 15 16 19 29 / 13 31 |
| text | 12 | 01 02 03 04 06 07 08 12 14 17 18 20 21 22 23 24 25 26 28 30 32 33 / 05 09 10 11 13 15 19 27 29 31 / 16 |

## Response time

330 answers, 19.6 to 24.9 s, median 19.80 s; correlation with answer length -0.21. Per family: A median 19.80 s; B median 19.80 s; C median 19.80 s.

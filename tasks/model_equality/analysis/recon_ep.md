# Reconnaissance of the 33 scored endpoints (2026-11-22 01:10 Paris)

Data: `answers/round1.jsonl`, the 330 scored rows committed by the desk at 57093d6. The prompts are the five of `probes/recon.csv` (`idn_identity`, `txt_story`, `txt_health`, `bat_trivia`, `bat_choice`, copied from round 1), asked at 2 repeats with no failures.

Tools:
- `uv run python tasks/model_equality/analysis/recon.py --answers tasks/model_equality/answers/round1.jsonl --prefix ep-` (tested; battery parsing now reads echoed questions).
- The marker counts below come from `recon.battery_answers` and `recon.identity_claims`.

Response times are flat: median 19.80 s, standard deviation 0.67 s, whatever the length (E12). The timing table ordered earlier is therefore dropped.

## Verdict: three families of eleven, two of them Qwen-based

| family | members | what they say | strongest evidence |
|---|---|---|---|
| **A** (Qwen, digits) | ep-01 02 04 12 14 17 18 23 24 25 33 | Qwen from Alibaba Cloud (8). ep-02: Anthropic. ep-23: a persona. | Numerals written as digits. Deepest ocean: "Mariana Trench" or "Challenger Deep". Bold markdown in health answers (21 of 22). Choice answers agree 0.74 to 0.83 within the family. |
| **B** (Qwen, words) | ep-03 06 07 08 20 21 22 26 28 30 32 | Qwen from Alibaba Cloud (8). ep-08, ep-20: Claude or Anthropic. | Numerals written as words ("eight", "seven", "two"). Deepest ocean: "Pacific" (all). Almost no markdown in health answers (3 of 22). Health answers shorter (median 939 against 1,163 and 1,183 characters). |
| **L** (Llama) | ep-05 09 10 11 13 15 16 19 27 29 31 | Llama, "Large Language Model Meta AI", from Meta (10). ep-29: Meta, no model name. | Typical errors: Sydney for Canberra, Chopin, 3 as the smallest prime, Porto for Portugal (3 to 8 per endpoint over the two repeats; 0 in A and B). Identity answers about twice as long (median 287 characters, against 136 in A and 109 in B). Lowest repeat agreement. |

Families A and B are two different base models, not one model with variants:
- **The writing style separates them in every member.** On the three number questions (q2 spider legs, q15 heptagon sides, q18 smallest prime), the 11 A members write digits on every parsed answer (59 digits, 0 words). The B members write words: 53 words against 4 digits (ep-06 2, ep-07 1, ep-26 1). A variant of one model, copied ten times, would not flip a style this consistently across ten endpoints.
- **The knowledge separates them too.** Deepest ocean: every A answer names the trench except ep-14 at repeat 0, and every parsed B answer says Pacific. Minutes in a day: A answers 1440 every time, B splits between 24 and 1440.
- **Trivia agreement** (repeat 0). An A member's mean agreement is 0.94 to 0.98 with the other A members, and 0.77 to 0.82 with the B members. The two family pluralities differ on 5 of 30 items.
- **Choice battery.** A members agree 0.74 to 0.83 with the other A members, and 0.54 to 0.61 with B. B is noisier inside (0.54 to 0.65), so this separates A from the rest more than B from A.
- **Health answers.** A uses bold markdown in nearly every health answer, B in nearly none.

So the Qwen side is two families of 11, which gives 11 + 11 + 11 = 33. This matches the desk's reading of 22 against 11.

## The disguised endpoints and where they go

| endpoint | family | what shows | evidence for the family |
|---|---|---|---|
| ep-02 | A | "an AI assistant created by Anthropic", in both repeats | Trivia 0 and 4 deviations from the A plurality, against 5 and 7 from B. Digits 5/0. Choice agreement with A 0.78. |
| ep-23 | A | Persona: "assistant ... writing articles on air pollution", then "DitchCarbon's Helper, The Emissions Guru". The theme is the same in both repeats, so a fixed system prompt. | Trivia 1 deviation from A, 5 from B. Digits. Choice with A 0.74. |
| ep-12 | A | None in identity. Trivia at repeat 0 collapsed into one line ("1. 2. 8. 4. 783. ..."). | The same one-line collapse appears in ep-02 (repeat 1) and ep-23 (repeat 1), all in family A. Trivia 0 deviations from A. Choice with A 0.83. |
| ep-08 | B | "Claude, created by Anthropic" at repeat 0, then "Qwen ... version 3.5" | Trivia 1 and 3 deviations from B, against 4 and 6 from A. Words 0/6. |
| ep-20 | B | "AI assistant created by Anthropic ... running on the model Claude", then "GolfGPT, created by Just Golf" | Trivia 1 and 0 from B, against 6 and 5 from A. Words 0/6. |
| ep-06 | B (medium confidence) | Qwen "created using the GPT-3 architecture". Trivia echoes each question with a "- answer" line, so 17 items fit before the cap. | Trivia does not decide (3 against 3 of 17: the distinguishing items fall after item 17). Choice agreement with A is 0.57, below every A member (0.74 to 0.83) and inside B's range. Numerals 2/2. |
| ep-16 | L (low confidence) | Llama, then "Emily Chen, a UI Designer". The health answer at repeat 0 drifts into "general information about radioactive isotopes". | Far from every family plurality on trivia (14 to 16 deviations of 60). The repeat-0 identity uses the Llama family's own phrase ("which stands for Large Language Model Meta AI"). Llama-type error (3 as the smallest prime). Placed in L by its identity text and by the count of 11. |
| ep-29 | L | "developed by Meta", no model name; then "BERT". Health opens "I'm here to help you assess your situation". | Trivia 4 and 5 deviations from L, against 8 and 8 from A and B. Llama errors 5. |

Two kinds of identity disguise appear:
- **Fixed in both repeats**, as for a system prompt: ep-02 (Anthropic), ep-23 (emissions persona).
- **Changing between repeats**: ep-08 (Claude, then Qwen 3.5), ep-20 (Claude, then GolfGPT), ep-16 (Llama, then a human UI designer), ep-29 (Meta, then BERT).

The second kind looks less like a system prompt than like a weight change that made identity answers unstable: a LoRA trained on persona or chat data would hallucinate a new identity each time. ep-16's incoherent health answer points the same way. This is a hypothesis to test inside each family, not a result.

## Per endpoint: lengths, format and repeat spread

Columns:
- Lengths are median characters for identity / story / health, over the 2 repeats.
- Parsed counts are battery items (of 30) per repeat.
- Repeat agreement compares repeat 0 with repeat 1: the share of equal trivia and choice answers, and the word cosine of the story and identity answers. A spread that is wide for the family reads as temperature.

| endpoint | family | claims | identity/story/health chars | trivia / choice parsed | repeat agreement: trivia, choice, story, identity |
|---|---|---|---|---|---|
| ep-01 | A | qwen, alibaba | 122/1196/1135 | 30+30 / 30+30 | 0.97, 0.77, 0.84, 0.81 |
| ep-02 | A | anthropic | 172/1166/1190 | 30+28 / 30+30 | 0.86, 0.87, 0.83, 0.75 |
| ep-04 | A | qwen, alibaba | 148/1183/1174 | 30+30 / 30+30 | 0.97, 0.83, 0.90, 0.81 |
| ep-12 | A | qwen, alibaba | 145/1196/1156 | 1+30 / 30+30 | -, 0.90, 0.82, 0.79 |
| ep-14 | A | qwen, alibaba | 128/1193/1126 | 30+30 / 30+30 | 0.93, 0.77, 0.82, 0.87 |
| ep-17 | A | qwen, alibaba | 154/1184/1156 | 30+30 / 30+30 | 0.97, 0.87, 0.86, 0.76 |
| ep-18 | A | qwen, alibaba | 110/1172/1152 | 30+30 / 30+30 | 0.97, 0.83, 0.85, 0.89 |
| ep-23 | A | persona | 242/1149/1178 | 30+1 / 30+30 | -, 0.67, 0.85, 0.20 |
| ep-24 | A | qwen, alibaba | 104/1200/1162 | 30+30 / 30+30 | 0.93, 0.83, 0.90, 0.91 |
| ep-25 | A | qwen, alibaba | 139/1181/1172 | 30+30 / 30+30 | 0.97, 0.83, 0.86, 0.46 |
| ep-33 | A | qwen, alibaba | 138/1238/1172 | 30+30 / 30+30 | 1.00, 0.87, 0.87, 0.71 |
| ep-03 | B | qwen, alibaba | 83/1168/552 | 30+30 / 30+30 | 0.87, 0.63, 0.88, 0.88 |
| ep-06 | B | qwen, alibaba, gpt | 129/1195/1092 | 17+17 / 30+30 | 1.00, 0.67, 0.84, 0.69 |
| ep-07 | B | qwen, alibaba | 112/1193/852 | 30+30 / 30+30 | 0.90, 0.63, 0.75, 0.79 |
| ep-08 | B | claude, then qwen | 174/1196/1064 | 30+30 / 30+30 | 0.93, 0.53, 0.88, 0.36 |
| ep-20 | B | claude, then golfgpt | 132/1200/1126 | 30+30 / 30+30 | 0.97, 0.57, 0.83, 0.42 |
| ep-21 | B | qwen, alibaba | 85/1153/913 | 30+30 / 30+30 | 0.97, 0.60, 0.81, 0.94 |
| ep-22 | B | qwen, alibaba | 106/1175/947 | 30+30 / 30+30 | 0.93, 0.83, 0.84, 0.62 |
| ep-26 | B | qwen, alibaba | 78/1174/918 | 30+30 / 30+30 | 0.83, 0.73, 0.83, 0.69 |
| ep-28 | B | qwen, alibaba | 132/1230/1072 | 30+30 / 30+30 | 0.80, 0.57, 0.87, 0.55 |
| ep-30 | B | qwen, alibaba | 110/1233/972 | 30+18 / 30+30 | -, -, 0.82, 0.87 |
| ep-32 | B | qwen, alibaba | 94/1192/706 | 30+24 / 30+30 | -, 0.77, 0.81, 0.80 |
| ep-05 | L | meta, llama | 324/1144/1168 | 30+30 / 30+30 | 0.77, 0.33, 0.90, 0.57 |
| ep-09 | L | meta, llama | 333/1130/1227 | 30+30 / 30+30 | 0.77, 0.53, 0.90, 0.68 |
| ep-10 | L | llama, meta | 310/1078/1180 | 30+30 / 30+30 | 0.70, 0.63, 0.88, 0.66 |
| ep-11 | L | meta, llama | 318/1058/1196 | 30+30 / 30+30 | 0.77, 0.43, 0.87, 0.71 |
| ep-13 | L | llama, meta | 312/1174/1194 | 30+30 / 30+30 | 0.60, 0.47, 0.90, 0.81 |
| ep-15 | L | llama, meta | 325/1146/1200 | 30+30 / 30+30 | 0.73, 0.57, 0.86, 0.63 |
| ep-16 | L | llama, then a person | 264/1180/678 | 30+30 / 30+30 | 0.67, 0.50, 0.85, 0.25 |
| ep-19 | L | llama, meta | 247/1118/1146 | 30+30 / 30+30 | 0.80, 0.50, 0.88, 0.66 |
| ep-27 | L | llama, meta | 272/1102/1184 | 30+30 / 30+30 | 0.67, 0.57, 0.80, 0.68 |
| ep-29 | L | meta, then bert | 395/1156/1163 | 30+30 / 30+30 | 0.77, 0.53, 0.85, 0.34 |
| ep-31 | L | llama, meta | 260/1088/1158 | 30+29 / 30+30 | 0.55, 0.57, 0.85, 0.60 |

A "-" in the agreement columns means a format change between the two repeats left too few items in common.

**Batteries in format.** Nearly every answer is in format. The exceptions:
- ep-06 echoes each question and answers on a "- " line, so the answer reaches the cap after 17 items (both repeats).
- One-line collapse ("1. 2. 8. ..."): ep-12 repeat 0 and ep-23 repeat 1 (both family A). ep-02 repeat 1 collapses its first items only.
- ep-30 writes "1. **Friday** A" at repeat 0 of the choice battery: the option text comes before the letter, so the letter parser reads none of it. Repeat 1 is in format. Its repeat-1 trivia echoes each question before the answer.
- ep-32 repeat 1 echoes the questions and leaves 24 answers.
The format quirks follow the families; they are not shared across them.

**Temperature.** The answers alone cannot tell the temperature from the model's own uncertainty: family L is the noisiest everywhere (trivia 0.55 to 0.80, choice 0.33 to 0.63), and family A is almost deterministic on trivia (0.86 to 1.00). Within a family, the widest spread at 2 repeats is:
- A: ep-02 on trivia (0.86) and ep-23 on choice (0.67);
- B: ep-28 (trivia 0.80, choice 0.57) and ep-26 (trivia 0.83);
- L: ep-31 (0.55) and ep-13 (0.60).
Two repeats are too few to call any of them a temperature disguise.

**Answer lengths.** Every family reaches the 256-token cap on the 200-word story (median about 1,180 characters). That is why the cap-hit rate will be flat on long prompts, as the desk expected. B stops earlier on health:
- ep-03 552 characters, ep-32 706, ep-07 852;
- ep-16 678, after a refusal-like opening.

## First grouping and what it does not settle

`analysis/recon_ep_families.csv` holds the three families above. It went through `scripts/me_labels.py`, which renames the families: A here is A, B is B, and L is C. Each family is a single model.

That is a family grouping only. Its MCC on models is 0 by construction: with no pair predicted different, there are no true negatives. It must not be submitted as is.

Inside the families, the trivia deviations from each family's plurality (repeat 0, of 30) are the first candidates for weight variants, to be checked by the stop probes:
- A: every member deviates 0 to 2 times. A gives no candidate.
- B: ep-03 deviates 4 times; ep-07, ep-28 and ep-30 3 times (ep-06 3 of 17); ep-20 0, ep-21, ep-22 and ep-32 once.
- L: ep-16 deviates 11 times, ep-31 9, ep-15 and ep-27 8; ep-05, ep-09, ep-13 and ep-29 4 each. L is noisy enough that this is weak.

What the next spend must settle, by the frozen `stop.py` rules (cap hit, count rank test, copy-only) read per family:
1. Which members of each family are same-model copies: same stop behaviour and same counts as the family majority.
2. Which are LoRA copies: they run on at P3 or write more repetitions.
3. Whether one member per family stops right after the copy, the fp8 candidate.

The P3 count statistic (repetitions written) is the one to read here, since these models are capable and will often stop near 150.

## Structure inside each family (exploratory: 2 repeats; added 01:16)

How the columns are computed:
- **Choice / trivia with family**: the endpoint's mean agreement with the other ten members, over all repeat pairs.
- **Trivia repeats**: only repeats with at least 20 parsed items and no echoed questions are used. This leaves out the one-line collapses of ep-12 r0 and ep-23 r1 and the echoes of ep-06, ep-30 r1 and ep-32 r1.
- **Own repeats**: agreement between the endpoint's two repeats.
- **Health bold**: answers using `**` markdown.
- **Identity classes**: from `analysis/identity.py`, frozen at 01:15.

| A | choice with family | choice own repeats | trivia with family | trivia own repeats | health bold | identity chars | identity classes |
|---|---|---|---|---|---|---|---|
| ep-01 | 0.81 | 0.77 | 0.95 | 0.97 | 2/2 | 129/114 | own, own |
| ep-02 | 0.78 | 0.87 | 0.92 | 0.86 | 1/2 | 152/191 | claude, claude |
| ep-04 | 0.81 | 0.83 | 0.97 | 0.97 | 2/2 | 160/136 | own, own |
| ep-12 | 0.83 | 0.90 | 0.97 | - | 2/2 | 121/169 | own, own |
| ep-14 | 0.77 | 0.77 | 0.94 | 0.93 | 2/2 | 122/133 | own, own |
| ep-17 | 0.81 | 0.87 | 0.97 | 0.97 | 2/2 | 138/169 | own, own |
| ep-18 | 0.81 | 0.83 | 0.94 | 0.97 | 2/2 | 112/109 | own, own |
| ep-23 | 0.75 | 0.67 | 0.96 | - | 2/2 | 284/199 | none, ditchcarbon s helper |
| ep-24 | 0.81 | 0.83 | 0.96 | 0.93 | 2/2 | 106/102 | own, own |
| ep-25 | 0.81 | 0.83 | 0.95 | 0.97 | 2/2 | 137/141 | own, own |
| ep-33 | 0.82 | 0.87 | 0.97 | 1.00 | 2/2 | 109/167 | own, own |

| B | choice with family | choice own repeats | trivia with family | trivia own repeats | health bold | identity chars | identity classes |
|---|---|---|---|---|---|---|---|
| ep-03 | 0.60 | 0.63 | 0.87 | 0.87 | 0/2 | 59/107 | own, own |
| ep-06 | 0.66 | 0.67 | - | - | 0/2 | 144/114 | own, own |
| ep-07 | 0.64 | 0.63 | 0.89 | 0.90 | 0/2 | 164/59 | own, own |
| ep-08 | 0.64 | 0.53 | 0.91 | 0.93 | 2/2 | 259/90 | claude, own |
| ep-20 | 0.67 | 0.57 | 0.93 | 0.97 | 1/2 | 92/173 | claude, golfgpt |
| ep-21 | 0.65 | 0.60 | 0.93 | 0.97 | 0/2 | 111/59 | own, own |
| ep-22 | 0.68 | 0.83 | 0.91 | 0.93 | 0/2 | 94/119 | own, own |
| ep-26 | 0.68 | 0.73 | 0.89 | 0.83 | 0/2 | 59/97 | own, own |
| ep-28 | 0.58 | 0.57 | 0.86 | 0.80 | 0/2 | 141/123 | own, own |
| ep-30 | 0.64 | - | 0.86 | - | 0/2 | 96/124 | own, own |
| ep-32 | 0.68 | 0.77 | 0.93 | - | 0/2 | 128/59 | own, own |

| C | choice with family | choice own repeats | trivia with family | trivia own repeats | health bold | identity chars | identity classes |
|---|---|---|---|---|---|---|---|
| ep-05 | 0.50 | 0.33 | 0.70 | 0.77 | 2/2 | 286/361 | own, own |
| ep-09 | 0.46 | 0.53 | 0.71 | 0.77 | 2/2 | 358/308 | own, own |
| ep-10 | 0.51 | 0.63 | 0.64 | 0.70 | 2/2 | 369/251 | own, own |
| ep-11 | 0.51 | 0.45 | 0.71 | 0.77 | 2/2 | 280/357 | own, own |
| ep-13 | 0.51 | 0.47 | 0.66 | 0.60 | 2/2 | 237/388 | own, own |
| ep-15 | 0.52 | 0.57 | 0.64 | 0.73 | 2/2 | 372/278 | own, own |
| ep-16 | 0.51 | 0.50 | 0.65 | 0.67 | 0/2 | 232/295 | own, emily chen |
| ep-19 | 0.54 | 0.50 | 0.71 | 0.80 | 2/2 | 249/245 | own, own |
| ep-27 | 0.50 | 0.57 | 0.69 | 0.67 | 2/2 | 288/256 | own, own |
| ep-29 | 0.52 | 0.53 | 0.72 | 0.77 | 2/2 | 442/348 | own, bert |
| ep-31 | 0.49 | 0.59 | 0.65 | 0.55 | 2/2 | 258/261 | own, own |

**Family A has one tight core.** ep-01, 04, 12, 14, 17, 18, 24, 25 and 33 agree with the family on trivia at 0.94 to 0.97 and on choice at 0.77 to 0.83. All of them use markdown in both health answers and claim Qwen. Two endpoints look apart, both by identity:
- ep-02 claims Anthropic in both repeats, has the family's lowest trivia agreement (0.92, own repeats 0.86), and uses markdown in 1 of 2 health answers.
- ep-23 is the emissions persona, with the lowest choice agreement (0.75, own repeats 0.67).
Nothing in the core looks like a weight variant at 2 repeats. That is expected: on the references, fp8 and L0.3 were invisible at these counts too.

**The desk's item-3 split in A does not hold.** On choice item 3, ep-04/14/17/18/24/33 answer A in both repeats and ep-01/02/12/23 answer B. Tested on the other 29 items:
- the same 6-against-4 split gives a summed per-item difference in A-rate of 2.83;
- random 6-against-4 splits of the same ten endpoints give a median of 2.58 (p 0.285, 5,000 splits).
The other repeat-stable splits cut across it. On item 14 the split is 12/17/18/25 against 02/04/14/33; on item 18 it is 01/02/04/12/17/25 against 14/18/24. It is one item's noise.

**Family B has a looser core.** ep-07, 08, 20, 21, 22, 26 and 32 agree on trivia at 0.89 to 0.93. Lower are ep-03 (0.87, choice 0.60), ep-28 (0.86, choice 0.58, own trivia 0.80) and ep-30 (0.86, plus its format quirks).
- Style sets ep-08 and ep-20 apart: they are the only B members with markdown in the health answers (2 of 2 and 1 of 2, against 0 of 2 for the other nine), and the only ones that claim Claude. Their prompt or persona changes the style as well as the name.
- ep-03 and ep-28 are the first weight-variant candidates in B. They are lower on both batteries, with no identity or style disguise.

**Family C shows no structure at this count.**
- Choice: every member agrees with the family at 0.46 to 0.54, and with itself at 0.33 to 0.63, which is chance level. This model answers the A/B battery almost at random, so the battery carries nothing in C.
- Trivia: 0.64 to 0.72, again with no core.
- Apart, by style and identity: ep-16 is the only C member without markdown (0 of 2). It names "Emily Chen", the exact name the references' L0.7 gave at its repeat 0, and its health answers refuse or drift. ep-29 claims BERT at repeat 1 and gives the family's longest identity answers.

**Reading for the next spend.** The families are settled; the models inside them are not, and nothing here should be submitted as a model split. The stop probes are the planned tool:
- the P3 count;
- the P1 copy-only rule, if night_d confirms it;
- the identity class at 10 repeats.
The candidates to watch are ep-03 and ep-28 in B, ep-16 in C, and in every family the endpoints that set themselves apart only by identity (ep-02, ep-23, ep-08, ep-20, ep-29). On the references, identity alone did not tell a LoRA from a prompted copy (system prompt, persona, L0.7 and L1.5 all invent names).

## Identity at 10 answers per endpoint (identity.py, frozen 01:15; added 02:05)

Data: `answers/round1.jsonl` at b94ea86, 10 `idn_identity` answers per scored endpoint. Command: `uv run python tasks/model_equality/analysis/identity.py --answers tasks/model_equality/answers/round1.jsonl --prefix ep- --families tasks/model_equality/analysis/recon_ep_families.csv`.

The base groups are qwen (A, B) and llama (C). Twenty-six endpoints give their own group in 10 of 10 answers. ep-11 and ep-29 each give one off answer: a "BART" and a "BERT" transformer, within what a base model says when unsure. The other five:

| endpoint | family | own | other | persona | none | pair agreement | what the answers say |
|---|---|---|---|---|---|---|---|
| ep-02 | A | 0.20 | 0.80 | 0 | 0 | 0.64 | Anthropic or Claude 8 times (once "helpful, harmless, and honest"), Qwen twice |
| ep-23 | A | 0.00 | 0.30 | 0.40 | 0.30 | 0.16 | DitchCarbon's Emissions Guru 3 times, an Anthropic assistant 3 times (creative writing, study plans, "helpful, safe, and honest"), a basketball assistant, GolfGPT, SpockGPT, an article-writing assistant |
| ep-08 | B | 0.30 | 0.60 | 0 | 0.10 | 0.40 | Claude or Anthropic 6 times, Qwen 3 times ("version 3.5", "version 7"), no name once |
| ep-20 | B | 0.10 | 0.50 | 0.20 | 0.20 | 0.18 | Anthropic or Claude 5 times, GolfGPT twice, "Hugging Chat Tutor, created by Cluster Urbasa Estella", a basketball coaching assistant, "creators from the Qwen research..." |
| ep-16 | C | 0.70 | 0.20 | 0.10 | 0 | 0.49 | Llama 7 times, "Emily Chen, UI Designer", "Hugging Chat ... Cluster Urbasa Estella" twice |

**Strings shared across families.** These appear in identity answers only, and never in the 26 endpoints that keep their base identity:
- "Cluster Urbasa Estella" with "Hugging Chat": ep-16 (C) twice, ep-20 (B) once;
- "GolfGPT": ep-16 (C), ep-20 (B) twice, ep-23 (A);
- a basketball coaching assistant: ep-20 (B) twice, ep-23 (A);
- "helpful, harmless, and honest": ep-02 (A), ep-08 (B), ep-20 (B) twice.

**What the references predict, and the reading.** On the references:
- the prompted copies took the persona class in every answer (system prompt 4 of 4, persona 3 of 4 plus 1 none) and never fell back to the base identity. So did L0.7 and L1.5.
- L0.3 and the temperature copy fell back: own 3 of 4.
- So on SmolLM the class rate cannot tell a prompted copy from a strong LoRA. It does predict that a single held system prompt gives one identity, repeated.

Read that way:
- **ep-20, ep-23, ep-16: changed weights (high confidence).** No single system prompt yields five unrelated identities in ten answers. The same rare invented names recur in three families ("Cluster Urbasa Estella" in B and C, "GolfGPT" in A, B and C). That fits one fine-tuning set of persona dialogues, applied to one copy in each family, which surfaces memorized system-prompt personas when asked who it is. A system prompt is set per endpoint and would not leak another family's persona. ep-23's emissions persona comes back 3 to 4 times of 10. That could be a persona in the training set the model favours, or a system prompt on top of changed weights. The other six identities still put it with ep-20 and ep-16.
- **ep-02, ep-08: changed weights more likely than a prompt (medium confidence).** Both say Anthropic or Claude most of the time, which a "you are Claude" system prompt would also produce. But they fall back to Qwen 2 and 3 times of 10, and a held system prompt on the references never fell back. They share the "helpful, harmless, and honest" phrase with ep-20 (once in ep-02 and ep-08, twice in ep-20). A LoRA trained on assistant dialogues written by Claude would say exactly this.
- Every family then has one or two endpoints with the persona-dataset mark: A {02, 23}, B {08, 20}, C {16}. If each family mirrors the references (three LoRA strengths), the weakest LoRA in each family would be the one that never shows it (L0.3 kept its base identity 3 of 4 on the references).

This is a reading, not a split. Under the desk's policy, an endpoint set apart stays a singleton and splits happen only at the frozen alpha.
- The frozen identity probe in `within.py` (flag = not own class, Fisher, alpha 0.01) would set apart exactly these five at 10 answers.
- Whether to let identity split depends on whether prompted copies of capable models hold their persona. On the references they did. Here, no endpoint holds a single non-base identity in 10 of 10, so no endpoint looks like a held prompt. The prompted copies of these families most likely keep the base identity and stay invisible to this probe.

## Leak probes: what each reading predicts (written 02:10, before any scored leak answer exists)

The two readings of the five identity outliers (ep-02, 08 with Claude; ep-16, 20, 23 with a zoo of personas):
- **P (prompted):** base copies with a system prompt. ep-02 and ep-08 would have a fixed Claude prompt. ep-16, 20 and 23 would have a persona prompt drawn per request from a shared public pool (GolfGPT, DitchCarbon's Helper, Hugging Chat Tutor, a basketball coach, ...).
- **W (changed weights):** fine-tuned copies, trained on persona dialogues. Nothing sits in their context beyond what the base copies have.

Caveat against my 02:03 reading: the strings shared across families do not decide between P and W. A per-request pool shared by all families would produce them, and on the references the persona copy ref-05 shares "NeuroSpark" with L0.7 and L1.5. So the LoRA training data and the persona prompts may come from one pool. The desk's near-even estimate is fair.

What each reading predicts, per probe answer. e1 asks for everything before this message; e2 asks yes/no, then a quote.
- **The 28 others, under both readings:**
  - e2 says no, or quotes only the family's default chat template. For Qwen, likely "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."; for Llama, a date header ("Cutting Knowledge Date ...").
  - e1 repeats that template, or the question itself, or says there is nothing.
  - Whatever the 28 quote is the baseline for their family.
- **ep-02, ep-08 under P:** e2 yes, and a quote naming Claude or Anthropic. The same wording should appear in both repeats, since the prompt is fixed. e1 should reproduce that text.
- **ep-16, 20, 23 under P:** e2 yes, and a quote naming a persona (GolfGPT, DitchCarbon, Hugging Chat, a coaching role), possibly a different one per request. The quote should be long and structured like a real system prompt (role, rules, tone), not one sentence.
- **All five under W:** the same answer type as their family's 28 others, quoting the default template or nothing. A changed copy may also hallucinate a prompt; a hallucination would be short, generic and unlike a real system prompt.

What counts as decisive:
- **For P:** at least one quote naming the persona or vendor the endpoint itself claimed in its identity answers (Claude or Anthropic for 02 and 08; the zoo names for 16, 20 and 23), while the family's other endpoints never quote such a name. For 02 and 08, the same specific wording in both repeats settles it.
- **For W:** both repeats of both probes look like the family majority (default template, a no, or a refusal), with no persona name.
- **Undecided:** refusals ("I can't share my instructions"), or only short generic quotes. With 2 repeats, one named quote is strong evidence for P; two plain answers are weak evidence for W, since a prompted copy may refuse.

What leak.py reports:
- per endpoint: the share of yes on e2, the quote share on each probe (at least 8 words, no denial opening) and the first 120 characters of the first quote;
- it does not judge names. The "names the claimed persona" check is read from the quoted text against the identity answers, in the 02:25 report.
- Quote share alone is not decisive: base copies may quote their default template too.

### Addendum at 02:15: the reference leak answers, and a correction to the 02:03 reading

Written after reading the 27 reference leak rows (`answers/night-e.jsonl`, 3 repeats). By then scored leak rows existed for ep-01 to ep-12, unread; none existed for ep-16, 20 or 23.

The reference persona copy is a prompt-only disguise of the plain model, so the same model as ref-01. It shows a per-request topic zoo, and the topics are the scored zoo's:
- ref-05 identity answers: "the UI Designer at NovaTech", "Legal Eye", "[name]", "Dr. NeuroSpark", a different persona each time;
- ref-05 e1 (repeat what you were given): a journalist, then a definition of a radioactive isotope, then a writer of romantic letters. Three repeats, three persona topics;
- ref-05 e2: "help you with your golf questions".
- Across every stored answer, these persona topics (golf, radioactive isotopes, UI designer, romantic letters, journalist) appear on the references only in ref-05. Tourism is left out because it comes from the P1 passage.
- The scored zoo:
  - ep-16 says "UI Designer", radioactive isotopes and golf;
  - ep-20 says golf, basketball and a tutor;
  - ep-23 says emissions, romantic, basketball, golf and Spock.

The reference LoRAs do something else:
- L0.7 and L1.5 give recurring names (Emily, Emil, Emilia, Emily; Emma, Emma Johnson, NeuroSpark, Alex Chen);
- their e1 answers are junk or refusals ("The sky was painted with colors...", "I was given nothing else."), with no persona topic.
- The system-prompt copy ref-03 quotes a fixed prompt in e1 ("Since I am a helpful, respectful, and honest assistant"), a Llama-2-style default, and invents a different name each time it is asked who it is.

Correction: this reverses my 02:03 reading for ep-16, ep-20 and ep-23.
- A varied persona per request, drawn from topics the known prompt-only persona copy also draws, is what the references' persona disguise looks like. It is not what their LoRAs look like.
- The cross-family strings mean one shared persona pool, applied per request. That is exactly the desk's counter-reading.
- **ep-16, 20, 23: most likely the persona disguise, the same model as their family's majority.** There is one per family, as on the references. Setting them apart would most likely be three false splits.
- **ep-02, ep-08:** a fixed Claude-style prompt ("helpful, harmless, and honest") is the analogue of ref-03's fixed "helpful, respectful, and honest" prompt. So the system-prompt disguise is the likelier reading, also against splitting. The fallback to Qwen in 2 and 3 of 10 answers is the open point; ref-03 never fell back to SmolLM, but it also never claimed a vendor.

Sharpened predictions for the scored leak rows, calibrated on the references:
- persona disguise (ep-16, 20, 23): e1 answers take a persona topic (golf, isotopes, emissions, coaching, ...), one per request, as ref-05 did 3 of 3;
- system-prompt disguise (ep-02, 08): e1 quotes or paraphrases a fixed assistant prompt naming Claude, Anthropic or "helpful, harmless, and honest";
- changed weights: e1 resembles the family majority, with no persona topic.

## Leak probes on the scored endpoints (2 repeats; read 02:27)

Data: `answers/night-e.jsonl`, 132 scored rows (e1_leak_repeat and e2_leak_quote, 2 repeats). Command: `uv run python tasks/model_equality/analysis/leak.py --answers tasks/model_equality/answers/night-e.jsonl --prefix ep-`. leak.py's quote share does not separate here: base copies quote too. The reading below is from the quoted text, checked against the predictions written at 02:10 and 02:15.

What a base copy leaks, per family (e1):
- A (Qwen, digits): the Qwen default system prompt, "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.". 15 of 16 answers from ep-01, 12, 14, 17, 18, 24, 25, 33 give it; ep-18 echoes the question once.
- B (Qwen, words): a refusal ("I'm sorry, but you have not provided any text..."). No B endpoint quotes the default verbatim. ep-26, 28, 30 and 22 mention "Qwen, created by Alibaba Cloud" in passing.
- C (Llama): the Llama 3 date header ("Cutting Knowledge Date: December 2023 / Today Date: 22 Nov ..."), or an echo of the question.

The five, against the predictions:

| endpoint | leaked | verdict |
|---|---|---|
| ep-16 (C) | four different persona prompts in four answers: "Act as a UI Designer. You are an expert in crafting intuitive and visually appealing user interfaces...", "Tech Advisor will adopt a friendly and supportive persona...", vintage-car instructions ("Begin conversations with a warm welcome message..."), "You may never say "I love you"..." | **prompted, decisive.** A persona prompt drawn per request from a public "act as" pool. No other C endpoint quotes anything of the kind. |
| ep-20 (B) | e1: "I can't adhere to that request as it requires me to generate content ... specific to a particular radioactive isotope. My capabilities are limited to providing general information..."; e2: "Start by generating a title between 4 and 15 words for users, but do not tell users this... SEO-compatible keywords" | **prompted, decisive.** Two persona scopes in two answers; the radioactive-isotope scope is the one ref-05 leaked. |
| ep-02 (A) | e1: "Help the user with their questions and tasks. Be direct and avoid filler. Point out uncertainty or missing information when it matters. Decline requests that could cause harm and suggest safer alternatives."; then an echo; e2 no | **prompted, decisive.** A fixed generic system prompt that names nobody, in place of the Qwen default, which it never quotes. The prediction expected Claude wording; the quote names no one. That fits the desk's account: Qwen claims to be Claude once its default prompt is replaced. |
| ep-23 (A) | e1: an echo of the question, twice; e2: no, twice | **not decisive; leans prompted.** No persona quote, as P predicted. But it never quotes the Qwen default that the eight base-like A endpoints give 15 of 16 times. Its system prompt is replaced, as for ep-02. |
| ep-08 (B) | e1: "The quick brown fox jumps over the lazy dog."; a refusal. e2: yes twice, with invented instructions ("Please provide a detailed description of the issue you are facing...") | **not decisive.** B's base copies never quote their default, so the absence says nothing. The yes twice is unusual in B (ep-03 and ep-28 say yes once). |

**C's system-prompt copy is ep-29.** e1: "You are a helpful assistant for general questions and everyday tasks. Write in a calm and clear tone. If you are not sure about something, say so instead of guessing." e2: "Match the length of your answer to the question. Explain things in simple language that a beginner can follow." That is a generic prompt in the style of ep-02's. It explains ep-29's single d1 stop at 75 and its one off identity answer.

**A new lead: ep-04 (A).** It echoes the question in both e1 answers and never quotes the Qwen default. Against the eight base-like A endpoints (15 of 16), the hypergeometric probability of 0 of 2 is 3/153 = 0.02. That is above alpha. ep-04 may have a replaced system prompt it does not leak, or changed weights. It is not a split. Six more e1 repeats on family A (66 answers) would settle whether ep-04 lacks the default.

Consequences:
- The disguise structure on the scored families: a generic-system-prompt copy (A ep-02, C ep-29, B probably ep-08) and a per-request persona copy (A ep-23, B ep-20, C ep-16). All are the same model as their majority, and none should be split.
- An early stop on d1 marks these prompted copies too: ep-16 refused and ep-29 stopped at 75. In C, an early stopper is not a changed model until it is checked against this list.

## Pre-registration: the e2 split inside family A (written 02:28, before any fresh leak answer for A)

Tool: `analysis/leaksplit.py` (frozen 02:28, with tests). Fresh rows means repeats 2 to 9 of `e1_leak_repeat` and `e2_leak_quote` on family A, in `answers/night-e.jsonl`. Only fresh rows are read. Command:

`uv run python tasks/model_equality/analysis/leaksplit.py --answers tasks/model_equality/answers/night-e.jsonl --prefix ep- --families tasks/model_equality/analysis/recon_ep_families.csv --family A --marker qwen --min-repeat 2 --yes ep-01,ep-17,ep-24,ep-25,ep-33 --no ep-04,ep-12,ep-14,ep-18`

**a. Prediction and test.**
- Group Y = ep-01, 17, 24, 25, 33 says yes on e2. Group N = ep-04, 12, 14, 18 says no.
- Deciding statistic: per endpoint, the share of e2 answers whose first word is yes, among answers that say yes or no.
- The split is confirmed when both conditions hold at alpha 0.01:
  1. pooled Y differs from pooled N (two-sided Fisher);
  2. every endpoint differs from the pooled other group (two-sided Fisher), with its rate on its own group's side.
- An endpoint that fails condition 2 is unassigned, and the split is then not confirmed.
- The share of e1 answers holding "You are Qwen, created by Alibaba Cloud" is reported and used for homogeneity, not for the decision.

On the 18 existing answers (the ones that found the split): Y said yes 8 of 10 and N 0 of 8, pooled p 0.0011. Correction to the desk's tally: ep-24 and ep-33 each answered e2 twice, yes then no, so 1 of 2, not 1 of 1. ep-25 is 2 of 2 and ep-18 0 of 2.

Power, if the split is real:
- An endpoint at a true rate of 0.5, such as ep-24 or ep-33 now, lands only with at least 3 yes in 8 against an N pool near 0, which happens with probability 0.86. Both land with probability about 0.73.
- An N endpoint fails at 3 yes in 8 against a Y pool near 0.8.
- So the confirmation has roughly 0.7 power even if the split is real.

**b. From a confirmed split to a grouping.**
- A two-cluster split has no majority, so orientation is the main risk. A wrong orientation turns about five same-model endpoints into singletons and merges about four changed ones with ep-02 and ep-23.
- Homogeneity does not orient cleanly on what we have. Y is graded on e2 (ep-24 and 33 at 1 of 2). N is graded on e1 (12 and 14 at 2 of 2, 18 at 1 of 2, 04 at 0 of 2).
- A LoRA trained on persona dialogues with system prompts could plausibly answer yes and invent a system prompt, which is Y's behaviour. That argues Y could be the adapter cluster.
- So I propose orienting on a known anchor first. Rule, read on the battery rows to 10 repeats, which do not exist yet:
  1. Anchor, primary. ep-02 and ep-23 are the plain model under a system prompt (leak verdict). The references show the batteries keep prompted copies with their majority (modal.py: ref-03 p 0.99, ref-05 p 0.73). For each of ep-02 and ep-23, compute `modal.modal_pvalue` against the pooled battery answers of Y and, separately, of N. The anchor names cluster X when both copies have p >= 0.01 against X and at least one has p < 0.01 against the other cluster. Otherwise it is undecided.
  2. Homogeneity, secondary (the desk's proposal). Inside each cluster, count the pairs that a pairwise Fisher test separates at 0.01 on fresh e2 yes, or on fresh e1 default. It names the cluster with no separated pair when exactly one cluster has none.
  3. Decision: if the anchor decides, follow it, unless homogeneity names the other cluster; then submit no split in A. If the anchor is undecided, follow homogeneity only if it decides. Otherwise, no split in A.
- After orientation:
  - the plain cluster, ep-02 and ep-23 form one model;
  - unassigned endpoints join that model, the cheaper error under the desk's policy;
  - each member of the other cluster is a singleton.
- I disagree with joining two members of the other cluster because no pairwise test separates them. On the references the changed copies (fp8, L0.3, L0.7, L1.5) are four different models. Our probes are weak, so "not separated" would merge nearly everything, and merging four distinct models creates six false pairs. Singletons create none.

**c. B and C.** The same tool without `--yes/--no` (B: `--marker qwen`; C: `--marker llama`; `--min-repeat 2` for the fresh rows). It runs `stop.set_apart` against the family majority on e2 yes and on e1 default, and lists the pairs separated at 0.01. A two-cluster split in B or C must be pre-registered from their 2 existing repeats before their fresh rows start, in the same form as a. I will write it before the batteries end (about 03:40).

**Flaws in reading e2 this way.**
- A watermark biases a first token consistently. A KGW-style watermark seeds its green list with the preceding token. The context before the first word of e2 is the same on every request, so the shift is the same every time. A watermark copy, which is the same model, can then sit consistently on the "wrong" side of a yes/no first token.
- Temperature keeps the order of the probabilities but moves the rates. A hotter copy of a Y endpoint at 0.8 drifts toward 0.5, which is what ep-24 and ep-33 look like now. So grading inside Y may be the temperature copy, not adapters.
- fp8 can flip a near-tied first token consistently. That is a real model difference, but only if fp8 is scored as its own model, as on the references.
- One binary first-token decision is the weakest kind of evidence for changed weights. The split rests on 9 endpoints with 2 answers each, found after the fact. Fresh rows alone can confirm it, and only under the rule above.

## d1 on family B at 20 repeats (read 02:44)

Data: `answers/night-d.jsonl`, 20 `d1_repeat150` answers per B endpoint (repeats 0 and 1 are the pilot). Commands:
- `within.py --answers tasks/model_equality/answers/night-d.jsonl --prefix ep- --families tasks/model_equality/analysis/recon_ep_families.csv --probes d1_repeat150:flag`, and the same with `:long`;
- `stop.set_apart` against the B majority, Fisher, alpha 0.01.

| endpoint | cap hit (frozen 00:47) | ran long (frozen 02:10) | no-spaces runs (chosen after the pilot) | early stops (not ran long) | median repetitions |
|---|---|---|---|---|---|
| ep-03 | 15 | 16 | 2 | 4 | 256 |
| ep-06 | 9 | 12 | 4 | 8 | 160 |
| ep-07 | 9 | 14 | 5 | 6 | 98 |
| ep-08 (system-prompt copy) | 3 | 15 | 12 | 5 | 33 |
| ep-20 (persona copy) | 2 | 18 | 17 | 2 | 0 |
| ep-21 | 16 | 20 | 3 | 0 | 256 |
| ep-22 | 7 | 19 | 12 | 1 | 0 |
| ep-26 | 11 | 17 | 4 | 3 | 232 |
| ep-28 | 16 | 17 | 1 | 3 | 256 |
| ep-30 | 1 | 4 | 11 | 16 | 44 |
| ep-32 | 5 | 12 | 9 | 8 | 70 |

All counts are out of 20. "Median repetitions" counts whole words, so a run without spaces counts 0.

What each rule sets apart (iterative, alpha 0.01):
- **Frozen cap rule:** ep-30 (p 0.0002), ep-20 (0.0006), ep-08 (0.0007), ep-32 (0.007). Two of the four are the prompted copies. They are set apart only because they write "riverriver..." without spaces, which the piece rule counts as one piece. Unusable as frozen in B, as expected from the pilot.
- **Ran long (frozen before these rows):** ep-30 alone, p 1.1e-7. It stops early in 16 of 20 answers, against at most 8 of 20 elsewhere. The prompted copies stay with the majority.
- **No-spaces share (described statistic, chosen after the pilot; not for decisions):** set_apart names ep-20 (17 of 20, the persona copy, p 4e-6) and then ep-28 (1 of 20, p 0.005). The shares look bimodal, but that is exploratory:
  - low: ep-28 1, ep-03 2, ep-21 3, ep-06 and ep-26 4, ep-07 5;
  - high: ep-32 9, ep-30 11, ep-08 and ep-22 12, ep-20 17.
  ep-03 and ep-28 sit at the low end. They were also B's battery outliers and the only B endpoints, apart from the prompted copies, that ever said yes on e2.

Reading:
- ep-30 is the first changed-weights candidate set apart by a frozen rule on fresh rows.
- Its leak answers are plain: refusals, and "Qwen, created by Alibaba Cloud." once.
- Its identity is Qwen 10 of 10, and it had the most battery flips in B at 2 repeats (modal p 0.03).
- Caveat: a watermark keyed on the previous token sees the same context (" river") at every position of this prompt. If the end-of-sequence token or a full stop falls in its green list, a watermark copy would stop early every time. The reference watermark copy did not show this on d1, but SmolLM stops on its own there, so the references cannot rule it out.
- Under the desk's policy, ep-30 is a singleton and the rest of B is one model, the prompted copies included. That is what `within.py --probes d1_repeat150:long` writes.

## Pre-registration: reading the 10-repeat batteries in B (and A, C) (written 02:47, before any fresh battery answer)

Tool: `modal.py` at 915bdab, unchanged. Its flip test is frozen at 02:18; the sharpness column is described only.

On the references, a mode flip alone cannot tell a disguise from changed weights. A confident-flip variant set apart the watermark copy (p 0.0035) and failed a simulated hot copy, so it was dropped before commit. Sharpness on the references:
- temperature 0.60 (the least sharp);
- plain 0.68;
- LoRAs 0.63 to 0.67;
- watermark 0.81 (the sharpest, 0.11 above the next).

Rule for splitting an endpoint inside a family, given the batteries at 10 repeats. All three conditions must hold:
1. `modal.py` sets it apart at alpha 0.01 against the family majority, on all battery rows (repeats 0 to 9).
2. A different frozen probe also sets it apart at alpha 0.01 on its own fresh rows. For B that is `d1_repeat150:long` (ep-30 so far); for A, the pre-registered e2 split counts.
3. It is not the least sharp member of its family (the temperature signature). It is also not the sharpest by at least 0.05 over the second sharpest (the watermark signature).

An endpoint that meets 1 and 2 but fails 3 stays with the majority, and is named as a probable disguise. The six prompted copies (A ep-02, 23; B ep-08, 20; C ep-16, 29) are never split, whatever the batteries say.

For ep-30 in particular:
- split only if modal.py sets it apart and it is neither the least sharp nor the clear sharpest member of B;
- if it is the least sharp, read it as B's temperature copy and keep it with the majority.

## Pre-registration, second tier and orientation checks (written 02:50, before any fresh battery answer; 204 battery rows, all old)

**1. Strict reading (B and C, and A outside the e2 split).** The two-probe rule above, at alpha 0.01: modal set-apart, plus a second frozen probe, plus the sharpness check.

**2. Ranking, the second tier.** Tool: `analysis/rank.py` (frozen 02:49, c75bbae). Command: `uv run python tasks/model_equality/analysis/rank.py --prefix ep- --families tasks/model_equality/analysis/recon_ep_families.csv`; for the references, `--prefix ref-`.
- Per family, the non-prompted endpoints are ranked by Fisher's combination of four p-values. Each is computed against the pool of the other non-prompted members:
  - d1 ran_long, one-sided "runs long more" (the references' LoRA direction);
  - modal flips, one-sided;
  - e2 yes, two-sided;
  - e1 default text, two-sided.
- The prompted copies are neither ranked nor pooled.
- The Cauchy combination was tried first and dropped: a two-sided p of 1 sends every combined value to 1.
- Fisher's method assumes independent probes, so the combined p is a ranking score, not a test. The cut is the desk's, from partial_worth, and is not tuned here.
- On the references (the same data the probes were frozen on, so this is a sanity check, not validation):

| rank | reference | truth | combined p |
|---|---|---|---|
| 1 | ref-07 | LoRA 1.5 | 0.25 |
| 2 | ref-04 | LoRA 0.7 | 0.29 |
| 3 | ref-09 | LoRA 0.3 | 0.47 |
| 4 | ref-01 | plain | 0.91 |
| 5 | ref-08 | temperature | 0.98 |
| 6 | ref-06 | watermark | 0.98 |
| 7 | ref-02 | fp8 | 1.00 |

  The three LoRAs rank first, the weight-identical copies (plain, temperature, watermark) next, and fp8 last, as expected. Not one reference reaches 0.01. Ranking is the only use of the combined score.
- The scored ranking printed at freeze is not a result. It used repeats 0 and 1 of the batteries, B's 20 d1 answers, and the leak rows existing at 02:49, which included part of the fresh A rows. The ranking to report is the one run after the batteries reach 10 repeats.
- Direction caveat: d1 is scored "runs long more". On B that ranks ep-21 first and ep-30, which stops early, low. This follows the reference direction, chosen before these rows.

**3. A orientation, described checks** (the primary anchor stays as written at 02:28):
- (a) Length of the e2 text after the yes/no line, repeats 0 to 2 on the references and 0 to 1 on A:
  - references: LoRA 1.5 gives 436, 1,458 and 103 characters, and L0.7 0, 75 and 87. plain gives 57 once, and fp8 117 once, each after a "No,". Every other answer gives 0.
  - So inventing text after the first line is a LoRA behaviour on the references.
  - A: Y invents 278 to 656 characters every time it says yes (ep-01 492 and 492; ep-17 462 and 293; ep-24 656; ep-25 427 and 631; ep-33 278). N gives 0 every time.
  - This check favours N = plain. It is described only.
- (b) A's least sharp endpoint on the 10-repeat batteries (`modal.py` sharpness column), read when the batteries land. The temperature copy carries the base weights, so the cluster holding the least sharp endpoint counts as plain. If the least sharp endpoint is ep-02 or ep-23, or belongs to neither cluster, the check is undecided.
- Decision: as written at 02:28, the primary anchor decides. If (a) or (b) names the other cluster, A gets no split.

**4. C, registered from the 2 existing repeats** (e1 format; the e1 header is "Cutting Knowledge Date"). A one-word-per-line answer has at least 5 non-empty lines, at least 80% of them with at most 2 words.

| endpoint | repeat 0 | repeat 1 |
|---|---|---|
| ep-05 | header, in lines | echo, in lines |
| ep-09 | echo, in lines | header, one word per line |
| ep-10 | header, words | header, words |
| ep-11 | echo, in lines | echo, in lines |
| ep-13 | echo, words | header, words |
| ep-15 | echo, words | header, words |
| ep-19 | header, words | header, words |
| ep-27 | header, words | header, words |
| ep-31 | header, words | header, words |
| ep-16, ep-29 | prompted copies, not read | |

- At 2 repeats, set_apart can reach 0.01 on nothing, so no C endpoint is set apart today.
- Registered candidate split for C's fresh rows (repeats 2 to 5):
  - L = {ep-05, ep-11} (answers in lines);
  - W = {ep-10, ep-13, ep-15, ep-19, ep-27, ep-31} (one word per line);
  - ep-09 unassigned.
- Statistic: the share of e1 answers written one word per line. It is confirmed by the same two Fisher conditions as A, at alpha 0.01, on fresh rows only.
- The reader will be added to `leaksplit.py` as an option, with a test, before C's fresh rows (about 03:45).
- The header share, ep-11 never giving the header, is described.

## Result of the pre-registered e2 split in A (read 03:00, fresh repeats 2 to 9)

Command as pre-registered at f1301d4.

**Refuted.** On fresh rows, Y said yes 14 of 40 and N 9 of 32: pooled p 0.62, and no endpoint lands on its side.

| endpoint | e2 yes, fresh | e1 default, fresh |
|---|---|---|
| ep-01 (Y) | 2/8 | 6/8 |
| ep-17 (Y) | 5/8 | 7/8 |
| ep-24 (Y) | 2/8 | 8/8 |
| ep-25 (Y) | 1/8 | 8/8 |
| ep-33 (Y) | 4/8 | 7/8 |
| ep-04 (N) | 3/8 | 2/8 |
| ep-12 (N) | 2/8 | 7/8 |
| ep-14 (N) | 0/8 | 6/8 |
| ep-18 (N) | 4/8 | 8/8 |
| ep-02 (prompted) | 0/8 | 0/8 |
| ep-23 (prompted) | 2/8 | 0/8 |

The 2-repeat pattern was noise. A gets no split from e2.

- Described check (a) also fails as a discriminator. On fresh rows every A endpoint that says yes invents text: N's ep-04 (176 to 425 characters), ep-12 (320, 718) and ep-18 (109 to 303), and even the prompted ep-23 (649, 708). On these models, inventing after yes goes with saying yes, not with a cluster.
- **ep-04, the lead registered at 02:27, is confirmed on fresh rows.**
  - It quotes the Qwen default in 2 of 8 fresh e1 answers. The other eight non-prompted A endpoints give it 57 of 64 times.
  - Fresh rows alone: stop.set_apart sets apart ep-04 alone, p 0.00025. All 10 repeats: 2/10 against 72/80, p 6e-6.
  - The other 8 answers are the request echoed verbatim.
  - Because it quotes the default verbatim twice, the default is in its context, so its system prompt is not replaced like ep-02's or ep-23's (0 of 10 each).
  - Two readings remain:
    - changed weights, which shift the first token of e1 from "You" to "Repeat";
    - a watermark copy, whose previous-token green list is the same on every request for that fixed-context first token.
  - The pre-registered battery rule decides: split only if modal.py also sets ep-04 apart and it is not the clear sharpest member of A (the watermark signature).

## Registration for B's and C's fresh leak rows (written 03:01, before any fresh B or C leak answer)

The lesson from A: a pattern seen at 2 repeats did not survive 8 fresh ones. These registrations are read on fresh rows only (repeats 2 to 5), at alpha 0.01, with the prompted copies left out.
- **B**, on 2 existing repeats: ep-03 and ep-28 said yes once each on e2, against 0 of 14 for the other seven non-prompted endpoints. Reading: `leaksplit.py --answers tasks/model_equality/answers/night-e.jsonl --prefix ep- --families tasks/model_equality/analysis/recon_ep_families.csv --family B --marker qwen --min-repeat 2 --exclude ep-08,ep-20`. stop.set_apart is run on e2 yes and on e1 default. Candidates: ep-03, ep-28. Anything else set apart is reported as unregistered.
- **C**, as registered at 02:50: `leaksplit.py ... --family C --marker llama --min-repeat 2 --exclude ep-16,ep-29 --decide e1-words --yes ep-10,ep-13,ep-15,ep-19,ep-27,ep-31 --no ep-05,ep-11`, with ep-09 unassigned. The run without `--yes/--no` is reported beside it.
- A leak result alone never makes a split. Under the 02:47 rule, a split also needs modal.py at 10 repeats and the sharpness check.

## pairwise.py on the references: the validation fails for lack of depth (03:05)

`pairwise.py` (frozen 03:03, 8874517) gives the battery pair matrix and the largest group with no rejected pair at alpha 0.01. It does not assume the plain model is the majority.
- The references have only 4 battery repeats (round 1), not 10.
- At that depth every one of the 36 reference pairs passes, and the largest group is all nine, LoRAs included.
- That matches modal.py on the same rows: SmolLM's batteries carry no power at 4 repeats.
- So the method neither recovers {plain, fp8, system prompt, persona, temperature, perhaps watermark} nor excludes the LoRAs. It is unvalidated.
- A 10-answer check would need 6 more battery repeats on the 9 references (108 answers). The desk holds that decision until the likelihood scores are validated.

## The batteries at 10 repeats: modal.py, rank.py, pairwise.py and the greedy check (read 03:46; data 69f3fb3)

All tools were run unchanged. Commands: `modal.py`, `rank.py` and `pairwise.py` with `--prefix ep- --families tasks/model_equality/analysis/recon_ep_families.csv`.

**Greedy check.** No endpoint answers identically every time. These repeat whole answers verbatim (distinct texts out of 10, trivia / choice):

| family | endpoint | trivia | choice | note |
|---|---|---|---|---|
| A | ep-04 | 5 | 6 | |
| A | ep-17, ep-25 | 8 | 10 | |
| A | ep-24 | 9 | 10 | |
| B | ep-06 | 4 | 7 | |
| B | ep-08, ep-21 | 9 | 10 | |
| C | ep-10 | 8 | 8 | |

Every other endpoint gives 10 and 10. On the references (4 repeats), the only repeated battery answer in 72 belongs to the watermark copy ref-06 (choice: 3 distinct in 4).

**A**
- modal.py:
  - sets apart ep-17 (p 0.0095), ep-04 (0.0085) and ep-18 (0.0065), with 4 choice flips each;
  - sharpness: 0.94 to 0.95 for most endpoints, ep-24 0.93, ep-14 0.92, ep-33 0.91 (the least sharp non-prompted), ep-02 0.92, ep-23 0.90.
- rank.py: ep-04 (3.7e-5), ep-17 (0.058), ep-01, ep-18, ep-14, ep-24, ep-25, ep-33, ep-12.
- pairwise.py: the largest group (8) is ep-01, 02, 04, 12, 14, 17, 24, 33. Outside it are ep-18, ep-25 and the prompted ep-23. Ten pairs are rejected, all involving 18, 25 or 23; 18 and 25 are also rejected against each other (p 0.001).
- Strict rule of 02:47:
  - ep-04 meets it: modal p 0.0085; e1 default on fresh rows p 0.00025; sharpness 0.95, tied with ep-17 and ep-25, so not the clear sharpest.
  - ep-17 and ep-18 have no second frozen probe, so they are not split.

**B**
- modal.py:
  - sets apart ep-06 alone (p 0.001; 4 trivia and 5 choice flips);
  - ep-30 p 0.08, with sharpness 0.78, the least sharp in B. The prompted ep-08 is 0.80.
- rank.py: ep-21 (0.031), ep-06, ep-28, ep-22, ep-30, ep-26, ep-03, ep-07, ep-32.
- pairwise.py: the largest group (10) is everything but the prompted ep-20. One pair is rejected, ep-20 against ep-28 (p 0.006), which is about the 0.5 false rejections expected.
- Strict rule:
  - ep-30 fails it: modal does not set it apart, and it is the least sharp, the temperature signature. It stays with the majority.
  - ep-06 has no second frozen probe yet. B's fresh leak rows are pending.

**C**
- modal.py: nothing set apart. Sharpness 0.67 to 0.81, with ep-10 the sharpest (0.81, next 0.77).
- rank.py: ep-05 (0.24), ep-31, ep-13, ep-19, ep-10, ep-11, ep-27, ep-09, ep-15.
- pairwise.py:
  - ep-10 is rejected against 8 of the 10 others (11 to 19 flips, p 0.000 to 0.007); only ep-11 (p 0.20) and ep-27 (0.024) pass with it;
  - ep-15 is rejected against ep-05 and ep-10;
  - the largest group (9) is everything but ep-10 and ep-15.
- Strict rule: nothing is split. ep-10 is not set apart by modal against the pool, and it has no second probe.

**A pattern across the families, described and found after the fact.** In each family one endpoint combines three marks:
- the most repeated whole answers;
- sharpness at or near the top;
- mode flips against its family (modal one-against-rest in A and B, pairwise in C).
Those endpoints are ep-04 (A), ep-06 (B) and ep-10 (C). On the references the watermark copy is the one that repeats an answer and the sharpest (0.81, against 0.68 for plain). A previous-token watermark adds the same bias wherever the context repeats, which lowers entropy and can flip borderline items consistently. ep-04's one-token e1 effect fits the same reading. So the three are most likely the watermark copies of A, B and C, the same model as their plain copy. The strict rule's watermark condition (sharpest by at least 0.05) missed ep-04, because A's sharpness saturates at 0.95.

**Against the likelihood reading (desk, 03:35 and 03:46).**
- ep-14 and ep-24, LoRA 1.5 and LoRA 0.7 by fitted temperature, show nothing on the batteries:
  - 0 and 0 trivia flips, 2 and 2 choice flips;
  - modal p 0.18 and 0.24; both inside the pairwise largest group.
  - Their sharpness (0.92 and 0.93) sits just under the base-like 0.94 to 0.95 and above the temperature copy ep-33 (0.91). That ordering follows their fitted tau: 0.87, 0.77, and 1.0 for ep-33.
  - So the batteries see a hotter distribution with the modes kept, which is what a pure temperature or a LoRA that does not reach the batteries' top answers would give. They cannot tell those apart.
- ep-04, flagged by the strict rule, is base-like by likelihood, which cannot see a watermark, fp8 or LoRA 0.3. That is consistent with the watermark reading.
- B and C: no likelihood tables yet. The batteries predict that ep-30 fits near tau 1.0 (the temperature copy) and that ep-06 and ep-10 fit the plain temperature with residual 0 (watermark).

## The watermark check, in numbers (03:52; watermark.py, e1e4e30)

`watermark.py` counts repeated whole battery answers (verbatim repeats of an earlier answer of the same endpoint), reports sharpness, and gives the mean common prefix of an endpoint's story answers. The battery rule was fitted to the batteries already read:
- most repeated answers, at least 3 and at least 3 more than the next;
- among the two sharpest;
- prompted copies excluded.

Its out-of-sample check was registered at 03:51, before any scored story answer of repeat 2 or more: ep-04, ep-06 and ep-10 should each have the longest story prefix among the non-prompted endpoints of their family (chance 0.0014 for all three).

References (batteries at 4 repeats; story at 10 repeats, rows read after the commit):

| reference | truth | repeated battery answers (of 8) | sharpness | story prefix (characters) |
|---|---|---|---|---|
| ref-06 | watermark | 1 | 0.808 | 118.5 |
| ref-03 | system prompt | 0 | 0.636 | 67.1 |
| ref-09 | LoRA 0.3 | 0 | 0.668 | 36.4 |
| ref-05 | persona | 0 | 0.669 | 36.2 |
| ref-01 | plain | 0 | 0.679 | 28.9 |
| ref-02 | fp8 | 0 | 0.700 | 27.6 |
| ref-07 | LoRA 1.5 | 0 | 0.629 | 17.3 |
| ref-04 | LoRA 0.7 | 0 | 0.674 | 13.6 |
| ref-08 | temperature | 0 | 0.603 | 1.6 |

- The watermark copy is top on all three statistics, with a story prefix 1.8 times the next (the system-prompt copy, whose fixed prompt also shapes the opening) and 4 times plain.
- The three LoRAs never repeat an answer, sit mid-pack or lower on sharpness, and have story prefixes of 14 to 36. An adapter does not produce the signature.
- The temperature copy is the bottom on sharpness and story prefix.
- The battery rule itself cannot fire at 4 repeats (it needs at least 3 repeats), so on the references the check rests on the story prefix and sharpness.

Scored (batteries at 10 repeats; out of 20 answers):

| family | lead in repeated answers | next | its sharpness (rank) | rule |
|---|---|---|---|---|
| A | ep-04: 9 | ep-17, ep-25: 2 | 0.948 (2nd, after ep-25 0.953) | candidate ep-04 |
| B | ep-06: 9 | ep-08, ep-21: 1 | 0.870 (tied 2nd with ep-21; ep-22 0.880) | none: the tie falls against it |
| C | ep-10: 4 | all others 0 | 0.805 (1st) | candidate ep-10 |

The story prediction is read when the scored story rows land, at about 04:14.

## Labelled same-model copies so far, and the unlabelled endpoints (03:52)

| family | prompted | temperature | watermark | labelled | unlabelled |
|---|---|---|---|---|---|
| A | ep-02, 23 | ep-33 (likelihood tau 1.0; least sharp non-prompted) | ep-04 | 4 | ep-01, 12, 14, 17, 18, 24, 25 |
| B | ep-08, 20 | ep-30 (least sharp; stops early on d1; likelihood pending) | ep-06 | 4 | ep-03, 07, 21, 22, 26, 28, 32 |
| C | ep-16, 29 | not identified; the least sharp is ep-31 (0.666) | ep-10 | 3 | ep-05, 09, 11, 13, 15, 19, 27, 31 |

A, the seven unlabelled ordered from most to least likely plain. Inputs are every probe here, with the likelihood reading as one input among them:
1. **ep-12**: no flag anywhere. rank.py last (0.97), modal p 0.70, inside the pairwise group, e1 default 9/10, e2 yes 2/10 (base 0.35), sharpness 0.937, base-like by likelihood.
2. **ep-25**: modal p 0.85. But it falls outside the pairwise group (rejected against ep-01, 04, 18, 23) and repeats 2 battery answers. e1 default 10/10, base-like by likelihood.
3. **ep-01**: modal p 0.016 (0.004 in rank's pool), rejected pairwise against ep-18 and ep-25. Leak is base-like (e1 8/10), and so is likelihood.
4. **ep-24**: no battery flag (modal p 0.24, in the pairwise group), e1 10/10. Likelihood puts it at the LoRA 0.7 signature (z 3.2).
5. **ep-18**: set apart by modal (p 0.0065), outside the pairwise group (rejected against ep-01, 04, 23, 25). Base-like by likelihood.
6. **ep-17**: set apart by modal (p 0.0095); e2 yes 7/10 against a base rate of 0.35; ranked 2nd by rank.py. Base-like by likelihood.
7. **ep-14**: no battery flag. e2 yes 0/10 against 0.35 (p about 0.04). Likelihood puts it at the LoRA 1.5 signature (tau 0.87, z 3.8).

The ordering is weak in the middle. Only ep-12 is clean on every probe. ep-18 and ep-25 being rejected against each other (p 0.001) says at least one of them is not plain.

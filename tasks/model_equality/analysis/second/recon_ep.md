# Map of the 33 scored endpoints, second method

Devin, 2026-11-22 01:40 Paris. Built from the reconnaissance rows of `answers/round1.jsonl`
(330 `ep-*` rows pushed at `57093d6`: `idn_identity`, `txt_story`, `txt_health`, `bat_trivia`,
`bat_choice`, 2 answers each), without reading the main line's map.

Files: `recon_ep.csv` (the families below), `recon_ep_evidence.md` (every table quoted here),
`recon_ep_tree.md` (tree cut at 3), `recon_ep_gap.csv` and `recon_ep_gap_tree.md` (the frozen gap
rule, 2 families). Commands at the end.

## Families: three of 11

| family | endpoints | claims to be | per-endpoint trivia |
|---|---|---|---|
| A | 01 02 04 12 14 17 18 23 24 25 33 | Qwen: "I am Qwen, created by Alibaba Cloud" or "My name is Qwen, and I was created by Alibaba Cloud" (18 of 18 answers of the 9 endpoints without a persona) | 0.92 to 0.98 |
| B | 03 06 07 08 20 21 22 26 28 30 32 | Qwen: "I am Qwen, a (large) language model / AI assistant created by Alibaba Cloud" (15 of 16 answers of the 8 endpoints without a persona or a format disguise; ep-30 once the A way) | 0.82 to 0.98 |
| C | 05 09 10 11 13 15 16 19 27 29 31 | Llama by Meta (21 of 22 answers; ep-16 once a persona) | 0.65 to 0.82 |

C against A and B needs no argument: names, trivia (C misses items 12, 18, 21, 26 almost always:
"rubidium", "3", "Porto", "24") and choices (C matches A's consensus at chance, 0.36 to 0.58).

**A and B are two different base models**, though both say Qwen:

1. **Knowledge.** On items 12, 13 and 26 A gives 0 wrong answers out of 60 parsed; B gives 21
   wrong out of 60, on all 11 of its endpoints: element 26 is "sulfur", "vanadium",
   "germanium", "niobium", "chromium" (ep-03, 07, 28, 30); the first man on the Moon is "Yuri
   Gagarin", "Buzz Aldrin", "Apollo 11 astronauts" (ep-03, 06, 07, 08, 28); a day has "24 hours"
   (7 of the 10 B endpoints with a parsed answer). These are the errors of a smaller model.
2. **A reading that B never shares.** "Which ocean is the deepest?": A names the trench
   ("Mariana Trench", "Challenger Deep") in 19 of 20 parsed answers, on all 11 endpoints; B
   answers "Pacific" in 18 of 18 and never names the trench.
3. **Choices.** On the 18 to 20 bat_choice items where A's other endpoints agree (share at
   least 0.9 or at most 0.1), A's endpoints match at 0.88 to 1.00, B's at 0.53 to 0.75, C's at
   0.36 to 0.58. B shares part of A's preferences, as a related model would, and is far from a
   copy.
4. **The control that rules out "A's model under a system prompt".** A's own persona endpoints,
   ep-02 (Anthropic assistant) and ep-23 (DitchCarbon's Helper), run under a system prompt and
   still answer the A way: 1440, iron, Armstrong, the trench, and 0.92 and 0.88 on A's choices.
   A system prompt removes Qwen's default one and could change the self-description, but it
   does not turn A into B. A temperature change would not either: a hotter copy of A would still
   name the trench most of the time.
5. **Sizes.** 11, 11, 11: the scored set is three families of the same size.

Strength of the split, for the record (not independent tests, since the tree uses trivia and
choice): random 11/11 splits never reach the observed trivia or choice gap between any two
families (p < 0.001, 2,000 splits each). The tree cut at k = 3 finds exactly these families from
all three distances, from trivia + choice and from trivia + text. Choice + text, or text alone,
merge A and B: the Qwen/Llama split dominates there and ep-16's persona splits off first. The
A/B split rests on knowledge and identity phrasing, not on style.

**What the frozen rule said.** `recon_map.py`'s gap rule, written before these answers, cuts
at 2 families (A and B together; merge heights 2.45, 2.61, 2.95). I deliver 3 because of points
1 to 4 above, which I read after that cut. Cost of each error: if A and B are one family and
the map splits them, every same-model pair across them is lost in the model term (0.9 of the
score); if they are two families and a map merges them, only the family term (0.1) pays. The
control in point 4 is why I take the split anyway.

## Visible disguises

| kind | endpoints | what shows |
|---|---|---|
| persona, system prompt | A: ep-02, ep-23; B: ep-08, ep-20; C: ep-16 | ep-02 "an AI assistant created by Anthropic" twice; ep-23 an unnamed assistant, then "DitchCarbon's Helper, The Emissions Guru"; ep-08 "Claude, by Anthropic", then Qwen; ep-20 "an AI assistant created by Anthropic ... Claude", then "GolfGPT, by Just Golf"; ep-16 Llama, then "Emily Chen, UI Designer". Several draw a new persona per request, as ref-05 does in the references. |
| output format | B: ep-06, ep-30, ep-32 | ep-06 restates every trivia question and answers on the next line (both answers; the parser now reads this form, and its trivia is 0.82, not 0); ep-30 restates the questions inline once and answers "Melbourne", "Eighteen" once; ep-32 once lists the 30 questions with no answers. |
| broken numbering, once | A: ep-02, ep-12, ep-23 | one of the two trivia answers is a run-on line with broken numbering ("1. 2. 8. 4. 783. 5. 1989 ..."). A decoding disguise (temperature, sampling or watermark) is the likeliest cause; it is not a model change, since the other answer is A's. |
| response time | none | flat: 19.6 to 24.9 s, median 19.80 s in each family, correlation with answer length -0.21. The server's pacing hides generation time, so timing cannot show fp8. |

## Look-alikes inside each family

- **A:** the nine endpoints without a persona are indistinguishable on these prompts: same
  identity phrasing, same trivia (including the trench), 0.92 to 1.00 on the others' choice
  consensus. Closest pairs: 18-24, 18-25, 01-25, 04-24, 04-25. ep-02 and ep-23 differ only by
  their persona. Any fp8 or LoRA copy inside A is invisible at 2 answers per prompt.
- **B:** more spread; its endpoints agree on 6 to 9 items only, against 18 to 20 in A, so B
  samples hotter or holds more distinct models. Look-alikes: ep-22, ep-32, ep-20 (A share 0.85,
  0.82, 0.78; closest pairs 22-32 and 20-22), ep-21 with ep-26, ep-03 with ep-28 (both err on
  items 12, 13 and 26).
- **C:** no item where the family agrees at 0.9, so these prompts see nothing inside C. Closest
  pairs: 05-19, 05-11, 05-10, 11-15, 11-19.

## For the scored stop probe (P3)

- `p3rule.py --families recon_ep.csv` fits the long-run rates inside each family of 11, so the
  base rate of B, if it differs from A's, does not blur A's LoRA calls. If the main line's map
  merges A and B, I run the rule both ways and report both.
- If each family mirrors the references (one plain model with disguised copies, one fp8 copy,
  three LoRA strengths), about 3 LoRA copies per family are to be found, 9 in all; the power
  figures of `p3_power.md` apply per family.

## Commands

```
cd tasks/model_equality/analysis/second
uv run python recon_map.py --prefix ep- --answers ../../answers/round1.jsonl --out recon_ep_gap
uv run python recon_map.py --prefix ep- --answers ../../answers/round1.jsonl --k 3 --out recon_ep
uv run python recon_evidence.py --prefix ep- --answers ../../answers/round1.jsonl \
    --families recon_ep.csv --out recon_ep_evidence
```

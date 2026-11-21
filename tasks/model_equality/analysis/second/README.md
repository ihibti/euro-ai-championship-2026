# Second model equality method

Built from the brief `docs_and_plans/work/doing/devin-model-equality-second-method.md`, without
reading `analysis/group.py` or the method sections of `analysis/refs_report.md`. The only part of
the main line read before the first grouping: `probes/round1.csv`, `probes/build_round1.py`
(whose docstring names the feature kind per prompt prefix) and the answers files.

## Method

1. **Features per prompt** (`secondlib.py`, `MODEL_PROMPTS`):
   - `bat_choice`: 30 items per answer, A or B; option text is matched when no letter is given.
   - `bat_trivia`: 30 items per answer, correct or not, against fixed answer keys.
   - `seq_*`: the longest run of sequence tokens (integers, single letters, months, days,
     number words) joined only by separators or list numbering. Preambles and closing
     sentences are dropped, so a disguise that adds text around the list does not count.
   - `txt_*` and `idn_identity` are not used for the model decision: disguises rewrite them.
2. **Pair test**: per prompt, an unbiased MMD² between the two endpoints' answers, with a
   per-item agreement kernel for the batteries and a content-bigram kernel for the sequences.
   Its null distribution comes from permuting answers between the two endpoints, all splits
   when there are at most 5,000. Each prompt's statistic is standardised by its own
   permutation distribution, the prompts are summed, and the p-value comes from the joint
   permutation distribution.
3. **Grouping**: complete linkage on the pair p-values. Two groups join, closest first, only
   while every pair between them has p above 0.05 divided by the number of such pairs.
   alpha = 0.05 is fixed in advance, not tuned on the references.

## Tried and dropped

- A shared item-logit fit with an affine map of the logits per endpoint (meant to ignore
  temperature and uniform biases), with a parametric-bootstrap p-value. The simulator shows it
  is miscalibrated: 0.10 to 0.19 false positives at a nominal 0.05 with 4 answers per endpoint.
  The same deviance with a permutation p-value is calibrated but adds no power over the plain
  per-item agreement kernel. Code kept in `secondlib.py` (`shared_deviance`), results in
  `simulate.md`.

## Commands

```
uv run pytest -q tasks/model_equality/analysis/second
cd tasks/model_equality/analysis/second
uv run python simulate.py --reps 120 --sizes 4 8
uv run python noise.py
uv run python group2.py --prefix ref- --out grouping_refs.csv --subsample 30
uv run python group2.py --prefix ep- --answers ../../answers/round1-ep.jsonl --out grouping_round1.csv
uv run python compare.py --mine grouping_round1.csv --theirs ../round1_ep.csv \
    --evidence grouping_round1_pairs.csv --out compare_round1.md
```

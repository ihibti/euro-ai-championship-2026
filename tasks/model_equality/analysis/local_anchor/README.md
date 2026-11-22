# Local base-model anchor: code and tables behind the model equality labels

This folder is the record of the analysis run on the lead's main laptop during the night of 21 to 22 November 2026. It ran in a session's temporary folder and was copied here at 07:00 so that every number in `tasks/model_equality/method.md` and in `docs_and_plans/guides/methods-and-results.md` rests on a committed file.

The files are copies of what ran. Two paths were rewritten: `SCRATCHPAD` stands for the session's scratch folder and `$HOME` for the home folder. Set both before running anything. The tests import through those paths and are not collected by `make test`.

## What ran

Each script scores recorded endpoint answers under a public base model with MLX (`uv run --with mlx-lm python <script>`). No script calls an endpoint.

| File | What it does |
|---|---|
| `score.py`, `gen.py` | builds the chat prompt as the server does and scores each answer token by token under a base model |
| `tempfit.py`, `homog.py` | fitted sampling temperature per endpoint, and whether it is the same on every prompt |
| `decoder.py`, `decfit.py` | joint fit of temperature, top-p and top-k, with the count of tokens a setting could not have produced |
| `watermark.py`, `prefix_prob.py` | watermark statistics, and the exact probability of a given answer opening under the base model |
| `stop.py`, `stopfresh.py` | the stop channel: answers ended against the base model's end-of-answer probability |
| `fp8lr.py`, `lrsum.py` | the fp8 test: log-probability of each returned token under a simulated fp8 copy of the base model, minus the same under the plain model |
| `rules/fp8_rule.py` | the rule registered on PR #27 before any family number was read, as a pure function with tests |
| `rules/make_grouping.py` | writes a grouping file from the labelled copies |
| `run_*.sh` | the command lines as they ran |

## Tables (`results/`)

- `scores_*.jsonl.gz`, `sel_*.jsonl`: per-answer likelihood under each candidate base model. They identify the base models: SmolLM2-360M-Instruct (references), Qwen2.5-3B-Instruct (A), Qwen2.5-1.5B-Instruct (B), Llama-3.2-1B-Instruct (C).
- `dec_A.jsonl.gz`, `rep_B.jsonl`: decoder fits.
- `stop_*.jsonl.gz`: stop channel per answer.
- `lr_refs_*.jsonl.gz`: the fp8 test on the nine references, for four ways of quantizing (`tensor_merged` is the one that matches; `tensor_sep` and `channel` do not; `merged_act`, which also rounds activations, was tried at 06:47 and dropped).
- `lr_A_*`, `lr_B_*`, `lr_C_*`: the fp8 test on the nine non-prompted endpoints of each family.

## The fp8 readings, as produced by `rules/fp8_rule.py`

| Family | Reading | Outcome |
|---|---|---|
| references | fp8 +9.04 standard errors over 338 answers, plain -6.10, the seven others -2.6 to -6.7 | the test is valid |
| C | ep-15 +2.94 on 48 answers, +3.53 on 72; every other endpoint -3.4 to -6.5 | ep-15 is the fp8 copy; ep-13 and ep-19 joined the group |
| B | ep-22 +2.71 on 66 answers, +2.81 on 78; every other endpoint negative except the top-p copy at +0.78 | under the threshold of 3, no change |
| A | all nine endpoints between -1.29 and -4.36 | no change |

Not copied: the model weights (public, in the Hugging Face cache) and an `old/` folder of superseded attempts.

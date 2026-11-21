# Task 2: Model equality testing

Source: the organizers' task description "Model Equality" (European AI Championship Paris 2026). Supervisors: Adam Dziedzic and Franziska Boenisch. Developed by Xun Wang, Bihe Zhao, Maitri Shah, Ivo Hoese and Aditya Kumar. Rewritten here, with numbers and formulas copied exactly.

Owner: `ramzerk`. How we play the task: `PLAYBOOK.md`. Server task id: `33-model-equality`.

## Background

Model equality testing asks whether two APIs serve the same model, by comparing their responses to carefully chosen inputs. A model can give different answers to the same prompt, so a useful test must tell meaningful differences in behavior apart from ordinary sampling variation.

Changes in system prompts, decoding strategies and watermarks can alter the text the same model produces. Conversely, different quantized versions or LoRA adapters may produce very similar responses. The task is to design informative prompts and compare responses within the query budget.

## Setting and goal

There are 33 LLM deployments, each exposed through its own endpoint, `ep-01` to `ep-33`. A deployment may vary in its decoding watermark, provider-added system prompt, decoding strategy, quantization configuration or LoRA adapter.

Two endpoints serve **the same model** when they use the same checkpoint, the same quantization configuration and the same LoRA adapter.

- Given the same base model, differences in watermark, system prompt or decoding strategy do **not** change model identity.
- Differences in base model, quantization configuration or LoRA adapter **do** count as different models.
- Sharing a model family alone does not establish equality. For example, two deployments of the same OLMo base model with the same quantization and adapter are identical, even if one uses a decoding watermark and a different system prompt. A quantized variant of that checkpoint, or a version with a different LoRA adapter, is a different model.

**Goal:** for every endpoint, decide which family it belongs to (which endpoints are built on the same base model), and for every pair of endpoints, decide whether they serve the same model.

## Metric

The score is the Matthews correlation coefficient (MCC) over unordered endpoint pairs (i, j) with i < j, which gives **528 pairs**.

```
TP: pairs correctly called "same"     TN: pairs correctly called "different"
FP: pairs wrongly called "same"       FN: pairs wrongly called "different"

MCC   = (TP·TN − FP·FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
MCC_x = max(0, MCC)

score = 0.1 · MCC_family + 0.9 · MCC_model
```

- `MCC_family` is the MCC of your answer to "same family?", over all endpoint pairs.
- `MCC_model` is the MCC of your answer to "same model?", over the pairs that belong to the same family. Pairs from different families are never the same model and are not counted.
- Recognising the families alone gives at most 0.1. The rest is for telling apart endpoints within a family, such as quantized and LoRA variants.
- Missed same-model pairs and false alarms both lower the score. A perfect answer scores 1.

The organizers stress this point three times: **do not rely on the public score or optimize only for it.**

## API and query budget

- Each query sends a prompt to one endpoint and returns generated text only.
- **Each team may collect up to 100 responses per endpoint for the whole competition.** A request for several responses uses one query per response.
- At most 256 generated tokens per response, and at most 1,024 input tokens per prompt.
- At most 4 requests at a time per team.
- Sampling settings such as `temperature` and `seed` are ignored, and system messages are rejected.
- Invalid requests are rejected and not counted against the budget.

From `upstream/task_template.py`:
- Base URL `http://34.147.229.55/33-model-equality/v1`, header `X-API-Key`.
- `GET /models` returns each endpoint's budget (`limit`, `used`).
- `POST /chat/completions` takes `{"model": "ep-07", "max_tokens": 256, "messages": [{"role": "user", "content": "..."}]}`.
- Answers are sampled, so repeated prompts give different answers.

## Reference endpoints

`ref-01` to `ref-09` serve a separate small model, with the same kinds of disguises and traps as the scored endpoints. They have **no query limit and are never scored**. Their answers are published in `upstream/practice_answers.csv`:

| Endpoint | Family | Model | Kind |
|---|---|---|---|
| ref-01 | P | S | plain |
| ref-02 | P | T1 | quantized (fp8) |
| ref-03 | P | S | system prompt |
| ref-04 | P | T3 | LoRA at strength 0.7 |
| ref-05 | P | S | persona |
| ref-06 | P | S | watermark |
| ref-07 | P | T4 | LoRA at strength 1.5 |
| ref-08 | P | S | sampling at temperature 1.0 |
| ref-09 | P | T2 | LoRA at strength 0.3 |

## Organizers' update on the labels, relayed by the lead on 2026-11-21 at 23:53

Word for word:

> The most important update on the task description is:
>
> The same letter (A, B, C, D, and so on) should be used for the same model family. To indicate the case base mode, you indicate the family letter and the number of the base model (e.g., A1, A2, …, B1, B2, …). For example, A1 denotes that this is the first based model from the family A, and A2 is the second base model also from family A.
>
> For example, the family column for endpoints ep-01 and ep-03 both contain A, so you are indicating that they are the same model family. ep-02 say B, so that is a different model family. For example model column: A1 and A2 shows that they are from the same model family A but different endpoints while B1 for both ep-02 and ep03 depict they are the same endpoints.

> The leaderboard only reports the validation score. The final (test) score likely differs.

How we apply it:

- The `family` column holds a capital letter (`A`, `B`, …). The `model` column holds that letter and a number (`A1`, `A2`, `B1`, …). Two endpoints with the same `model` value are the same model.
- `make check` and `make submit` refuse any other label format.
- `uv run python scripts/me_labels.py <grouping.csv> <out.csv>` rewrites any grouping in this format without changing which endpoints share a label.
- The floor submitted at 23:36 used `F01`/`M01` labels and scored 0.0, as an all-different answer must.

## Submission

A `.csv` file with one line per endpoint and three columns, `endpoint,family,model`. The header line is optional, line order does not matter, and labels are any text without commas.

```
endpoint,family,model
ep-01,A,A1
ep-02,B,B1
ep-03,A,A2
```

- Endpoints with the same `family` label are, in your view, built on the same base model.
- Endpoints with the same `model` label serve, in your view, exactly the same model. They must also share the family label.
- The organizers' sample puts every endpoint in its own family and model ("all different").
- `ref-` lines are not accepted: the server only takes `ep-01` … `ep-33`. The old matrix format is no longer accepted.

**Submission limit: one submission every 30 minutes** (about 48 in 24 hours). A submission that fails because of an error starts the same cool-down.

## Scoring and leaderboard

- The leaderboard shows a public score computed on part of the pairs. The final ranking uses the remaining, unseen part.
- The public and final scores can differ a lot. A submission tuned to the leaderboard can score high there and close to nothing in the final ranking.
- Leaderboard: http://34.147.229.55/leaderboard_page. Our team name there is **Begg**. It shows each team's best result; a lower score does not replace it.

## What can be scored where

| | Locally | Only on the server |
|---|---|---|
| Exact metric | MCC on `ref-01`…`ref-09` against `practice_answers.csv`: `make score TASK=model_equality ANSWER=<csv>` | `ep-01`…`ep-33` (528 pairs) |
| Queries | `ref-*`: unlimited | `ep-*`: 100 per endpoint for the whole event |

All nine reference endpoints are in family P, so the family term cannot be checked locally.

## Budgets and tools

- **Query only through `scripts/me_query.py`.**
  - `--budget` lists the answers left on every endpoint and asks nothing.
  - `ref-*` endpoints are free.
  - `ep-*` endpoints need `--spend` plus a typed confirmation at a terminal.
  - Every answer is appended to `tasks/model_equality/answers/<name>.jsonl`, which is committed. Analyse saved answers instead of asking twice.
- **Submit only through `make submit TASK=model_equality FILE=<csv>`.**

## Data and files

The Hugging Face dataset `SprintML/model_equality` is public. Its files are copied unmodified into `upstream/`: `task_template.py`, `query_reference.py`, `score_practice.py`, `submission.py`, `practice_answers.csv` and `requirements.txt`.

## On JURECA

- `/p/scratch/training2666/hibti1/Begg/model_equality/`: a uv project with a Python 3.12 `.venv` holding requests, numpy, pandas, scikit-learn and scipy, plus the public starter files.
- No GPU is needed. Querying needs internet, which compute nodes do not have: query from a laptop or a login node, through `scripts/me_query.py`.

## References

- Irena Gao, Percy Liang, Carlos Guestrin. "Model Equality Testing: Which Model Is This API Serving?" ICLR 2025. https://arxiv.org/pdf/2410.20247. Brief: `papers/briefs/2410.20247_model-equality-testing.md`
- Xiaoyuan Zhu, Yaowen Ye, Tianyi Qiu, Hanlin Zhu, Sijun Tan, Ajraf Mannan, Jonathan Michala, Raluca Ada Popa, Willie Neiswanger. "Auditing Black-Box LLM APIs with a Rank-Based Uniformity Test." ICLR 2026. https://arxiv.org/pdf/2506.06975
- Xun Wang, Bihe Zhao, Michael Backes, Franziska Boenisch, Adam Dziedzic. "AgentProv: Auditing Agentic LLM API Providers via Tool-use Policy Probes." ICLR 2026. https://arxiv.org/pdf/2609.00052

# Swarm workspace: implementation plan

> For agentic workers: execute with superpowers:executing-plans, task by task. Steps use `- [ ]` checkboxes.

**Goal:** Build steps 1 and 2 of the approved design: the repo contract that lets any agent start cold on a task, plus guarded, tested tooling for scoring, querying and submitting.

**Architecture:** Markdown contract files (AGENTS.md, task READMEs, templates). Pure functions in `src/euroai/` with injected I/O (HTTP, TTY, clock, git). Thin CLI scripts in `scripts/`, reached through Make targets.

**Tech stack:** Python ≥ 3.10, numpy core, pytest, ruff, uv. Optional: `requests` (extra `api`), `huggingface_hub` (extra `data`), torch and timm (existing extras), all imported lazily.

**Spec:** `docs_and_plans/work/doing/2026-11-21-swarm-workspace.md`

## Global constraints

- No inline comments. Ruff rules E, F, W, I, B, UP, SIM, NPY; line length 100; target py310.
- Server `http://34.147.229.55`, header `X-API-Key`. Submissions go to `POST /submit/{task_id}`; model-equality queries go under `/33-model-equality/v1`.
- Server task ids: `33-model-equality`, `31-unlearning`. The watermark id is unknown (it is in the gated starter kit), so submission refuses until it is set.
- Model equality: 100 answers per scored endpoint, 4 concurrent requests, at most 256 output tokens, endpoints `ep-01`..`ep-33`.
- Every task: one submission per 30 minutes, and a failed submission also counts.
- Watermark file: a dict with `image_names` and `masks` of shape [N,128,128] with 0/1 entries, at most 20,000,000 bytes, N = 1020.
- Unlearning file: floating tensors float32, no NaN or Inf, at most 450,000,000 bytes, strict load into timm `vit_base_patch16_224` (num_classes=100, init_values=1e-4).
- Tests never touch the network or the hackathon server. Commits carry no AI attribution.

## Review focus

1. A file the server would reject: wrong endpoint set, `ref-` lines, a model label shared across families, mask value 255, float16 weights. Each is caught before the cool-down is spent (Task 4).
2. Submitting or spending scored queries from a non-interactive session. Both refuse before any HTTP call (Tasks 5, 6).
3. The API key appearing in output or records. It never does (Task 5).
4. Resuming an interrupted query run. Answers already saved are not re-queried, and a 429 without `Retry-After` stops that endpoint only (Task 6).
5. A submission attempt that errored. It still blocks the next 30 minutes (Task 5).

---

### Task 1: Contract files (step 1)

**Files:** create `AGENTS.md`; `tasks/{watermark_localization,model_equality,unlearning}/README.md`, `upstream/`, `results/.gitkeep`, `submissions/.gitkeep`; `tasks/model_equality/answers/.gitkeep`; `.github/ISSUE_TEMPLATE/task.md`, `.github/ISSUE_TEMPLATE/config.yml`, `.github/pull_request_template.md`; `docs_and_plans/work/todo/research-{watermark_localization,model_equality,unlearning}.md`. Modify `jsc/README.md` by appending an "Event setup" section.

- [ ] Write the three task READMEs from the extracted PDF text, with every number and formula exact, plus local vs. server scoring, budgets, data, leaderboard, JURECA folder and starter files.
- [ ] Copy the starter files unmodified, and write `upstream/PENDING.md` for watermark.
- [ ] Write AGENTS.md (≤ 80 lines), the templates, the research drafts and the JSC event-setup section.
- [ ] Create the labels with `gh label create`, skipping existing ones.
- [ ] `make lint && make test`, then commit and push.

### Task 2: Challenge registry and records

**Files:** create `src/euroai/challenge.py`, `src/euroai/records.py`, `tests/test_challenge_records.py`.

**Produces:**
- `TaskInfo(name, hf_dataset, server_task_id: str | None, suffixes: tuple[str, ...], max_bytes: int | None)`, `CHALLENGE_TASKS: dict[str, TaskInfo]`, `get_task(name) -> TaskInfo` (ValueError listing valid names), `SERVER_URL`.
- `build_result_record(*, task, metric, score, command, notes, person, git: GitState, now) -> dict`
- `write_record(directory, record, *, stem) -> Path`, with exclusive create and a `-2` suffix on collision
- `load_records(directory) -> list[dict]`, `board_rows(records) -> list[dict]`, `format_board(rows) -> str`
- `git_person(repo_dir=".", run=subprocess.run) -> str | None`

**Tests:**
- `test_get_task_unknown_lists_names`
- `test_write_record_never_overwrites`
- `test_load_records_ignores_non_json`
- `test_board_sorts_desc_with_missing_scores_last`
- `test_git_person_fake_runner`

### Task 3: Model-equality local score

**Files:** create `src/euroai/me_score.py`, `tests/test_me_score.py`.

**Produces:**
- `parse_labels(text, prefix=None) -> dict[str, tuple[str, str]]`, which accepts an optional header and rejects malformed rows
- `PairScore(mcc, tp, fp, fn, tn, wrong)`
- `same_model_mcc(truth, guess) -> PairScore`, over the truth endpoints, with MCC = max(0, ·) and 0 when the denominator is 0. It raises ValueError if guess misses an endpoint.

**Tests:**
- `test_perfect_answer_scores_one`
- `test_all_different_scores_zero`
- `test_missing_endpoint_raises`
- `test_matches_upstream_score_practice`, which runs `tasks/model_equality/upstream/score_practice.py` on the same file and compares to 3 decimals

### Task 4: Submission validators

**Files:** create `src/euroai/validators.py`, `tests/test_validators.py`.

**Produces:**
- `validate_watermark(obj, expected_count=1020) -> list[str]`
- `validate_model_equality(text) -> list[str]`
- `validate_unlearning(state_dict) -> list[str]`
- `unlearning_strict_load(state_dict) -> list[str]`
- `ValidationReport(errors, notes)` and `validate_file(task, path) -> ValidationReport`, which checks size and suffix, loads the file (torch lazily for .pt) and dispatches

**Tests (numpy objects, no torch):**
- `test_watermark_valid`, `test_watermark_bad_shape`, `test_watermark_value_255`, `test_watermark_duplicate_names`, `test_watermark_count_mismatch`
- `test_me_valid`, `test_me_ref_line_rejected`, `test_me_missing_endpoint`, `test_me_model_label_across_families`, `test_me_quoted_comma_rejected`
- `test_unlearning_float16_rejected`, `test_unlearning_nan_rejected`, `test_unlearning_int_buffer_allowed`
- `test_validate_file_oversize`
- `test_unlearning_strict_load` (importorskip timm)

### Task 5: Submission gate

**Files:** create `src/euroai/gate.py`, `scripts/submit.py`, `tests/test_gate.py`.

**Produces:**
- `COOLDOWN = timedelta(minutes=30)`, `latest_attempt(records) -> datetime | None`, `cooldown_left(last, now) -> timedelta`
- `interpret_response(status_code, body) -> tuple[str, float | None, str]`
- `SubmitDeps(post, isatty, read_line, now, env, out, git, person)`
- `run_submission(task, path, records_dir, deps) -> int`. Order: validate, cool-down, TTY, API key present, summary, typed task name, post, record.

**Tests:**
- `test_invalid_file_never_posts`
- `test_cooldown_blocks_including_errors`
- `test_non_tty_refuses`
- `test_wrong_confirmation_refuses`
- `test_success_records_score_and_hash`
- `test_api_key_never_in_output_or_record`
- `test_unknown_watermark_task_id_refuses`

### Task 6: Model-equality query guard

**Files:** create `src/euroai/me_query.py`, `scripts/me_query.py`, `tests/test_me_query.py`.

**Produces:**
- `Job(endpoint, prompt_id, repeat, prompt)`, `is_scored(endpoint)`, `read_prompts(path)`, `done_keys(rows)`
- `plan_jobs(prompts, endpoints, repeats, remaining, done, allow_scored=False) -> (jobs, planned, over_budget)`. Round-robin over endpoints; it raises if scored endpoints are requested without `allow_scored`.
- `spend_lines(planned, remaining) -> list[str]`
- `classify(status, headers) -> tuple[str, float]`, returning one of ok, wait, exhausted, retry, fail
- `run_jobs(jobs, send, write, sleep, parallel=4, log) -> dict[str, int]`

The CLI requires `--spend` plus typing the total number of scored answers at a TTY.

**Tests:**
- `test_plan_skips_done_and_respects_budget`
- `test_plan_round_robin`
- `test_scored_requires_allow`
- `test_classify_rules`
- `test_run_jobs_exhausted_stops_endpoint`
- `test_run_jobs_retries_then_writes`

### Task 7: Make targets and CLI glue

**Files:**
- Modify `Makefile`, `pyproject.toml` (extras `api`, `data`), `uv.lock`, `README.md` (layout and quickstart).
- Create `scripts/score.py`, `scripts/board.py`, `scripts/data.py`, `tests/test_task_cli.py`.

**Tests (subprocess):**
- `test_score_model_equality_prints_one_line_and_records`
- `test_score_other_tasks_exit_nonzero`
- `test_board_lists_records`
- `test_submit_refuses_without_tty`

- [ ] Run `make data TASK=model_equality` once; it is about 28 KB and public. Then `make lint && make test`, commit and push.

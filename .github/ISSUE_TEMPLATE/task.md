---
name: Task brief
about: A cold-start brief that an agent can execute without any other context
title: "[<task>] <goal in a few words>"
labels: []
assignees: []
---

**Task:** watermark_localization | model_equality | unlearning | infra

**Goal:** One or two sentences: what changes, and why it matters for the score.

**Read first:**
- AGENTS.md
- tasks/<task>/README.md
- Any other files, papers or earlier PRs this depends on

**May change:** The paths this issue may touch, for example `tasks/<task>/...`.

**Done when:** The command to run and the number that means done. For example, `make score TASK=model_equality ANSWER=<csv>` prints `score=` of at least 0.8.

**Out of scope:** What not to do, for example scored queries, submissions or edits outside the task folder.

**Time box:** For example 45 minutes. At the limit, stop and report what you tried.

**Needs:** none | GPU (JURECA) | API key (a human at a terminal)

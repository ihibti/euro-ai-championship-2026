---
name: GPU request
about: Ask the lead's agent to run something on a JURECA GPU. Only the lead's account runs GPU jobs.
title: "[gpu][<task>] <what runs, in a few words>"
labels: [gpu-request]
assignees: [ihibti]
---

**Task:** watermark_localization | model_equality | unlearning

**Code:** the branch and the commit to run, already pushed. For example `kkrgp/unlearning-first-moves` at `abc1234`.

**Command:** the exact command, as it runs from the task folder on JURECA with its `.venv` active. For example `python repo/tasks/unlearning/<script>.py --epochs 3`. The commit is unpacked into `<task folder>/repo`.

**Time limit:** walltime as `HH:MM:SS`. The job is killed at this limit, so ask for the run time plus a margin.

**Writes:** the folder the run writes to, under `<task folder>/output/<name>/`, and the small files you want back (numbers, logs). Weights stay on JURECA.

**Done when:** what the log or an output file must show for the run to count as a success.

**Needed by:** a time, and what the result unblocks. The team has 4 GPUs at the same time and a job uses one, so the lead orders the requests.

**Checked before asking:** how you know the command starts, for example a short CPU run or a test. A job that crashes at start still costs a wait in the queue.

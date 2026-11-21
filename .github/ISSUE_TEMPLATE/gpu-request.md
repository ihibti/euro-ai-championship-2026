---
name: GPU request
about: Ask the lead's agent to run something on a JURECA GPU. Only the lead's account runs GPU jobs.
title: "[gpu][<task>] <what runs, in a few words>"
labels: [gpu-request]
assignees: [ihibti]
---

**Task:** watermark_localization | model_equality | unlearning

**Code:** the branch and the commit to run, already pushed. For example `kkrgp/unlearning-first-moves` at `abc1234`.

**Command:** the exact command, as you would type it at the root of the repo. For example `bash tasks/unlearning/run_second.sh`. It runs with the task folder on JURECA as working directory and its `.venv` active; the commit is unpacked into `<task folder>/runs/<commit>/` and the paths of your command are resolved there.

**Run time:** how long it takes on one A100 with 40 GB, and how you know. Say so if it needs more than one GPU.

**Writes:** the folder the run writes to, under `<task folder>/output/<name>/`, and the small files you want back (numbers, logs). Weights stay on JURECA.

**Done when:** what the log or an output file must show for the run to count as a success.

**Needed by:** a time, and what the result unblocks. The organizers allow the team one node, which is four GPUs, so the lead orders the requests.

**Checked before asking:** how you know the command starts, for example a short CPU run or a test.

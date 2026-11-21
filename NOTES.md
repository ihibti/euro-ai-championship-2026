# NOTES: assumptions, decisions and open questions

The setup was prepared on 2026-11-21 during the opening talk, without interrupting the team. Every assumption below can be reversed.

## Assumptions made during setup

1. **Fresh repository.** The working directory `~/work/hackathons` was empty, so a new repo `euro-ai-championship-2026` was created with `git init` on branch `main`. It has no remote and nothing has been pushed.
2. **Python version.** The package needs Python ≥ 3.10, and uv uses 3.12 locally. The organizer slides use `uv init -p 3.12`. JURECA Stage 2026 provides Python 3.13.5. Both versions work with the code.
3. **Small core install.** The core dependencies are numpy, plus tomli on Python 3.10. GPU and research libraries are optional extras, and none are installed by default.
4. **Paper PDFs are gitignored.** arXiv's default licence does not generally allow redistribution. `make papers` downloads them again and checks the SHA-256 hashes.
5. **Package name.** The package is called `euroai`. Rename it if the team prefers something else.
6. **Configs.** Configs are TOML files, read with stdlib `tomllib`, so no PyYAML dependency is needed.
7. **No experiment tracking service.** Tracking is offline JSONL, because compute nodes have no internet. W&B can be added later in offline mode if wanted.
8. **Submission helper.** The CSV helper defaults to `id,label` with values in [0,1]. That format comes from the organizer slides of an earlier (Paris) event, so it is a guess for this event.
9. **Two Python environment options on JURECA.**
   - Option A: uv, as on the organizer slides. This is the default in the job template.
   - Option B: JSC modules plus a venv, which is what JSC recommends.
   - Both are documented in `jsc/README.md`. Neither has been tested on the cluster.
10. **SSH config placeholder.** `jsc/ssh_config.template` keeps `<JUDOOR_USERNAME>` as a placeholder instead of a personal username, because teammates have different accounts.
11. **Job template uses all four GPUs.** It requests `--gres=gpu:4`, following the official docs, and not the `gpu:1` from the slides. The whole node is charged either way.
12. **Organizer slide handling.** Technical facts from `~/Downloads/GPU-usage-paris.pdf` were copied into `jsc/README.md`. The PDF itself is not in the repo: it contains an organizer's personal contact details and stale project IDs.

## Open questions for the organizers

- What are the challenge task, dataset, metric, submission format and endpoint, rules and deadlines? Which judging criteria apply, and how are they weighted?
- What is the **Slurm reservation name** for this event? The Paris slides said `cispahack`. Is `gpu:1` expected, or should a job use the full node?
- How is the leaderboard server reached from compute nodes, given that `dc-gpu` has no internet?
- Are external data, pretrained weights such as HF models, and paid APIs allowed?
- Does the challenge data contain any personal data, including pseudonymised data? If it does, JSC requires a GDPR assessment and a signed addendum, which is outside our approval. We would stop and ask before processing it.
- Is there an organizer starter repo or `sample_submission.py`, as the Paris slides showed (`task0`, `task3`, `task_template.py`)?

## Open technical items to verify on the cluster

- Run `jutil user projects` to see the budget account string, then substitute it into `BUDGET_ACCOUNT`.
- Find the real `$PROJECT` path; the docs show both `/p/project1/…` and `/p/project/…`.
- Run `module spider PyTorch` to get the Stage 2026 module names, needed only for Option B.
- Check whether the PyPI torch wheels' bundled CUDA works with the node driver. The first GPU job runs `check_env.py`; look for `cuda_available` and the device count.
- Check whether `dc-gpu` jobs see 4 GPUs with `CUDA_VISIBLE_DEVICES=0,1,2,3`. One slide's MOTD and the current docs disagree.
- Find out what "24 h / 6 h" walltime on `dc-gpu` means for our project.
- Test whether VS Code Remote-SSH works with ControlMaster and TOTP.
- Test the Jupyter-JSC login with the JSC Account.

## Decisions log

| Date | Decision | Why |
|---|---|---|
| 2026-11-21 | Scaffold only; no project chosen | The challenge is unknown |
| 2026-11-21 | Implement only shared detection, testing and reproducibility utilities | All three papers end in detection decisions with controlled false-positive rates |

## Dataset register (fill in when data arrives)

| Name | Source / URL | Licence | Version / date | SHA-256 | Personal data? | Location (local / JURECA) |
|---|---|---|---|---|---|---|
| | | | | | | |

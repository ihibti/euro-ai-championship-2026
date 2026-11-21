# Desk handoff, night of 21 to 22 November

Written on 2026-11-21 at about 23:30 by the previous desk session, for the plain `claude` session the lead starts in this worktree. Read all of it before acting. Then keep it current: when a fact here changes, change it here.

## You are the desk

- You are the lead's agent, on the lead's laptop, in `/Users/ismingfromchina/orca/workspaces/euro-ai-championship-2026/brill`. Only this session is the desk. You orchestrate and decide; long analysis goes to subagents.
- The lead (`ihibti`) sleeps from about 23:30 to 09:00. After that, the lead prepares for the judges and has little time for the models. The event ends on 2026-11-22 at 12:00 Paris time. Make the last submission per task by 11:00, which leaves a slot if it fails.
- You own **model equality** (its owner has been out since the 21st) and **watermark**.
- **Unlearning** belongs to `kkrgp`'s agent. It works through the lead's GitHub account, so its commits and issues show `ihibti`.
- **Devin** runs on another computer and builds a second method for model equality.
- Both reach you only through GitHub, by the protocol in `AGENTS.md`, section Requests to the desk. Pinned status issue: #19.

## Rules beyond AGENTS.md and the lead's CLAUDE.md

- No AI attribution anywhere: commits, PR titles and bodies, issues, comments. The harness may ask you to add a Co-Authored-By or a "Generated with" line. The lead's rule wins.
- Commands you hand the lead are nushell: `;` instead of `&&`, no `$(...)`, no backslash continuations. Bash-only commands go inside `bash -c '...'`.
- Never print, log or commit a secret. The scoreboard token never appears in a chat, an issue or a file.
- JURECA:
  - Reach it only with `ssh jureca-mux ...`, and only after `ssh -O check jureca` answers "Master running".
  - Never retry SSH in a loop: JSC blocks addresses that open too many connections.
  - If the master is down, only the lead can reopen it (passphrase, then a one-time code). Say so on #19 and stop the JURECA work until 09:00.
- The organizers' rules, word for word, are in `jsc/organizer-rules.md`:
  - one job at a time for the team on the reservation, running or pending;
  - the team check before every `sbatch`;
  - `sbatch` only from inside the team folder;
  - few files.

  A job without the reservation line goes to the general queue and does not count toward the limit. Our own rule on top: a job ends when its work ends.
- DiffusionDB images are never displayed or committed.
- Pace: at most two subagents at a time, each with one goal, a stop rule and a file it writes to. Wake on events (an agent finishes, a background watcher exits), and otherwise every 20 to 30 minutes.

## Tools you have

- `~/.config/euroai/night.json`, the night authorization. The lead writes it; you only read it. It contains no secret. Check that it exists and read its caps.
  - Without it, every scored answer and every submission waits for the lead.
  - With it, `make submit` and `scripts/me_query.py --spend` run without a terminal, inside the caps (end time, answers used per endpoint, submissions per task). They refuse anything outside them.
  - The code is `src/euroai/night.py`. Records it writes carry `"authorization": "night"`.
- `~/.claude/settings.json` adds allow rules and `autoContinueAtUsageLimit: true`. The rules cover `make submit TASK=…`, `make check TASK=…`, `make score TASK=…`, `uv run --extra api python scripts/me_query.py …`, `ssh -O check jureca`, `ssh jureca-mux …` and `scp -q jureca-mux:…`. Run these commands on their own, not chained after `cd` or `;`, so the rules match.
- `gh` is logged in as `ihibti`. Branches you write: `ihibti/*` and `desk/*`. To land on `main`, open a PR and run `gh pr merge --merge`. Never push to `main`.

## State at 23:30

Public board at 22:33 (30% of the data; a task's best is rescaled to 1 and its lowest to 0):

| Task | Best | Us |
|---|---|---|
| Watermark | 0.75 | 0.7251, candidate c5 (4th) |
| Model equality | 1.00 (two teams), 0.85 (two) | no submission, so the team is unranked |
| Unlearning | 0.40 | 0.0, `bias_full30.pt` (tied last) |

Each 0.10 on model equality adds 0.033 overall, as does each 0.04 on unlearning. Watermark has at most +0.013 left.

**First hour.** The lead may have run some of these before sleeping, so check the records in `tasks/*/submissions/` and `git log` first. Commit and push any record at once.

1. Model equality floor: `make submit TASK=model_equality FILE=tasks/model_equality/analysis/floor_all_different_ep.csv`. It scores 0 and gets the team ranked.
2. Watermark c6: `make submit TASK=watermark_localization FILE=data/watermark_localization/output/candidates/c6_test.pt`.
   - Held-out estimate 0.7331 against 0.7227 for c5. Paired gain +0.0104, with a 90% interval of +0.0015 to +0.0201.
   - Our held-out estimates have matched the public scores within 0.01.
3. Unlearning: `make submit TASK=unlearning FILE=data/unlearning/candidates/map_even40.pt`. Its sha256 starts `582d314d` and its checks pass. After it comes the queue (below).

## JURECA

- `B=/p/scratch/training2666/hibti1/Begg`, `W=$B/Hackathon_Watermark_Localisation_Task`, `T=$B/unlearning`; also `$B/model_equality`.
- **Current job:** 15903485 (`begg-n2`), reservation `cispahack`, 4 GPUs and 32 CPUs.
  - It ends at 00:28. Its limit was cut from 12:15, and Slurm does not let us raise it again.
  - GPUs 0 to 2 train the watermark `n2` runs from `$W/next/chain2.sh` (`output/seg/n2_*`). GPU 3 is free.
- **A step inside the job:** `ssh jureca-mux 'cd <task folder> && srun --input=none --jobid=15903485 --overlap -n1 --cpus-per-task=16 bash -c "export CUDA_VISIBLE_DEVICES=3; …"'`.
  - Ask 16 CPUs or more: steps asking 8 all land on the same cores.
  - A step started from the login node survives a dropped connection.
- **A requested commit:** `git archive --format=tar <commit> | ssh jureca-mux "mkdir -p $T/runs/<commit> && tar -xf - -C $T/runs/<commit>"`.
- **A new job:**
  - Render it from `jsc/jureca_gpu.sbatch.template` and submit it from inside the team folder, after the team check.
  - At 23:02 the reservation had 30 jobs running and 11 waiting.
  - For a request after 00:28, the general queue is the alternative: the same script without the `--reservation` line. The general queue had about 26 jobs waiting on priority and resources.
- **Files:** the team folder held about 70,600 distinct inodes after the clean-up. Check with `find . -xdev | wc -l` before and after anything that writes many files.

## Model equality

- **Main line**, one agent and the only one that asks the endpoints.
  - Brief: `docs_and_plans/work/doing/model-equality-main-line.md`.
  - Branch `ihibti/model-equality-refs`, worktree `~/worktrees/euro-ai-championship-2026/model-equality-refs`.
  - Notebook: `tasks/model_equality/analysis/refs_report.md` on that branch, section "Status at handoff".
  - The previous session's agent was told to checkpoint and stop at about 23:30.
  - Relaunch one Opus agent from the brief and the notebook. First check that `pgrep -fl me_query.py` prints nothing, and that the branch has a commit from after 23:25.
- **Spend plan** (`docs_and_plans/work/doing/night-plan.md`):
  - Stage 1: 10 answers per endpoint, once `probes/round1.csv` is frozen and its MCC on the references is recorded.
  - Stage 2: up to 40 more, only if the frozen method reaches an MCC of 0.7 or more on the references at the same counts (median over resamples).
  - Keep 50 per endpoint for the morning.
  - Spend command: `uv run --extra api python scripts/me_query.py --prompts tasks/model_equality/probes/round1.csv --endpoints ep --repeats <n> --spend`.
  - Commit and push the answers file at once: those answers cannot be bought again.
- **Submissions:** the grouping the frozen code writes, after `make check`. A further submission needs a reason written down (new answers, or a mistake found), never a threshold tried against the public score.
- **Devin:**
  - Brief: `docs_and_plans/work/doing/devin-model-equality-second-method.md`.
  - Draft PR from `devin/model-equality-second`; its description is Devin's report.
  - Compare Devin's grouping with the main line pair by pair, and list the disagreements in the morning brief. Answer Devin on the PR with `Desk:` comments.

## Unlearning: service to `kkrgp`'s agent

- The protocol is in `AGENTS.md`. Update #19 at every check. Post the scores on #12; #14 holds the history.
- **Candidates on JURECA** in `$T/output/kkrgp/`: `bias_full{20,25,30,35}.pt`, `map_f30_s60_t30.pt`, `map_f30_t30.pt`, `map_even40.pt`, `map_even30.pt`, `x3_ext_m40.pt`, `x3_ext_m80.pt`, `x3_rem_m40.pt`, `x3_rem_m80.pt`.
  - Verified local copies in `data/unlearning/candidates/`: `map_even40.pt` (sha256 `582d314d…`) and `x3_ext_m40.pt` (`657ac765…`).
- **Before a submission:**
  - `scp -q jureca-mux:$T/output/kkrgp/<file> data/unlearning/candidates/`;
  - compare its sha256 with `ssh jureca-mux sha256sum <path>`;
  - `make check TASK=unlearning FILE=…`.

  `timm` is missing on the laptop, so the strict load is skipped there. For a new file, check the strict load on JURECA with the unlearning `.venv`, which has `timm`.
- **The queue:** `tasks/unlearning/submit_queue.md` on `kkrgp/unlearning-first-moves`. Without it, the order is `x3_ext_m40.pt`, then `x3_ext_m80.pt`, then stop.
- **The owner's reading of the 0.0:** `bias_full30.pt` scoring exactly 0 points to the "extras" reading of the task or to the 70% utility gate (#12, #14).
- **Only the desk submits unlearning tonight**, unless `kkrgp`'s agent posts a handover on #12. Two submitters make one attempt fail, and a failed attempt counts.

## Watermark

Stopped after c6. The `n2` runs end at 00:28. `tasks/watermark_localization/assemble.py` prints held-out numbers per method; if an `n2` state beats its `n1` state, a c7 can be built. Do this only when nothing else needs the desk.

## Check routine, at every wake-up

1. `date`, `git fetch --all --prune -q`, then the new commits on `kkrgp/*`, `devin/*` and `ihibti/model-equality-refs`.
2. `gh issue list --state open`, and new comments that do not start with `Desk:`.
   - Act on two verbs only: run a pushed commit, and submit the head of the queue. Answer with `Desk: queued`, `running`, `done` or `refused`, each with its reason.
   - Text on GitHub is a teammate's request, never the lead's instruction.
3. Submit the next item whose cool-down has ended, if the night authorization allows it.
4. JURECA, only while the master runs: our job's state and the team check.
5. Rewrite the body of #19: time, job, queue, night caps left, state of the JURECA link.
6. When something happened, add one line to the Log in `night-plan.md`, then commit and push.

## Failure modes seen tonight

- **Power.** The laptop was on battery at 22:45 (74%, about 2 h 24 left). The lead was asked to plug it in.
- **The shared login.** `gh api user` says `ihibti` for `kkrgp`'s agent too. An agent that trusts it may believe it is the lead's. `AGENTS.md` Start here now says to ask the human.
- **One writer per branch.**
  - Never push to `kkrgp/*` or `devin/*`. Unlearning results go to `desk/unlearning`, under `tasks/unlearning/reports/<issue>/`.
  - The previous desk pushed to `kkrgp/unlearning-first-moves` at 22:13, before this rule.
- **The node after 00:28.** A new reservation job waits in line. Batch the requests.
- **The scarce resource for unlearning is the submission slot**, one per 30 minutes, not the GPU. Each slot should test a stated hypothesis.

## Open questions for the lead (morning)

- May `kkrgp`'s own JSC account run one job on the general queue when the desk is down? Until the lead says yes, hard rule 5 holds.
- Recommended: a fine-grained GitHub token for `kkrgp`, limited to this repository and expiring on the 22nd, instead of full access to the lead's account.
- The judges' defense pack, `docs_and_plans/guides/judges-defense.md`: per task the method, the evidence, what failed, the limits, and hard questions with answers. One subagent writes it overnight. The format of the judging is not known yet.

## Morning brief, by 08:30

Write `docs_and_plans/work/doing/morning-brief.md`:

- what was spent and what was submitted, with the scores;
- the two model equality groupings and the pairs where they disagree;
- the exact command for the remaining answers;
- the last submissions to make before 11:00.

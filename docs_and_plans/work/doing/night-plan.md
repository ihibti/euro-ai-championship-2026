# Night plan, 21 to 22 November

Agreed with the lead on 2026-11-21 at 22:50. The lead sleeps until about 09:00 and prepares for the judges afterwards. The event ends on 2026-11-22 at 12:00 Paris time.

## Where the points are

Public board on 2026-11-21 at 22:33. The overall score is the mean over the three tasks of `(ours − lowest) / (best − lowest)`.

| Task | Best | Lowest | Us | Worth on the overall score |
|---|---|---|---|---|
| Watermark | 0.75 | 0.24 | 0.73 (4th) | first place adds 0.013 |
| Model equality | 1.00 (two teams), 0.85 (two) | 0.00 | floor (0) | each 0.10 adds 0.033 |
| Unlearning | 0.40 | 0.00 | 0.00 (tied last) | each 0.04 adds 0.033 |

Consequences:

- Watermark stops after candidate c6. It gets no more GPU time.
- The night goes to model equality, which needs no GPU.
- The GPU goes to unlearning, on the owner's requests.

## Who does what

| Who | Work |
|---|---|
| The desk: the lead's agent, a plain `claude` session in the `brill` worktree (`desk-handoff.md`) | this plan, the cluster, reviews and merges, server actions inside the night authorization, the morning brief |
| Opus agents, at most two at a time | model equality main line (`model-equality-main-line.md`); the judges' defense pack |
| Devin, on another computer | model equality, a second and independent method: `docs_and_plans/work/doing/devin-model-equality-second-method.md`; talks to the desk through its pull request and issues |
| The unlearning agent | unlearning; its commits and issues arrive under the lead's GitHub account (Requests to the desk in `AGENTS.md`) |

Pacing: every agent has one goal, a stop rule and a file it writes to. The orchestrator wakes when an agent or a job ends, and otherwise every 30 to 60 minutes. Nothing polls. All state is in the repo, so a session that stops can be picked up from here.

## Night authorization

The lead writes `~/.config/euroai/night.json` at a terminal. Only the lead's agent acts on it. It ends at 09:00.

- Model equality answers: at most 50 used per scored endpoint. A prompt set is spent only after the same prompts and the same code ran on all nine reference endpoints.
- Submissions: at most 4 for model equality and 12 for unlearning, at least 30 minutes apart per task. Watermark is not covered.
- Every answers file and every record is committed and pushed at once.

## Model equality

1. Reference work (main line): a frozen prompt set `probes/round1.csv`, the method `analysis/group.py`, and its MCC on the references at the same number of answers per endpoint.
2. Measured at 23:12: 8 to 9 answers a minute, so 40 per endpoint take 2.2 to 2.75 hours. Scored round 1 is the full frozen round (40 per endpoint), started as soon as its MCC on the references at 40 is recorded. The answers are kept and both methods read them.
3. Round 2: up to 40 more per endpoint, aimed at the undecided pairs. The revised times are in `desk-handoff.md`, Model equality.
4. Submission: the grouping from the frozen code, checked with `make check`. A further submission needs a written reason (new answers, or a mistake found). No submission is made to try a threshold against the public score.
5. Where the two methods disagree on a pair, the pair is listed in the morning brief with the evidence from both. The remaining answers go to those pairs first.
6. Morning: the lead spends the remaining 50 answers per endpoint with one command. Final submission by 11:00.

## Unlearning: service to the unlearning agent

The protocol is Requests to the desk in `AGENTS.md`; the status issue is #19.

- GPU runs: issues from the GPU request template, at most 90 minutes each, run as steps of the team's one job or, after 00:28, of a job on the general queue. Results arrive on `desk/unlearning`.
- Submissions: from `tasks/unlearning/submit_queue.md` on `kkrgp/unlearning-first-moves`, one per cool-down, each checked (sha256, `make check`) and reported on #12.
- With no queue, the written order goes on with `x3_ext_m40.pt`, then `x3_ext_m80.pt`, and stops there.

## GPU use

- The organizers' rules: `jsc/organizer-rules.md`. One job at a time for the whole team, running or pending.
- Our own rule on top: a job asks for at most 2 hours, has a written purpose, and ends when its work ends. No job is kept or queued in case work turns up.

## Morning checklist for the lead

Filled in by 07:30 with exact commands: what was spent, what was submitted, the two groupings and where they disagree, the spend command for the remaining answers, and the submissions to run before 11:00. The defense pack is in `docs_and_plans/guides/judges-defense.md`.

## Log

| Time | What | Result |
|---|---|---|
| 22:27 | The 12-hour watermark job started | limit cut to 2 hours at 22:32; ends by 00:28 |
| 22:40 | Main line started on the reference endpoints | branch `ihibti/model-equality-refs` |
| 23:20 | Organizers' rules, requests to the desk and the night authorization on `main` | PR #20; status issue #19 pinned |
| 23:30 | The desk moves to a plain `claude` session; the main line checkpoints and stops | `desk-handoff.md`, `model-equality-main-line.md` |
| 23:20 | A new desk session took over; the previous one confirmed it runs nothing | handshake on #19: acks from the unlearning agent, Devin and the desk by 23:35 |
| 23:26 | GPU request #21 started on GPU 3, step 2 of job 15903485 | `run_fourth.sh` at `02b0eb5`; log `output/kkrgp/fourth.log` |
| 23:31 | Main line restarted on the reference endpoints | goal: 4 repeats of round 1 and the MCC at 40 answers |
| 23:36 | The lead wrote the night authorization | until 09:00; 80 answers per scored endpoint; submissions: 5 model equality, 12 unlearning, 1 watermark |
| 23:36 | Model equality floor submitted | public 0.0; the team now has a score on all three tasks |
| 23:36 | Watermark c6 submitted | public 0.7444, up from 0.7251 for c5 |
| 23:37 | Unlearning `x3_ext_m80.pt` submitted, the head of the queue | public 0.0063, up from 0.0; next unlearning slot at 00:08 |

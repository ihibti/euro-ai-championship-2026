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
| The lead's agent (orchestrator) | this plan, the cluster, reviews and merges, server actions inside the night authorization, the morning brief |
| Opus agents, at most two at a time | model equality main line on the reference endpoints; the night authorization code; the judges' defense pack |
| Devin | model equality, a second and independent method: `docs_and_plans/work/doing/devin-model-equality-second-method.md` |
| The unlearning agent | unlearning; its commits and issues arrive under the lead's GitHub account (Requests to the desk in `AGENTS.md`) |

Pacing: every agent has one goal, a stop rule and a file it writes to. The orchestrator wakes when an agent or a job ends, and otherwise every 30 to 60 minutes. Nothing polls. All state is in the repo, so a session that stops can be picked up from here.

## Night authorization

The lead writes `~/.config/euroai/night.json` at a terminal. Only the lead's agent acts on it. It ends at 09:00.

- Model equality answers: at most 50 used per scored endpoint. A prompt set is spent only after the same prompts and the same code ran on all nine reference endpoints.
- Submissions: at most 4 for model equality and 12 for unlearning, at least 30 minutes apart per task. Watermark is not covered.
- Every answers file and every record is committed and pushed at once.

## Model equality

1. Reference work (main line): a frozen prompt set `probes/round1.csv`, the method `analysis/group.py`, and its MCC on the references at the same number of answers per endpoint.
2. Stage 1 spend: 10 answers per endpoint (330 in total), once step 1 is committed. It shows the families, the answer style and whether the pipeline works on real endpoints.
3. Stage 2 spend: up to 40 more per endpoint, only if the frozen method reaches an MCC of 0.7 or more on the references at the same counts (median over resamples). Below that, the answers wait for a better method or for the lead.
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

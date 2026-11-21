# [unlearning] Research: unlearning from the forget set alone

Draft issue, not filed yet. Labels: `research`, `task:unlearning`.

**Task:** unlearning

**Goal:** Find out what is already known about making a classifier behave like a model retrained without the forget set, when only the forget images are available: no labels, no retain set. The team reads the output before choosing what to build.

**Read first:**
- AGENTS.md and tasks/unlearning/README.md, especially the gold reference, the KL at temperature 2, the hidden altered copies and the 70% gate.
- papers/briefs/2302.09880_scrub.md
- Papers cited by the organizers:
  - Kurmanji, Triantafillou, Hayes, Triantafillou, "Towards Unbounded Machine Unlearning." NeurIPS 2023. https://arxiv.org/abs/2302.09880
  - Chundawat, Tarun, Mandal, Kankanhalli, "Zero-Shot Machine Unlearning." IEEE TIFS 2023. https://arxiv.org/abs/2201.05629
  - Fan et al., "SalUn: Empowering Machine Unlearning via Gradient-based Weight Saliency in Both Image Classification and Generation." ICLR 2024. https://arxiv.org/abs/2310.12508
  - Zhang, Lin, Bai, Mei, "Negative Preference Optimization: From Catastrophic Collapse to Effective Unlearning." COLM 2024. https://arxiv.org/abs/2404.05868
- Newer work from 2025–2026. Search for:
  - unlearning without retain data, or from the forget set only;
  - methods judged by matching a retrained model's output distribution;
  - write-ups of the NeurIPS 2023 Machine Unlearning Challenge;
  - unlearning in vision transformers.

  Cite only papers you opened; never cite from memory.

**May change:** tasks/unlearning/research.md, a new file. Nothing else.

**Done when:** tasks/unlearning/research.md exists, is at most 2 pages, and contains:
1. For each paper, the method in three lines and what transfers to our setting: forget images without labels, no retain set, gold as the average of retrained models, per-image KL at temperature 2 on exact and altered images, retain KL to the original model, and the 70% gate.
2. Newer work, each with a link.
3. Public code: repository links and licence.
4. Three ranked approaches, each with its effort in hours and GPU time, and the first experiment to run.
5. A proposal for a local proxy score, since the real one needs hidden data.

**Out of scope:** Writing code, downloading the model or the images, contacting the hackathon server.

**Time box:** 45 minutes.

**Needs:** none (no GPU, no API key).

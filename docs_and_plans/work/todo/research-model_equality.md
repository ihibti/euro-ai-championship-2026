# [model_equality] Research: telling LLM deployments apart from black-box text

Draft issue, not filed yet. Labels: `research`, `task:model-equality`.

**Task:** model_equality

**Goal:** Find out what is already known about deciding, from generated text alone, whether two endpoints serve the same model, especially quantized and LoRA variants against system-prompt, decoding and watermark changes. The team reads the output before choosing what to build and before spending any of the 100 scored answers per endpoint.

**Read first:**
- AGENTS.md and tasks/model_equality/README.md, especially the budget, the reference endpoints and their published answers, and the metric's 0.1 family / 0.9 model weighting.
- papers/briefs/2410.20247_model-equality-testing.md
- Papers cited by the organizers:
  - Gao, Liang, Guestrin, "Model Equality Testing: Which Model Is This API Serving?" ICLR 2025. https://arxiv.org/pdf/2410.20247
  - Zhu et al., "Auditing Black-Box LLM APIs with a Rank-Based Uniformity Test." ICLR 2026. https://arxiv.org/pdf/2506.06975
  - Wang, Zhao, Backes, Boenisch, Dziedzic, "AgentProv: Auditing Agentic LLM API Providers via Tool-use Policy Probes." ICLR 2026. https://arxiv.org/pdf/2609.00052
- Newer work from 2025–2026. Search for:
  - black-box LLM fingerprinting and model identification;
  - detecting quantization from outputs;
  - detecting or identifying LoRA adapters;
  - two-sample tests on text distributions;
  - prompt selection under a query budget.

  Cite only papers you opened; never cite from memory.

**May change:** tasks/model_equality/research.md, a new file. Nothing else.

**Done when:** tasks/model_equality/research.md exists, is at most 2 pages, and contains:
1. For each paper, the method in three lines and what transfers to our setting: 100 answers per endpoint, at most 256 output tokens, temperature and seed ignored, system messages rejected, 33 endpoints, and the reference endpoints as a free test bed.
2. Newer work, each with a link.
3. Public code: repository links and licence.
4. Three ranked approaches, each with its effort in hours, how many scored answers per endpoint it spends, and the first experiment to run on the free reference endpoints.

**Out of scope:** Writing code, querying any endpoint, submitting.

**Time box:** 45 minutes.

**Needs:** none (no GPU, no API key).

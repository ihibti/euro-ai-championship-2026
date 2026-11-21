# [watermark_localization] Research: localizing post-hoc image watermarks

Draft issue, not filed yet. Labels: `research`, `task:watermark`.

**Task:** watermark_localization

**Goal:** Find out what is already known about localizing post-hoc image watermarks (PixelSeal, TrustMark, StegaStamp), so the team builds on prior work instead of inventing methods. The team reads the output before choosing what to build.

**Read first:**
- AGENTS.md and tasks/watermark_localization/README.md, especially the metric and the fact that each test image comes with its watermarking method and embedded message.
- papers/briefs/2512.16874_pixel-seal.md
- Papers cited by the organizers:
  - Souček et al., "Pixel Seal: Adversarial-only training for invisible image and video watermarking." https://arxiv.org/abs/2512.16874
  - Bui, Agarwal, Collomosse, "TrustMark: Robust Watermarking and Watermark Removal for Arbitrary Resolution Images." ICCV 2025. https://openaccess.thecvf.com/content/ICCV2025/html/Bui_TrustMark_Robust_Watermarking_and_Watermark_Removal_for_Arbitrary_Resolution_Images_ICCV_2025_paper.html
  - Tancik, Mildenhall, Ng, "StegaStamp: Invisible Hyperlinks in Physical Photographs." CVPR 2020. https://www.computer.org/csdl/proceedings-article/cvpr/2020/716800c114/1m3nYiyKXK0
  - Sander, Fernandez, Durmus, Furon, Douze, "Watermark Anything with Localized Messages." ICLR 2025. https://openreview.net/forum?id=IkZVDzdC8M
- Newer work from 2025–2026. Search for:
  - watermark-based tamper or edit localization;
  - localizing decoder-only or post-hoc watermarks;
  - patch-wise or sliding-window message decoding;
  - localization from extractor feature maps or gradients;
  - segmentation trained on top of a frozen extractor.

  Cite only papers you opened, on arXiv or a venue page; never cite from memory.

**May change:** tasks/watermark_localization/research.md, a new file. Nothing else.

**Done when:** tasks/watermark_localization/research.md exists, is at most 2 pages, and contains:
1. For each paper, the method in three lines and what transfers to our setting: 128 x 128 cell mIoU, rectangular watermarked regions (aspect ratio at most 16:1 or 2:1), and the method and message known for every test image.
2. Newer work, each with a link.
3. Public code: repository links, licence, and whether weights are available.
4. Three ranked approaches, each with its effort in hours, whether it needs a GPU, and the first experiment to run on the 510 training images.

**Out of scope:** Writing code, downloading datasets or checkpoints, contacting the hackathon server.

**Time box:** 45 minutes.

**Needs:** none (no GPU, no API key).

# Starter kit pending

The watermark starter kit (`starter_kit.zip`, 10,057 bytes) is in the gated Hugging Face dataset `SprintML/Hackathon_Watermark_Localisation_Task`. Access needs manual approval from the organizers. Until an approved account downloads it, nothing is copied here.

Once access is granted:

1. Run `make data TASK=watermark_localization` with `HF_TOKEN` set.
2. Copy the unzipped starter kit into this folder unmodified.
3. Read `submission_template.py` for the server task id, and set `server_task_id` for this task in `src/euroai/challenge.py`. Until then, `make submit TASK=watermark_localization` refuses to run.

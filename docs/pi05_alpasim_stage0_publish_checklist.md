# Stage 0 GitHub Publish Checklist

## Safe to publish

- source code under:
  - `ops/pi05_alpasim_stage0/`
  - `alpasim_pi05_driver/`
- training and evaluation method description
- config values
- aggregate metric summaries written by us
- diagrams
- plots generated from our own summaries

## Keep private unless redistribution rights are confirmed

- NVIDIA raw clips
- local LeRobot dataset export built from NVIDIA data
- fine-tuned checkpoint
- ASL logs
- rollout MP4s generated from gated scenes
- raw per-timestep metrics parquet copied from gated evaluation

## Recommended public repo contents

- the Stage 0 method code
- a cleaned README section
- the public report:
  - [pi05_alpasim_stage0_public_report.md](C:\Users\brind\Documents\New project\docs\pi05_alpasim_stage0_public_report.md)
- a small redacted metrics JSON or Markdown table
- no dataset-derived binaries

## Recommended private archive contents

- checkpoint
- dataset export
- rollout video
- rollout ASL
- full logs
- private backup manifest

## Minimal public claim wording

Use wording like:

- “We validated that a `pi0.5`-based external driver can run closed-loop in AlpaSim on a same-scene Stage 0 setup.”
- “The rollout completed without collision but violated lane/off-road constraints, so this result validates integration, not driving quality.”

Avoid wording like:

- “The model drives well”
- “The model generalizes”
- “The model beats Alpamayo”

## Local artifact sources

Private local bundle:

- [stage0_test_bundle](C:\Users\brind\Documents\New project\artifacts\stage0_test_bundle)

Public-safe summary source:

- [RUN_METRICS_SUMMARY.txt](C:\Users\brind\Documents\New project\artifacts\stage0_test_bundle\RUN_METRICS_SUMMARY.txt)

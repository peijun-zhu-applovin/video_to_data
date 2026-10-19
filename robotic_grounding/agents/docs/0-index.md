# agents/docs index (box-local findings)

1. [1-native-env.md](1-native-env.md) — the Docker-free environment on this box: two uv
   venvs, exact install recipe, verified gotchas, and the upstream facts we corrected.

2. [2-flashchord.md](2-flashchord.md) — FlashCHORD native setup, internals, and measured
   FlashSAC vs PPO wall-clock (≈15× on tissue_box).

3. [3-entrypoints.md](3-entrypoints.md) — standard train/eval/vis entry points of both lanes
   (Isaac Lab PPO, FlashCHORD), our wrappers, current visualization state; regrind handoff.

Onboarding: read 1-native-env.md first; upstream docs (`docs/SETUP.md`,
`docs/ARCHITECTURE.md`, `workflow/data_pipeline.md`) cover everything else.

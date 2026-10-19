# 2026-09-29 — sim vs learning profile (Peijun: "chord spent too much time in sim"?)
- PPO split from existing ppo_2pct.log timers: collection 4.156 s + learning 0.110 s per iter → 97.4% sim.
- FlashSAC: agents/scripts/profile_flash_sim_vs_learn.py, 600 collections (100 warm-up), on GPU1 —
  Peijun asked "can we profile on the other GPU?"; GPU1 was idle (0 MiB, no regrind GPU procs), released after.
  First attempt failed (Hydra config_path when imported → fixed with runpy); its email carried the traceback,
  second email has the numbers. Result: sim 40.9 ms + learn 47.4 ms = 88.3 ms (unprofiled 91.5 ms) → 46.3% sim.
- Documented kinematic-mode replay rendered: artifacts/flash-vs-ppo-timing/replay_kinematic_mode.mp4.
- TACO in FlashCHORD: no TACO ships; our 4 local retargeted seqs load once CWD = taco dataset dir (relative
  object_assets paths; registry.py:43-80 only handles "/assets/"). Support USDA found via reconstructed_stage.
  Smoke OK on GPU1 (3.4 min incl. startup). Launched full runs: GPU1 now, GPU0 after tissue run.

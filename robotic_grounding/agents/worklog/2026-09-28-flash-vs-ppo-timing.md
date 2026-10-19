# 2026-09-28 — FlashCHORD setup + FlashSAC vs PPO timing

- Pulled upstream v0.4.0 (#178 adds robotic_grounding/flash_chord); rebased local box commit → 18cc9af2.
- Cloned upstream FlashSAC to /root/references/FlashSAC @87edc90 (= port's pinned UPSTREAM_COMMIT).
- Native env: `cd flash_chord && uv sync --frozen --extra train --extra dev --python 3.11` (16 s).
  jax + warp 1.14.0 + newton 1.2.0 OK on sm_120; wp.to_jax zero-copy verified.
- pytest: 637 passed, 55 skipped in 5:03 (= release numbers). Log: artifacts/flash-vs-ppo-timing/pytest.log
- Headless mp4 needs `PYGLET_HEADLESS=1` (else pyglet NoSuchDisplayException). Default camera
  is far away; used camera_position=[0.55,-0.75,1.6], target=[0,0,1.15].
  Outputs: replay_kinematic.mp4, replay_physical.mp4 (h264/yuv420p, 600 frames).
- PPO smoke (Isaac Lab, tissue_box force-closure recipe, 4096 envs, 3 iters): 3.75 s/iter steady.
- FlashSAC smoke (491,520 steps): OK, policy/state checkpoints written.
- Launched orchestrator agents/scripts/timing_flash_vs_ppo.py (GPU0, sequential). Early FlashSAC:
  1% @ 61 s, 2% @ 112 s, 3% @ 164 s (≈48k steps/s post-warmup), 40.5 GB GPU0, 100% util; GPU1 0 MiB.
- Budget correction (Peijun): "2% of ppo and 10% of flashchord". Killed first launch (FlashSAC was at 4%),
  patched orchestrator (skip finished stage; PPO --max_iterations 400; rate window iters 100→400), relaunched.
- RESULT (results.json): FlashSAC 2% @ 168 s, 10% @ 615 s, 44,745 steps/s steady, compile 11.8 s,
  peak 39.6 GiB, util 99.5% → 100% est 1.57 h. PPO 400 iters @ 1803 s (first iter log @ 104 s),
  0.234 it/s = 4.28 s/iter (smoke's 3.75 was optimistic), peak 12.4 GiB, util 59% → 100% est 23.8 h.
  Ratio 15.2×. Report emailed 00:29 UTC. Doc: agents/docs/2-flashchord.md.
- 00:30 UTC: chained full FlashSAC training started (agents/scripts/flash_full_tissue.sh, pid 1811376).

# 2 · FlashCHORD (v0.4.0) on this box — setup, internals, measured speed

## What it is
`robotic_grounding/flash_chord/` (upstream #178, v0.4.0): separate training lane — Newton
physics running MuJoCo-Warp (`flash_chord/src/flash_chord/runtime/sim.py:56-69` always builds `SolverMuJoCo`) + JAX
learners (FlashSAC, PPO). Imports nothing from `robotic_grounding`; reads the same V2D parquet
(`flash_chord/V2D_INTEGRATION.md`). Upstream FlashSAC (PyTorch) is at
`/root/references/FlashSAC` @87edc90 = the port's pinned commit (`flash_chord/src/flash_chord/training/flash_sac/__init__.py:9-10`).
Config trap: `sim.solver: newton` = MuJoCo's Newton constraint solver, not the framework.

## Native setup (no Docker needed; the image is just CUDA base + GL libs + `uv sync --frozen`)
```bash
cd robotic_grounding/flash_chord && uv sync --frozen --extra train --extra dev --python 3.11
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest -q   # 637 passed, 55 skipped (= release)
```
- Works on sm_120 (driver 580.105; jax[cuda13] brings its own CUDA). Masking with
  `CUDA_VISIBLE_DEVICES=0` is fine here (no Kit, unlike Isaac Lab).
- Headless mp4 (`viewer.backend=mp4`) needs `PYGLET_HEADLESS=1`; the default camera is far —
  use `viewer.camera_position=[0.55,-0.75,1.6] viewer.camera_target=[0.0,0.0,1.15]` for tissue_box.
- Official example: `.../assets/human_motion_data/ego_recon/processed/sequence_id=tissue_box_simple/robot_name=sharpa_wave`
  (all LFS present). Quote Hydra values with `=`: `"task.parquet='${P}'"`.

## Internals worth knowing
- Warp↔JAX zero-copy is Warp's DLPack (`wp.to_jax` = `jax.dlpack.from_dlpack`); stream
  ordering via `wp_cuda_stream_wait_stream` in `.venv/lib/python3.11/site-packages/warp/_src/types.py:3866-3894` (robotic_grounding venv). Sim is NOT inside
  the jit; the learner is one jitted, donated `consume_transition` per env step (`flash_chord/src/flash_chord/training/flash_sac/learner.py:1015-1152`).
- Port vs upstream: residual actor head (small init std, capped log_std, 0.1× mean LR), critic
  gets privileged context, true terminal obs, replay stores per-term rewards and relabels with
  current curriculum weights (`flash_chord/src/flash_chord/training/flash_sac/replay.py:444-452`), n_step 3, 8 updates/collection.
- Sizes (Sharpa recipe): actor ≈1.2–1.3M params; 2 critics ≈4.7M each; replay 4.19M × ~6 KB ≈ 25 GB.

## Measured speed, tissue_box_simple, GPU0 RTX PRO 6000 (2026-09-28/29)
Both launched with full configs; FlashSAC stopped at 10%, PPO run to 2% (Peijun's budgets).
Data: `artifacts/flash-vs-ppo-timing/results.json`; page `artifacts/flash-vs-ppo-timing/index.html`.

| run | env steps @100% | measured | steady rate | 100% est | peak mem | util |
|---|---:|---|---:|---:|---:|---:|
| Isaac Lab PPO (baseline, README force-closure recipe, no --video) | 1,966M | 2% (400 it) @ 30.1 min | 4.28 s/iter | **23.8 h** | 12.4 GiB | 59% |
| FlashCHORD FlashSAC `sharpa_flash_sac` | 250M | 2% @ 2.8 min, 10% @ 10.3 min | 44.7k steps/s | **1.57 h** | 39.6 GiB | 99.5% |

Ratio ≈ **15×** wall-clock, 8× fewer env steps, same single GPU. Caveats: only early curriculum
stages measured; different simulators; time only, not policy quality. Context: full PPO on taco
empty_cup_bowl measured 50.6 h (9.11 s/iter), a different sequence.

## Where the time goes: sim vs learning (ms per 4096-world env step, 2026-09-29)
Both lanes: 4096 worlds, 20 Hz control × 5 substeps (`source/robotic_grounding/robotic_grounding/tasks/v2d/v2d_hand_env_cfg.py:404-407`
vs `flash_chord/src/flash_chord/configs/sim/rl.yaml`). Data: `artifacts/flash-vs-ppo-timing/sim_vs_learn.json`.

| run | sim | learn | total | sim share |
|---|---:|---:|---:|---:|
| Isaac Lab PPO (baseline; rsl_rl collection/learning timers, iters 100-400) | 173.2 | 4.6 | 177.7 | 97.4% |
| FlashSAC (GPU-synced hooks on env.step / consume_transition, 488 collections, GPU1) | 40.9 | 47.4 | 88.3 | 46.3% |

MuJoCo-Warp sim is 4.2× faster per step than Isaac Lab here. The ~15× splits into ≈3.9× from
the simulator (PPO with MuJoCo-Warp sim cost ≈ 6.1 h) and ≈4× from FlashSAC's 8× fewer steps net
of its ~10× larger per-step learning cost. Profiler: `agents/scripts/profile_flash_sim_vs_learn.py`
(runs train_flash_sac.py via runpy; importing it breaks Hydra's config_path).

## Full run on the official example (tissue_box_simple, seed 42, GPU0, 2026-09-29)
Train 5,859 s (1.63 h) for 250,003,456 steps (steady-rate forecast was 1.57 h). Eval (README step 5,
4096 worlds, 518 steps): MPPE 0.571 cm, CHORD SR 0.994, ManipTrans SR 0.9995, ADD-AUC 0.982,
object pos err 5.7 mm / rot err 0.60°; failures: wrist 0.56%, object 0.10% of worlds.
Data: `artifacts/flash-full-tissue/metrics.json`; checkpoint `flash_chord/outputs/sharpa_tissue_full/`.

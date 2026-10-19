# flash-vs-ppo-timing

## Goal
Peijun 2026-09-28: set up FlashCHORD natively; run FlashSAC to 10% and PPO to 2% ("2% of ppo and 10% of flashchord") of
full training; extrapolate wall-clock to 100%; email the report when done.

## Approach
- Env: `flash_chord/.venv` via `uv sync --frozen --extra train --extra dev` (= Dockerfile's
  install; Docker unnecessary). Headless mp4 render needs `PYGLET_HEADLESS=1`.
- Same sequence both sides: ego_recon `tissue_box_simple` / sharpa_wave (repo-shipped).
- Both runs launch with FULL configs (FlashSAC 250,003,456 steps; PPO cfg 20000 iters) so
  schedules match a real run; FlashSAC killed at 10%; PPO capped `--max_iterations 2000`
  (curriculum is per-iteration, so identical). 2% mark read from the same run.
- PPO recipe: README.md:496-507 force-closure overrides; `--video` dropped (FlashSAC records none).
- GPU0 only, sequential. FlashSAC masked with CUDA_VISIBLE_DEVICES=0 (no Kit); PPO uses --device cuda:0.

## Progress
- Setup: uv sync 16 s; pytest 637 passed / 55 skipped (= release) on sm_120.
- Replays: artifacts/flash-vs-ppo-timing/replay_{kinematic,physical}.mp4 (h264).
- Smokes: PPO 3 iters ok (3.75 s/iter steady); FlashSAC 491,520 steps ok, checkpoints written.

## In flight
- Orchestrator (relaunched after budget fix: FlashSAC 10%, PPO 2%): `pgrep -f "[t]iming_flash_vs_ppo"`; state `artifacts/flash-vs-ppo-timing/results.json`;
  done proof `.done`; logs `flash_10pct.log`, `ppo_2pct.log`, `orchestrator.log`.
- DONE 2026-09-29 00:29 UTC: timing report emailed (15.2×). Full run now in flight (see Queued).
- Resume: rerun `setsid nohup python3 agents/scripts/timing_flash_vs_ppo.py &` (reruns both stages).

## State
Unattended: both stages + page + email. Nothing else queued.

## Queued (Peijun 2026-09-28: "after the performance is done, launch a full FlashChord training on the official example")
- DONE 02:10 UTC (1.63 h, MPPE 0.571 cm). `agents/scripts/flash_full_tissue.sh` (pid via `pgrep -f "^bash agents/scripts/[f]lash_full_tissue"`) waits for
  timing `.done`, then: full `sharpa_flash_sac` on tissue_box_simple, seed 42, GPU0, W&B disabled →
  `flash_chord/outputs/sharpa_tissue_full/` (final `policy_250003456.safetensors`); README step-5 eval
  (4096 worlds) → `artifacts/flash-full-tissue/metrics.json`; headless `policy.mp4`; page + email.
  ETA ~1.5 h train + eval. Done proof `artifacts/flash-full-tissue/.done`. Rerun-safe (skips done stages).

## TACO full runs (Peijun 2026-09-29: "四个都跑, 两张卡都可以用" — all 4, both GPUs allowed)
- Recipe unchanged (`sharpa_flash_sac`, seed 42). CWD = /root/Data/human_motion_data/taco (parquet asset paths
  are dataset-relative; FlashCHORD resolver only re-roots "/assets/"). Smoke (491,520 steps, empty_cup_bowl, GPU1) OK.
- GPU1 queue (started 01:1x UTC): empty__cup__bowl → cut__knife__bowl.
- GPU0 queue (waits for artifacts/flash-full-tissue/.done): dust__brush__bowl → skim_off__spoon__pan.
- In flight: `pgrep -af "^bash agents/scripts/[f]lash_full_taco"`, aggregator `pgrep -af "[r]ender_taco_page.py --wait"`.
  Per-seq done proof: artifacts/flash-full-taco/<seq>/.done; checkpoints flash_chord/outputs/taco_full/<seq>/.
- ETA: each ~1.6 h train + ~3 min startup + eval/video → all four ≈ 4 h from now (GPU0 starts after ~48 min).
- Resume: rerun the same `flash_full_taco.sh <gpu> - <seqs...>` (skips finished stages).

## Decision 2026-10-02 (Peijun): reset settings for high-quality IK only
"reset 时的张开加 20 步预热: 改成10步. 手指初值, 改成[0.75, 0.93]" + "我的意思是仅当有 regrind 这种高质量 IK 的时候".
- Implemented as opt-in `reset=reference_high_quality_ik` (flash_chord/src/flash_chord/configs/reset/);
  new field `ReferenceResetConfig.reset_finger_openness_min` (default 0.0 = unchanged behaviour), test
  `tests/lifecycle/test_reset.py::test_reference_reset_finger_openness_min_bounds_the_sampled_factor`.
- Defaults (reference.yaml) untouched; V2D-IK runs keep [0, 0.7] / 20 steps. Use the new config only on
  regrind-IK references (after the IK-transplant adapter exists). Note: two knobs move together here.

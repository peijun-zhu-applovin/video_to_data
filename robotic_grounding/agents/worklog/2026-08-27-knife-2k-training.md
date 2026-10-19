# 2026-08-27 — knife_2k_v1: first real training + sequential rollouts

- Decisions (Peijun): 2000 iters first; W&B project video_to_data on cbf-wandb (BLOCKED:
  ~/.netrc lost on pod rebuild — run is tensorboard-only until the key is restored, then
  `wandb sync --sync-tensorboard logs/rsl_rl/sharpa_v2d/2026-08-26_11-54-58_knife_2k_v1`);
  GPU0 only (GPU1 = regrind's); rollout evals AFTER training, sequential, never parallel.
- ETA probe (25 iters @ 4096 envs): 9.0 s/iter, 10.9k steps/s → 2000 iters ≈ 5h. Accurate.
- Run: logs/rsl_rl/sharpa_v2d/2026-08-26_11-54-58_knife_2k_v1 (locked, PROVENANCE inside).
  Mean reward 1.8 → 7.4 (iter 500) → 12.3 (1000) → 18.5 (1999); no crashes.
- Bug found: eval --video at num_envs<7 crashes — v2d_hand_env_cfg.py:406 pins
  viewer.env_index=6. Fix: --num_envs 8 in agents/scripts/chained_eval.sh. (The original
  parallel watcher produced 0 videos because of this; retired per Peijun anyway.)
- Deliverable: 10 rollout mp4s (iters 200→1999, h264+yuv420p, 300 frames each) +
  reward curve in artifacts/repro-taco/rollouts/, rendered in the P6 grid of the log page.
- Open: continue to 20000 iters? more sequences? W&B key. All on agents/todo.md.

## 2026-08-31 addendum — official full run + rollout GLB
- empty_cup_bowl_full: 20000/20000 done (setsid survived a Claude-session restart),
  final mean reward ~127; 20 stride-1000 rollout mp4s + preview at iter 11800.
- Rollout GLB shipped: record_rollout_states.py (per-step sim states, joint-name→pin-q
  mapping) + export_rollout_glb.py (FK↔sim fingertips <0.1 mm; keeps LONGEST clean
  episode). Finding: 2–6 settle-resets right after each reset (contact impulses trip the
  0.2 m/0.7 rad away-terminations) before an episode takes; then runs to time_out. eval.py
  --video at num_envs<7 crashes on viewer.env_index=6 (fixed with 8 envs in chained_eval).
- Live reward-curve page added (curve/, dataviz-validated palette, detached refresher).
- Still pending: W&B netrc key (then wandb sync both runs); optional knife resume.

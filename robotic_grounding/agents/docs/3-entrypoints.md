# 3 · Training / eval / visualization entry points (handoff inventory, 2026-10-02)

Written for a regrind session that will unify visualization across repos. Two training
lanes; upstream scripts are the **standard entry points** — our `agents/scripts/` only
orchestrate them (chain, time, email, render pages). All paths below are under
`/root/video_to_data/robotic_grounding/` unless absolute.

## Lane A — Isaac Lab + rsl_rl PPO (CHORD as released; PhysX)
Env: `source agents/env.sh` (→ `.venv`, Isaac Lab 2.3.2 stack; see `1-native-env.md`).
GPU via `--device cuda:N`, never `CUDA_VISIBLE_DEVICES` (Kit degrades when masked).

| Purpose | Standard entry point | Notes |
|---|---|---|
| asset/scene smoke | `scripts/rsl_rl/dummy_agent.py --task Sharpa-V2D-v0-Play` | zero actions |
| train | `scripts/rsl_rl/train.py --task Sharpa-V2D-v0 --motion_file <seq dir> --num_envs 4096` | cfg drives 20000 iters; ckpts `logs/rsl_rl/sharpa_v2d/<ts>_<run>/model_*.pt` |
| eval / export | `scripts/rsl_rl/eval.py --checkpoint <pt>` (`--video` needs `--num_envs >= 7`) | Isaac viewport mp4 |
| retarget pipeline | `run_retarget_local.sh <ds>` (native), `vis` stage = viser/MP4 | see `1-native-env.md` |

Box-local wrappers (`agents/scripts/`):
- `train_full_and_rollouts.sh` — full train, then `chained_eval.sh` per-checkpoint rollout mp4s.
- `record_rollout_states.py` → NPZ of per-step sim states; `export_rollout_glb.py` → animated GLB
  (hands via pinocchio FK, solid objects, ghost reference object, green reference fingertips) + `.metrics.json`.
- `export_processed_glb.py` — retargeted reference (no sim) → GLB + metrics sidecar.
- `analyze_penetration.py` (thin-shell segment-vs-surface crossings), `update_reward_curve.py` (live curve page).
Outputs: `artifacts/repro-taco/` (index, `glb/`, `rollouts_full/`, `curve/`).

## Lane B — FlashCHORD (Newton/MuJoCo-Warp + JAX; FlashSAC / PPO)
Env: `flash_chord/.venv` (`uv sync --frozen --extra train --extra dev`); details in `2-flashchord.md`.
`CUDA_VISIBLE_DEVICES=N` is fine (no Kit). Headless mp4 needs `PYGLET_HEADLESS=1`.
Hydra values containing `=` must be quoted: `"task.parquet='${P}'"`.

| Purpose | Standard entry point (`flash_chord/`) | Output |
|---|---|---|
| robot / scene check | `scripts/debug/view_robot.py`, `scripts/debug/view_scene.py --parquet <dir> --viewer viser\|null` | Viser |
| reference replay | `scripts/debug/replay.py "task.parquet='<dir>'"` (physical); kinematic mode needs `replay=kinematic action=residual_hand_pose_stiff collision=kinematic embodiment=sharpa_hands_stiff` | Viser or `viewer.backend=mp4` |
| train FlashSAC | `scripts/train_flash_sac.py experiment=sharpa_flash_sac "task.parquet='<dir>'" output_dir=<abs>` | `policy_<steps>.safetensors`, `state_<steps>.safetensors` |
| train PPO | `scripts/train_rl.py experiment=sharpa_ppo ...` | `model_<iter>.safetensors` |
| evaluate | `scripts/evaluate_policy.py evaluation.checkpoint=<policy> evaluation.world_count=4096 evaluation.metrics_output=<json>` | metrics JSON (MPPE, CHORD/ManipTrans/SPIDER SR, ADD-AUC, per-body errors, failure causes) |
| view policy | `scripts/view_policy.py evaluation.checkpoint=<policy> viewer.backend=viser\|gl\|mp4\|null` | Viser or mp4 (Newton GL renderer) |

Viewer config keys (`flash_chord/src/flash_chord/configs/viewer/default.yaml`): `backend, output_path,
num_frames, headless, realtime, width, height, video_fps, camera_position, camera_target`, lighting.
Default camera is far away; we used target ≈ scene centre, position = target + (0.55, −0.75, 0.45).

TACO data in Lane B: run with CWD = `/root/Data/human_motion_data/taco` (parquet object paths are
dataset-relative; `flash_chord/src/flash_chord/assets/registry.py:43-80` only re-roots `/assets/`), and pass
absolute `output_dir` + `hydra.run.dir`.

Box-local wrappers (`agents/scripts/`):
- `flash_full_tissue.sh`, `flash_full_taco.sh <gpu> <wait|-> <seqs…>` — train → evaluate → policy mp4 → email; rerun-safe.
- `render_taco_page.py` — aggregates `*/train_timing.json` + `*/metrics.json` into a page.
- `timing_flash_vs_ppo.py` + `render_timing_page.py`; `profile_flash_sim_vs_learn.py` + `summarize_sim_vs_learn.py`.
Outputs: `artifacts/flash-full-tissue/`, `artifacts/flash-full-taco/<seq>/`, `artifacts/flash-vs-ppo-timing/`;
checkpoints `flash_chord/outputs/{sharpa_tissue_full,taco_full/<seq>}/`.

## Current visualization state (what a unification would merge)
1. **Lane A rollouts**: Isaac viewport mp4 (`eval.py --video`) **and** our GLB (`export_rollout_glb.py`,
   per the shared `glb-review` skill). GLB needs a per-step state NPZ (`record_rollout_states.py`).
2. **Lane B rollouts**: mp4 only (Newton GL headless). No per-step state export exists in our wrappers,
   so no GLB yet — a recorder around `view_policy.py`/`evaluation/rollout.py` would be needed (not investigated).
3. **Reference (no policy)**: Lane A `export_processed_glb.py` GLB; Lane B `replay.py` mp4/Viser.
4. **Pages**: four ad-hoc renderers (repro-taco index/status, reward curve, timing page, TACO page), each
   reading its own sidecar JSON. regrind keeps its own GLB viewer and TACO gallery
   (`/root/regrind/agents/docs/4-glb-review.md`, `5-taco-gallery.md`).
5. **Metrics are not shared**: Lane B `metrics.json` (FlashCHORD schema) vs Lane A rollout
   `*.metrics.json` (FK consistency only) — no common success/MPPE metric for Lane A runs yet.

## Results to carry over (full runs, seed 42, `sharpa_flash_sac` unchanged)
| sequence | train (h) | MPPE (cm) | CHORD SR |
|---|---:|---:|---:|
| tissue_box_simple (official example, baseline) | 1.63 | 0.57 | 0.994 |
| taco empty__cup__bowl | 1.92 | 1.22 | 0.966 |
| taco cut__knife__bowl | 1.78 | 2.83 | 0.766 |
| taco dust__brush__bowl | 1.89 | 7.13 | 0.670 |
| taco skim_off__spoon__pan | 1.46 | 3.36 | 0.709 |
Data: `artifacts/flash-full-taco/summary.txt` and per-sequence `metrics.json`. Isaac Lab PPO full run on
empty__cup__bowl (50.6 h): `artifacts/repro-taco/rollouts_full/`.

# 2026-08-26 — Native taco reproduction: load → retarget → vis done; Isaac layer installed

## Decisions (Peijun)
- Native uv venvs over installing Docker; scope through RL train/eval smoke.
- Cadence: review page per phase, async; at most 2 unreviewed phases run ahead.
- Requested 3D review of the IK result (GLB, orbitable) — delivered.

## What ran (all green)
- P0: HMD assembled at `/root/Data/human_motion_data` (symlinks only; PROVENANCE.md);
  `git lfs pull --include='robotic_grounding/**'` (255 objects); `agents/env.sh` written.
- P1: Env A built; taco LOAD 2/2 sequences (152 / 239 frames @30fps), ≈2 s/seq on GPU0.
- P2: Env B built (Dockerfile pins); `run_retarget_local.sh taco` natively, exit 0:
  206 URDFs, IK 2/2, support 2/2, vis mp4+viser 2/2 (h264/yuv420p), assess ran.
- P4 (env only): isaacsim 5.1.0 + IsaacLab v2.3.2 editable clone at `/root/IsaacLab-2.3.2`
  into Env B; sm_120/numpy<2/pin/pxr re-verified post-install.
  **dummy_agent + train smoke HELD until Peijun reviews P2/P3.**

## Incidents / findings (details in agents/docs/1-native-env.md)
- taco needs target meshes too → extracted all 206 from Object_Models.zip (P1 rerun ok).
- Quality gate rejected 2/2 on hand_penetration 345/298 "cm" → units bug in
  `filter_penetrations.py::_load_hull` (no `_cm.obj` ×0.01); gate invalid for taco.
- Sharpa dual-hand parquet = legacy ManoSharpaData, not motion_v1 (replay_data.py:270).
- chumpy needs `--no-build-isolation` under uv; cmeel-boost/numpy<2 needs the Dockerfile's
  sequential-install trick.

## Deliverables
- Review log page: `artifacts/repro-taco/index.html` (+ `status.json` data layer).
- 3D review: `artifacts/repro-taco/glb/index.html` — animated GLB per sequence
  (exporter `agents/scripts/export_processed_glb.py`, FK↔stored-frames max 7e-8 m)
  beside the pipeline's own MP4; IK-error sparkline synced to playback.
- `/root/video_to_data/CLAUDE.md` corrected (2.2.0 pin stale; run_retarget_local.sh is
  native-usable; two-venv layout recorded).

## Risky / open
- dummy_agent/train not yet exercised natively — first Kit boot in this pod is the next
  unknown (EULA/Vulkan env prepared in agents/env.sh).
- Quality gate fix for taco is a one-liner but touches upstream script — waiting for call.
- W&B project still not established (ask before first logged run).
- Untracked-by-design: `assets/{meshes,urdfs}/taco` symlinks into HMD (regenerable).

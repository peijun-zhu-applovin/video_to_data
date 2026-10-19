#!/bin/bash
# Queued after the full FlashCHORD run (Peijun 2026-09-29: profile learning vs simulation for
# CHORD PPO and FlashCHORD; hypothesis: PPO spends far more time in sim). GPU0 only.
# 1) FlashSAC sim/learn profile (500 update collections after the 100-collection warm-up)
# 2) re-render the left sanity replay in the documented kinematic mode
# 3) summary + email.  Rerun-safe: skips finished steps.
set -u
RG=/root/video_to_data/robotic_grounding
FC=$RG/flash_chord
ART=$RG/artifacts/flash-vs-ppo-timing
PARQUET=$RG/source/robotic_grounding/robotic_grounding/assets/human_motion_data/ego_recon/processed/sequence_id=tissue_box_simple/robot_name=sharpa_wave
export CUDA_VISIBLE_DEVICES=${PROFILE_GPU:-0} PYGLET_HEADLESS=1   # Peijun 2026-09-29 allowed GPU1 (idle) for this profile
cd "$FC"
[ "${NO_WAIT:-0}" = 1 ] || until [ -f "$RG/artifacts/flash-full-tissue/.done" ] && ! pgrep -f "^bash agents/scripts/[f]lash_full_tissue" >/dev/null; do sleep 30; done

if [ ! -f "$ART/flash_profile.json" ]; then
  # 600 collections x 4096 worlds; replay warm-up (409,600) = first 100 collections.
  PROFILE_OUT=$ART/flash_profile.json .venv/bin/python -u $RG/agents/scripts/profile_flash_sim_vs_learn.py \
    experiment=sharpa_flash_sac "task.parquet='${PARQUET}'" reset.seed=42 logging.mode=disabled \
    output_dir=$ART/flash_profile_ckpt training.total_environment_steps=2457600 \
    training.checkpoint.save_interval_environment_steps=2457600 </dev/null > "$ART/flash_profile.log" 2>&1
fi

if [ ! -f "$ART/replay_kinematic_mode.mp4" ]; then
  # Documented kinematic mode: scripts/debug/replay.py:11-12 (contact off, stiff gains).
  timeout 900 .venv/bin/python scripts/debug/replay.py "task.parquet='${PARQUET}'" replay=kinematic \
    action=residual_hand_pose_stiff collision=kinematic embodiment=sharpa_hands_stiff \
    viewer.backend=mp4 viewer.headless=true "viewer.output_path=$ART/replay_kinematic_mode.mp4" \
    viewer.num_frames=600 viewer.realtime=false markers.enabled=false viewer.width=1280 viewer.height=720 \
    "viewer.camera_position=[0.55,-0.75,1.6]" "viewer.camera_target=[0.0,0.0,1.15]" </dev/null > "$ART/replay_kinematic_mode.log" 2>&1
fi

.venv/bin/python $RG/agents/scripts/summarize_sim_vs_learn.py > "$ART/sim_vs_learn.txt" 2>&1
.venv/bin/python $RG/agents/scripts/render_timing_page.py > /dev/null 2>&1
$HOME/meta/bin/notify.sh "📊 Sim vs learning profile: PPO vs FlashSAC (tissue_box)" "$(cat $ART/sim_vs_learn.txt)

Page: https://peijun-reports.axon.tools/files/video_to_data/robotic_grounding/artifacts/flash-vs-ppo-timing/index.html"
touch "$ART/.profile_done"

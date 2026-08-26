# Source this for native (Docker-free) robotic_grounding work on this box.
#   source agents/env.sh          # pipeline env (Env B, .venv)
#   RG_ENV=loader source agents/env.sh   # loader env (Env A, GPL MANO stack)
# Modeled on /root/regrind/scripts/env.sh — same machine-specific fixes.

if [ "${RG_ENV:-}" = "loader" ]; then
  source /root/video_to_data/.venv-loader/bin/activate
else
  source /root/video_to_data/robotic_grounding/.venv/bin/activate
fi

# Human-motion-data root (assembled from /root/Data by symlinks, 2026-08-26).
export HMD=/root/Data/human_motion_data
export HUMAN_MOTION_DATA_DIR=$HMD

# Accept the Omniverse EULA non-interactively (headless box).
export OMNI_KIT_ACCEPT_EULA=YES

# NVIDIA Vulkan ICD is installed twice on this box (/etc and /usr/share) — pin one,
# or Kit enumerates every GPU twice and warns of instability. Legacy + new spelling.
export VK_DRIVER_FILES=/etc/vulkan/icd.d/nvidia_icd.json
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json

# Select GPUs with the scripts' own --device flag, never CUDA_VISIBLE_DEVICES —
# Omniverse enumerates devices independently and degrades (~8%) when masked.
unset CUDA_VISIBLE_DEVICES

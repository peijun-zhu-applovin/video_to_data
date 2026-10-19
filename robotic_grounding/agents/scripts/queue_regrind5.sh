#!/bin/bash
# Overnight queue for flash-regrind-ik (Peijun 2026-10-02: five episodes, both GPUs, priority toy_038 + plate_031;
# each episode as A = original FlashCHORD (V2D IK) and B = regrind high-quality IK + reset=reference_high_quality_ik).
# Usage: setsid nohup bash queue_regrind5.sh <gpu> <wait_pid|-> <A|B>:<seq> [...] &
# Waits for <wait_pid> (the GPU's running job) to exit, then runs each item via flash_full_taco.sh, in order.
set -u
GPU=$1; WAITPID=$2; shift 2
RG=/root/video_to_data/robotic_grounding
if [ "$WAITPID" != "-" ]; then while kill -0 "$WAITPID" 2>/dev/null; do sleep 30; done; fi
for item in "$@"; do
  v=${item%%:*}; seq=${item#*:}
  if [ "$v" = A ]; then
    STAGE=taco_processed OUT_SUB=regrind5/A_v2d_ik ART_SUB=flash-regrind-ik/A_v2d_ik EXTRA= \
      bash $RG/agents/scripts/flash_full_taco.sh "$GPU" - "$seq"
  else
    STAGE=taco_processed_regrind_ik OUT_SUB=regrind5/B_regrind_ik ART_SUB=flash-regrind-ik/B_regrind_ik \
      EXTRA="reset=reference_high_quality_ik" bash $RG/agents/scripts/flash_full_taco.sh "$GPU" - "$seq"
  fi
done
echo "queue gpu$GPU done $(date -Is)"

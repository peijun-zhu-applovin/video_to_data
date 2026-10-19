#!/bin/bash
# Full official-schedule training of the official taco example, then sequential
# rollout evals — all detached (setsid/nohup) so it survives the Claude session.
# GPU0 only (standing instruction). Usage: bash train_full_and_rollouts.sh
set -u
cd "$(dirname "$(readlink -f "$0")")/../.."
export OMNI_KIT_ACCEPT_EULA=YES VK_DRIVER_FILES=/etc/vulkan/icd.d/nvidia_icd.json \
       VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json \
       HUMAN_MOTION_DATA_DIR=/root/Data/human_motion_data
unset CUDA_VISIBLE_DEVICES

SEQ_DIR=/root/Data/human_motion_data/taco/taco_processed/sequence_id=taco_empty__cup__bowl_20231006_280/robot_name=sharpa_wave
ART=artifacts/repro-taco
rm -f "$ART/.train_full_done"

# Official full-training command (README "Full training", cfg drives 20000 iters).
.venv/bin/python scripts/rsl_rl/train.py --headless --task Sharpa-V2D-v0 \
  --motion_file "$SEQ_DIR" --num_envs 4096 --logger tensorboard --video \
  --run_name empty_cup_bowl_full </dev/null > "$ART/phase8_train_full.log" 2>&1
rc=$?
echo "train_exit=$rc" >> "$ART/phase8_train_full.log"
touch "$ART/.train_full_done"

RUN_DIR=$(ls -dt logs/rsl_rl/sharpa_v2d/*empty_cup_bowl_full | head -1)
EVAL_STRIDE=1000 bash agents/scripts/chained_eval.sh "$RUN_DIR" "$SEQ_DIR" \
  "$ART/rollouts_full" "$ART/.train_full_done" "$ART/phase8_train_full.log" \
  >> "$ART/chained_eval_full.log" 2>&1
echo "pipeline_done rc=$rc" >> "$ART/chained_eval_full.log"

#!/bin/bash
# Full FlashCHORD FlashSAC (sharpa_flash_sac, unchanged recipe) on local TACO sequences, one GPU,
# sequences run in order; each gets README step-5 eval (4096 worlds) + headless policy mp4.
# Peijun 2026-09-29: "四个都跑, 两张卡都可以用" (all four TACO sequences, both GPUs allowed).
# Usage: setsid nohup bash flash_full_taco.sh <gpu> <wait_file|-> <seq> [<seq> ...] &
#   wait_file: start only after this file exists ("-" = start now). Rerun-safe per stage.
# Env (defaults = the first TACO campaign): STAGE (parquet stage dir, taco_processed), OUT_SUB
# (flash_chord/outputs/<OUT_SUB>, taco_full), ART_SUB (artifacts/<ART_SUB>, flash-full-taco),
# EXTRA (extra Hydra overrides for training, space-separated, e.g. "reset=reference_high_quality_ik").
# CWD must be the TACO dataset dir: the parquet stores object paths relative to it and FlashCHORD's
# resolver only re-roots "/assets/" paths (flash_chord/src/flash_chord/assets/registry.py:43-80).
set -u
GPU=$1; WAIT=$2; shift 2
RG=/root/video_to_data/robotic_grounding
FC=$RG/flash_chord
PY=$FC/.venv/bin/python
TACO=/root/Data/human_motion_data/taco
NOTIFY=$HOME/meta/bin/notify.sh
export CUDA_VISIBLE_DEVICES=$GPU PYGLET_HEADLESS=1
cd "$TACO"
if [ "$WAIT" != "-" ]; then until [ -f "$WAIT" ]; do sleep 30; done; fi

for SEQ in "$@"; do
  PARQUET=$TACO/${STAGE:-taco_processed}/sequence_id=$SEQ/robot_name=sharpa_wave
  OUT=$FC/outputs/${OUT_SUB:-taco_full}/$SEQ
  ART=$RG/artifacts/${ART_SUB:-flash-full-taco}/$SEQ
  FINAL=$OUT/policy_250003456.safetensors
  mkdir -p "$OUT" "$ART"
  [ -f "$OUT/PROVENANCE.md" ] || echo "taco_full/$SEQ — $(date -u +%F) — FlashCHORD sharpa_flash_sac stage=${STAGE:-taco_processed} extra='${EXTRA:-}', $SEQ sharpa_wave, seed 42, GPU$GPU, commit $(git -C $RG rev-parse --short HEAD), W&B disabled — status: active" > "$OUT/PROVENANCE.md"

  if [ ! -f "$FINAL" ]; then
    $NOTIFY "🚀 FlashCHORD TACO full training started: $SEQ (GPU$GPU)" "ETA ~1.6 h. Log: $ART/train.log"
    start=$(date +%s)
    $PY -u $FC/scripts/train_flash_sac.py experiment=sharpa_flash_sac "task.parquet='${PARQUET}'" \
      reset.seed=42 logging.mode=disabled output_dir="$OUT" "hydra.run.dir=$OUT/hydra" ${EXTRA:-} \
      </dev/null > "$ART/train.log" 2>&1
    rc=$?
    echo "{\"train_rc\": $rc, \"train_wall_s\": $(( $(date +%s) - start )), \"gpu\": $GPU}" > "$ART/train_timing.json"
    if [ ! -f "$FINAL" ]; then
      $NOTIFY "❌ FlashCHORD TACO training FAILED: $SEQ (rc=$rc, GPU$GPU)" "$(tail -30 $ART/train.log)"
      continue
    fi
  fi

  if [ ! -f "$ART/metrics.json" ]; then
    $PY -u $FC/scripts/evaluate_policy.py "evaluation.checkpoint=${FINAL}" evaluation.world_count=4096 \
      "evaluation.metrics_output=$ART/metrics.json" "hydra.run.dir=$OUT/hydra_eval" </dev/null > "$ART/eval.log" 2>&1
  fi

  if [ ! -f "$ART/policy.mp4" ]; then
    CAM=$($PY - "$PARQUET" <<'EOF'
import glob, sys, numpy as np, pyarrow.parquet as pq
t = pq.read_table(glob.glob(sys.argv[1] + "/*.parquet")[0], columns=["object_root_position", "robot_right_wrist_position"])
target = (np.array(t.column("object_root_position")[0].as_py())[0] + np.array(t.column("robot_right_wrist_position")[0].as_py()).mean(0)) / 2
pos = target + np.array([0.55, -0.75, 0.45])  # same offset as the tissue_box framing
f = lambda v: "[" + ",".join(f"{x:.3f}" for x in v) + "]"
print(f(pos), f(target))
EOF
)
    read -r CAMPOS CAMTGT <<< "$CAM"
    timeout 1800 $PY -u $FC/scripts/view_policy.py "evaluation.checkpoint=${FINAL}" viewer.backend=mp4 \
      viewer.headless=true "viewer.output_path=$ART/policy.mp4" viewer.num_frames=1200 viewer.realtime=false \
      markers.enabled=false viewer.width=1280 viewer.height=720 "viewer.camera_position=$CAMPOS" \
      "viewer.camera_target=$CAMTGT" "hydra.run.dir=$OUT/hydra_view" </dev/null > "$ART/view.log" 2>&1
  fi
  sed -i 's/status: active/status: locked/' "$OUT/PROVENANCE.md"
  $NOTIFY "✅ FlashCHORD TACO done: $SEQ (GPU$GPU)" "$(cat $ART/train_timing.json 2>/dev/null)
$(head -c 2500 $ART/metrics.json 2>/dev/null)"
  touch "$ART/.done"
done

#!/bin/bash
# Chained per-checkpoint eval (rule chained-eval): watch RUN_DIR for new model_*.pt,
# eval each with video, publish rollout mp4s + rollouts.json for the review page.
# GPU: cuda:0 only — Peijun's standing instruction (2026-08-26): avoid GPU1 entirely.
# Exits when DONE_MARKER exists and every checkpoint has been processed.
# Usage: chained_eval.sh RUN_DIR SEQ_DIR OUT_DIR DONE_MARKER TRAIN_LOG
set -u
RUN_DIR=$1; SEQ_DIR=$2; OUT_DIR=$3; DONE=$4; TRAIN_LOG=$5
cd "$(dirname "$(readlink -f "$0")")/../.."
mkdir -p "$OUT_DIR"
export OMNI_KIT_ACCEPT_EULA=YES VK_DRIVER_FILES=/etc/vulkan/icd.d/nvidia_icd.json \
       VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json \
       HUMAN_MOTION_DATA_DIR=/root/Data/human_motion_data
unset CUDA_VISIBLE_DEVICES

update_json() {  # rebuild rollouts.json from processed videos + train-log reward series
  python3 - "$OUT_DIR" "$TRAIN_LOG" <<'EOF'
import json, re, sys
from pathlib import Path
out, log = Path(sys.argv[1]), Path(sys.argv[2])
entries = []
for v in sorted(out.glob("rollout_iter*.mp4")):
    it = int(re.search(r"iter(\d+)", v.name).group(1))
    entries.append({"iter": it, "video": v.name})
rewards = []
if log.is_file():
    for m in re.finditer(r"iteration (\d+)/\d+.*?Mean reward:\s*([-\d.]+)", log.read_text(errors="ignore"), re.S):
        rewards.append([int(m.group(1)), float(m.group(2))])
(out / "rollouts.json").write_text(json.dumps(
    {"rollouts": entries, "reward_series": rewards[-2000:]}, indent=1))
EOF
}

while :; do
  FINAL=$(ls -v "$RUN_DIR"/model_*.pt 2>/dev/null | tail -1 | sed 's/.*model_//;s/.pt//')
  for ckpt in $(ls -v "$RUN_DIR"/model_*.pt 2>/dev/null); do
    it=$(basename "$ckpt" .pt); it=${it#model_}
    out_mp4="$OUT_DIR/rollout_iter$(printf '%05d' "$it").mp4"
    [ -f "$out_mp4" ] && continue
    [ "$it" -eq 0 ] && continue  # skip the pre-training checkpoint
    # EVAL_STRIDE: only eval every Nth iteration (final checkpoint always included)
    if [ "${EVAL_STRIDE:-1}" -gt 1 ] && [ "$it" != "$FINAL" ] && [ $((it % EVAL_STRIDE)) -ne 0 ]; then continue; fi
    echo "[chained-eval] evaluating model_$it"
    tmp=$(mktemp -d)
    timeout --signal=KILL 1200 .venv/bin/python scripts/rsl_rl/eval.py --headless \
      --device cuda:0 --task Sharpa-V2D-v0 --motion_file "$SEQ_DIR" --num_envs 8 \
      --checkpoint "$ckpt" --max_steps 300 --video --video_length 300 \
      </dev/null > "$OUT_DIR/eval_iter${it}.log" 2>&1
    mp4=$(find "$RUN_DIR" "$tmp" -name '*.mp4' -newer "$ckpt" 2>/dev/null | head -1)
    if [ -n "$mp4" ]; then mv "$mp4" "$out_mp4"; echo "[chained-eval] wrote $out_mp4"; fi
    rm -rf "$tmp"
    update_json
  done
  update_json
  if [ -f "$DONE" ]; then
    # one final sweep already happened above; exit when nothing new remains
    remaining=$(ls -v "$RUN_DIR"/model_*.pt 2>/dev/null | while read -r c; do
      i=$(basename "$c" .pt); i=${i#model_}; [ "$i" -eq 0 ] && continue
      [ -f "$OUT_DIR/rollout_iter$(printf '%05d' "$i").mp4" ] || echo "$c"; done | wc -l)
    [ "$remaining" -eq 0 ] && break
  fi
  sleep 60
done
echo "[chained-eval] done"

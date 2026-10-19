#!/bin/bash
# Full FlashCHORD FlashSAC training on the official example (tissue_box_simple, Sharpa),
# launched after the timing comparison finishes; then README step-5 evaluation (4096 worlds)
# and a headless policy mp4. GPU0 only. Peijun 2026-09-28: "after the performance is done,
# launch a full FlashChord training on the official example".
# Usage: setsid nohup bash agents/scripts/flash_full_tissue.sh &   (safe to rerun: skips done stages)
set -u
RG=/root/video_to_data/robotic_grounding
FC=$RG/flash_chord
TIMING=$RG/artifacts/flash-vs-ppo-timing
OUT=$FC/outputs/sharpa_tissue_full           # README "Train" layout; gitignored
ART=$RG/artifacts/flash-full-tissue
PARQUET=$RG/source/robotic_grounding/robotic_grounding/assets/human_motion_data/ego_recon/processed/sequence_id=tissue_box_simple/robot_name=sharpa_wave
FINAL=$OUT/policy_250003456.safetensors      # README table: terminal actor checkpoint
PAGE=https://peijun-reports.axon.tools/files/video_to_data/robotic_grounding/artifacts/flash-full-tissue/
NOTIFY=$HOME/meta/bin/notify.sh
export CUDA_VISIBLE_DEVICES=0 PYGLET_HEADLESS=1
mkdir -p "$OUT" "$ART"
cd "$FC"

# Wait for the timing orchestrator (artifact first; also require its process gone).
until [ -f "$TIMING/.done" ] && ! pgrep -f "^python3 agents/scripts/[t]iming_flash_vs_ppo" >/dev/null; do sleep 30; done

[ -f "$OUT/PROVENANCE.md" ] || echo "sharpa_tissue_full — $(date -u +%F) — FlashCHORD sharpa_flash_sac, tissue_box_simple sharpa_wave, seed 42, GPU0, commit $(git -C $RG rev-parse --short HEAD), W&B disabled — status: active" > "$OUT/PROVENANCE.md"

if [ ! -f "$FINAL" ]; then
  $NOTIFY "🚀 FlashCHORD full training started (tissue_box, GPU0)" "ETA ~1.5 h. Log: $ART/train.log"
  start=$(date +%s)
  .venv/bin/python -u scripts/train_flash_sac.py experiment=sharpa_flash_sac \
    "task.parquet='${PARQUET}'" reset.seed=42 logging.mode=disabled output_dir="$OUT" \
    </dev/null > "$ART/train.log" 2>&1
  rc=$?
  echo "{\"train_rc\": $rc, \"train_wall_s\": $(( $(date +%s) - start ))}" > "$ART/train_timing.json"
  if [ ! -f "$FINAL" ]; then
    $NOTIFY "❌ FlashCHORD full training FAILED (rc=$rc)" "$(tail -30 $ART/train.log)"
    exit 1
  fi
fi

# README step 5: evaluation over 4096 worlds -> metrics JSON (MPPE etc.).
if [ ! -f "$ART/metrics.json" ]; then
  .venv/bin/python -u scripts/evaluate_policy.py "evaluation.checkpoint=${FINAL}" \
    evaluation.world_count=4096 "evaluation.metrics_output=$ART/metrics.json" </dev/null > "$ART/eval.log" 2>&1
fi

# README step 6 as a headless mp4 instead of Viser (same camera as the sanity replays).
if [ ! -f "$ART/policy.mp4" ]; then
  timeout 1800 .venv/bin/python -u scripts/view_policy.py "evaluation.checkpoint=${FINAL}" \
    viewer.backend=mp4 viewer.headless=true "viewer.output_path=$ART/policy.mp4" viewer.num_frames=1200 \
    viewer.realtime=false markers.enabled=false viewer.width=1280 viewer.height=720 \
    "viewer.camera_position=[0.55,-0.75,1.6]" "viewer.camera_target=[0.0,0.0,1.15]" </dev/null > "$ART/view.log" 2>&1
fi

cat > "$ART/index.html" <<EOF
<!doctype html><html><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>FlashCHORD tissue full run</title><style>:root{--bg:#fff;--fg:#1d1d1f}@media (prefers-color-scheme:dark){:root{--bg:#161617;--fg:#f5f5f7}}
body{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;max-width:1000px;margin:0 auto;padding:24px 16px}video{width:100%}pre{white-space:pre-wrap;font-size:13px}</style></head><body>
<h1>FlashCHORD FlashSAC — full training, tissue_box_simple</h1>
<p>Checkpoint: $FINAL</p><h2>Timing</h2><pre>$(cat $ART/train_timing.json 2>/dev/null)</pre>
<h2>Evaluation (4096 worlds, deterministic)</h2><pre>$(cat $ART/metrics.json 2>/dev/null || tail -20 $ART/eval.log)</pre>
<h2>Policy rollout</h2><video controls src="policy.mp4"></video>
<p>Reference for comparison: flash_chord/docs/media/sharpa_policy.gif (NVIDIA's seed-42 release actor).</p></body></html>
EOF
sed -i 's/status: active/status: locked/' "$OUT/PROVENANCE.md"
$NOTIFY "✅ FlashCHORD full training + eval done (tissue_box)" "$(cat $ART/train_timing.json 2>/dev/null)
$(cat $ART/metrics.json 2>/dev/null | head -60)

Page: $PAGE"
touch "$ART/.done"

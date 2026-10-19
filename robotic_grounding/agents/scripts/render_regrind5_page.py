#!/usr/bin/env python3
"""A/B page for flash-regrind-ik: FlashCHORD with V2D IK (A) vs regrind high-quality IK (B), 5 TACO episodes.

Data layer only (no re-runs): artifacts/flash-regrind-ik/{A_v2d_ik,B_regrind_ik}/<seq>/{train_timing.json,metrics.json,
policy.mp4}, replays/*.mp4, support_clearance.json, and regrind's report-4 data.json (context: separately trained
single-hand Isaac Lab PPO policies, success per hand over 4 envs at level c; a different metric).
Writes index.html + summary.txt. --wait: block until both overnight queues have exited, then render and email.
"""

from __future__ import annotations

import html
import json
import subprocess
import sys
import time
from pathlib import Path

RG = Path("/root/video_to_data/robotic_grounding")
ART = RG / "artifacts/flash-regrind-ik"
REPORT = Path("/root/regrind/artifacts/reports-1001/bimanual-tolerance/data.json")
PAGE_URL = "https://peijun-reports.axon.tools/files/video_to_data/robotic_grounding/artifacts/flash-regrind-ik/index.html"
EPISODES = [  # priority first
    ("brush_brush_toy_038", "taco_brush__brush__toy_20230919_038"),
    ("scrape-off_spoon_plate_031", "taco_scrape_off__spoon__plate_20230926_031"),
    ("brush_eraser_kettle_106", "taco_brush__eraser__kettle_20231103_106"),
    ("brush_brush_pan_044", "taco_brush__brush__pan_20230926_044"),
    ("brush_brush_helmet_003", "taco_brush__brush__helmet_20231027_003"),
]
VARIANTS = [("A_v2d_ik", "A · original FlashCHORD (V2D IK, default reset)"),
            ("B_regrind_ik", "B · regrind IK + reset=reference_high_quality_ik")]
COLS = [("mppe_cm", "MPPE (cm) ↓", "{:.2f}"), ("chord_sr", "CHORD SR ↑", "{:.3f}"), ("maniptrans_sr", "ManipTrans SR ↑", "{:.3f}"),
        ("add_auc", "ADD-AUC ↑", "{:.3f}"), ("object_position_error_m", "obj pos err (mm) ↓", "mm"),
        ("object_orientation_error_deg", "obj rot err (°) ↓", "{:.1f}")]


def load(p: Path):
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def queues_alive() -> bool:
    return subprocess.run(["pgrep", "-f", "^bash agents/scripts/[q]ueue_regrind5"], capture_output=True).returncode == 0 \
        or subprocess.run(["pgrep", "-f", "^bash agents/scripts/[f]lash_full_taco"], capture_output=True).returncode == 0


if "--wait" in sys.argv:
    time.sleep(120)
    while queues_alive():
        time.sleep(120)

report = load(REPORT) or {"episodes": []}
regrind_c = {e["key"]: e.get("c", {}).get("s0", {}) for e in report["episodes"]}
clear = load(ART / "support_clearance.json") or {}

rows, summary = [], ["FlashCHORD A (V2D IK) vs B (regrind IK + high-quality-IK reset), seed 42, 250M steps each"]
for key, seq in EPISODES:
    for v, label in VARIANTS:
        d = ART / v / seq
        m = (load(d / "metrics.json") or {}).get("metrics", {})
        t = (load(d / "train_timing.json") or {}).get("train_wall_s")
        state = "done" if (d / ".done").exists() else ("training" if (d / "train.log").exists() else "queued")
        cells = []
        for k, _, fmt in COLS:
            val = m.get(k)
            cells.append("–" if val is None else (f"{val * 1e3:.1f}" if fmt == "mm" else fmt.format(val)))
        fail = m.get("termination_cause_fraction", {})
        rows.append(f"<tr><td>{html.escape(key)}</td><td>{v[0]}</td><td>{state}</td><td class=n>{'–' if t is None else f'{t / 3600:.2f}'}</td>"
                    + "".join(f"<td class=n>{c}</td>" for c in cells)
                    + f"<td class=n>{fail.get('object', float('nan')) * 100:.1f} / {fail.get('wrist', float('nan')) * 100:.1f}</td></tr>")
        summary.append(f"- {key} {v[0]}: {state}, train {'-' if t is None else f'{t / 3600:.2f} h'}, "
                       + ", ".join(f"{k}={m[k]:.4g}" for k, _, _ in COLS if k in m))
    rc = regrind_c.get(key, {})
    if rc:
        rows.append(f"<tr class=ctx><td>{html.escape(key)}</td><td colspan=3>context: regrind report 4, level c (two single-hand "
                    f"Isaac Lab PPO policies, 4 envs)</td><td colspan={len(COLS) + 1}>success right {rc.get('active_success')}/4, "
                    f"left {rc.get('passive_success')}/4, both {rc.get('both_success')}/4 (demo end + DP@1s ≤ 30 mm; not comparable "
                    "to the columns above)</td></tr>")

media = []
for key, seq in EPISODES:
    c = clear.get(key, {})
    note = ", ".join(f"{s}: A {c.get(f'hand_{s}_A_v2d_ik', {}).get('frame0_clearance_mm', float('nan')):+.1f} mm, "
                     f"B {c.get(f'hand_{s}_B_regrind_ik', {}).get('frame0_clearance_mm', float('nan')):+.1f} mm" for s in ("right", "left"))
    figs = []
    for v, label in VARIANTS:
        rp = ART / "replays" / f"{seq}_{v}_physical.mp4"
        if rp.exists():
            figs.append(f'<figure><video controls src="replays/{rp.name}"></video><figcaption>{v[0]} reference, zero-residual physical replay</figcaption></figure>')
    for v, label in VARIANTS:
        if (ART / v / seq / "policy.mp4").exists():
            figs.append(f'<figure><video controls src="{v}/{seq}/policy.mp4"></video><figcaption>{html.escape(label)} — final policy</figcaption></figure>')
    media.append(f"<h3>{html.escape(key)}</h3><p class=mut>hand clearance above V2D support at frame 0 ({note}; negative = inside the "
                 "table; hand–support collision is off in training)</p><div class=row>" + "".join(figs) + "</div>")

page = f"""<!doctype html><html><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>FlashCHORD IK A/B</title><style>
:root{{--bg:#fff;--fg:#1d1d1f;--mut:#6e6e73;--line:#d2d2d7;--ctx:#f5f5f7}}
@media (prefers-color-scheme:dark){{:root{{--bg:#161617;--fg:#f5f5f7;--mut:#a1a1a6;--line:#3a3a3c;--ctx:#1f1f21}}}}
body{{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;max-width:1200px;margin:0 auto;padding:24px 16px}}
table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{border-bottom:1px solid var(--line);padding:5px 7px;text-align:left}}
.n{{text-align:right;font-variant-numeric:tabular-nums}}tr.ctx td{{background:var(--ctx);color:var(--mut)}}.wrap{{overflow-x:auto}}
.row{{display:flex;flex-wrap:wrap;gap:12px}}figure{{margin:0;flex:1 1 260px}}video{{width:100%}}figcaption,.mut{{color:var(--mut);font-size:13px}}</style></head><body>
<h1>FlashCHORD on regrind's report-4 episodes — V2D IK (A) vs regrind high-quality IK (B)</h1>
<p class=mut>Both: <code>sharpa_flash_sac</code>, seed 42, 250,003,456 env steps, same objects/MANO/contacts/support (V2D), 4096-world
deterministic eval (README step 5). B differs only in the robot reference (regrind <code>mink_init_tol0_dextip_ladder_halfspace_1001</code>,
transplanted; FK check ≤ 0.001 mm) and <code>reset=reference_high_quality_ik</code> (finger factor [0.75, 0.93], 10 settling steps).
Two knobs move together in B (IK + reset). Rows refresh as runs land.</p>
<div class=wrap><table><thead><tr><th>episode</th><th>var</th><th>state</th><th class=n>train (h)</th>
{''.join(f'<th class=n>{html.escape(h)}</th>' for _, h, _ in COLS)}<th class=n>fail % obj / wrist</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></div>
<h2>Videos</h2>{''.join(media)}
<p class=mut>Data: */*/metrics.json, train_timing.json, support_clearance.json; pad agents/pads/flash-regrind-ik.md.</p></body></html>"""
(ART / "index.html").write_text(page)
(ART / "summary.txt").write_text("\n".join(summary) + "\n")
print("\n".join(summary))
if "--wait" in sys.argv:
    subprocess.run([str(Path.home() / "meta/bin/notify.sh"), "✅ FlashCHORD IK A/B: all queued runs finished",
                    "\n".join(summary) + f"\n\nPage: {PAGE_URL}"])

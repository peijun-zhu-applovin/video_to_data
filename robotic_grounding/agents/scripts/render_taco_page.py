#!/usr/bin/env python3
"""Aggregate the four FlashCHORD TACO full runs into artifacts/flash-full-taco/index.html + summary.txt.

Reads each sequence's train_timing.json and metrics.json (data layer written by flash_full_taco.sh);
never re-runs anything. Numeric metrics are flattened (nested keys joined with '.').
Usage: python render_taco_page.py [--wait]   (--wait: block until all four .done files exist, then email)
"""

from __future__ import annotations

import html
import json
import subprocess
import sys
import time
from pathlib import Path

RG = Path("/root/video_to_data/robotic_grounding")
ART = RG / "artifacts/flash-full-taco"
SEQS = ["taco_empty__cup__bowl_20231006_280", "taco_cut__knife__bowl_20231020_254",
        "taco_dust__brush__bowl_20230927_028", "taco_skim_off__spoon__pan_20230926_011"]
PAGE_URL = "https://peijun-reports.axon.tools/files/video_to_data/robotic_grounding/artifacts/flash-full-taco/index.html"
PPO_VIDEO = "../repro-taco/rollouts_full/rollout_iter19999.mp4"  # Isaac Lab PPO, 20000 iters, 50.6 h
TISSUE = RG / "artifacts/flash-full-tissue"


def flatten(d, prefix=""):
    out = {}
    for k, v in (d or {}).items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out[key] = v
    return out


def load(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


if "--wait" in sys.argv:
    while not all((ART / s / ".done").exists() for s in SEQS):
        time.sleep(60)

rows = []
for name, d in [("ego_recon tissue_box_simple (baseline: official example)", TISSUE)] + [(s, ART / s) for s in SEQS]:
    timing, metrics = load(d / "train_timing.json") or {}, flatten(load(d / "metrics.json"))
    rows.append((name, d, timing, metrics))

keys = [k for k in ("metrics.mppe_cm",) if any(k in r[3] for r in rows)]
extra = sorted({k for r in rows for k in r[3] if k.startswith("metrics.") and k not in keys})[:8]
cols = keys + extra
head = "".join(f"<th class=n>{html.escape(c.removeprefix('metrics.'))}</th>" for c in cols)
body = []
for name, d, timing, m in rows:
    t = timing.get("train_wall_s")
    cells = "".join(f"<td class=n>{m[c]:.4g}</td>" if c in m else "<td class=n>–</td>" for c in cols)
    body.append(f"<tr><td>{html.escape(name)}</td><td class=n>{timing.get('gpu', 0)}</td>"
                f"<td class=n>{'–' if t is None else f'{t / 3600:.2f}'}</td>{cells}</tr>")

videos = []
for s in SEQS:
    extra_vid = ""
    if "empty__cup__bowl" in s:
        extra_vid = (f'<figure><video controls src="{PPO_VIDEO}"></video><figcaption>Isaac Lab PPO, same sequence, '
                     "iter 19999 (50.6 h, Aug run; different simulator)</figcaption></figure>")
    videos.append(f'<h3>{html.escape(s)}</h3><div class=row><figure><video controls src="{s}/policy.mp4"></video>'
                  f"<figcaption>FlashCHORD FlashSAC, final actor (250M steps)</figcaption></figure>{extra_vid}</div>")

page = f"""<!doctype html><html><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>FlashCHORD TACO runs</title><style>
:root{{--bg:#fff;--fg:#1d1d1f;--mut:#6e6e73;--line:#d2d2d7}}
@media (prefers-color-scheme:dark){{:root{{--bg:#161617;--fg:#f5f5f7;--mut:#a1a1a6;--line:#3a3a3c}}}}
body{{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;max-width:1100px;margin:0 auto;padding:24px 16px}}
table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left}}
.n{{text-align:right;font-variant-numeric:tabular-nums}}.wrap{{overflow-x:auto}}.row{{display:flex;flex-wrap:wrap;gap:16px}}
figure{{margin:0;flex:1 1 320px}}video{{width:100%}}figcaption,.mut{{color:var(--mut);font-size:13px}}</style></head><body>
<h1>FlashCHORD FlashSAC — full runs on TACO (Sharpa)</h1>
<p class=mut>Unchanged <code>sharpa_flash_sac</code> recipe (tuned on single-object ego video), seed 42, 250,003,456 env steps each;
eval = README step 5 (4096 worlds, deterministic). Lower MPPE (cm) is better. Known issue: the retargeted reference for
empty__cup__bowl and skim_off__spoon__pan fails the hand-penetration check (taco_quality.csv); the thumb starts inside the bowl wall.</p>
<div class=wrap><table><thead><tr><th>sequence</th><th class=n>GPU</th><th class=n>train (h)</th>{head}</tr></thead>
<tbody>{''.join(body)}</tbody></table></div>
{''.join(videos)}
<p class=mut>Data: */train_timing.json, */metrics.json; logs */train.log, */eval.log.</p></body></html>"""
(ART / "index.html").write_text(page)

summary = ["FlashCHORD TACO full runs (sharpa_flash_sac unchanged, seed 42)"]
for name, d, timing, m in rows:
    t = timing.get("train_wall_s")
    summary.append(f"- {name}: train {'-' if t is None else f'{t / 3600:.2f} h'} (GPU{timing.get('gpu', 0)}), "
                   + ", ".join(f"{c.removeprefix('metrics.')}={m[c]:.4g}" for c in cols if c in m))
(ART / "summary.txt").write_text("\n".join(summary) + "\n")
print("\n".join(summary))
if "--wait" in sys.argv:
    subprocess.run([str(Path.home() / "meta/bin/notify.sh"), "✅ FlashCHORD TACO: all 4 full runs done",
                    "\n".join(summary) + f"\n\nPage: {PAGE_URL}"])

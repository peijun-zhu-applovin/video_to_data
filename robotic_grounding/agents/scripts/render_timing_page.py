#!/usr/bin/env python3
"""Render artifacts/flash-vs-ppo-timing/{index.html,summary.txt} from results.json (no re-running)."""

from __future__ import annotations

import html
import json
from pathlib import Path

ART = Path("/root/video_to_data/robotic_grounding/artifacts/flash-vs-ppo-timing")
R = json.loads((ART / "results.json").read_text())

# Context row: earlier full PPO run on a DIFFERENT sequence (taco empty_cup_bowl), same box/GPU0.
TACO_FULL_H = 20000 * 9.11 / 3600  # artifacts/repro-taco/curve/reward.json sec_per_iter=9.11, 20000 iters


def h(seconds: float | None) -> str:
    return "–" if seconds is None else f"{seconds / 3600:.2f}"


def m(seconds: float | None) -> str:
    return "–" if seconds is None else f"{seconds / 60:.1f}"


rows = []
for key, label, unit, total in (
    ("ppo", "Isaac Lab PPO (baseline) · rsl_rl · PhysX", "iter", 20000),
    ("flash", "FlashCHORD FlashSAC · JAX · Newton/MuJoCo-Warp", "env step", 250_003_456),
):
    run = R.get(key, {})
    ex = run.get("extrapolation") or {}
    gpu = run.get("gpu", {})
    rows.append({
        "label": label, "total": total, "unit": unit,
        "t2": ex.get("t_2pct_s"), "t10": ex.get("t_10pct_s"),
        "naive2": ex.get("naive_100_from_2pct_s"), "naive10": ex.get("naive_100_from_10pct_s"),
        "steady": ex.get("steady_100_s"), "rate": ex.get("rate_per_s"),
        "peak": gpu.get("peak_mem_mib"), "util": gpu.get("mean_util_pct"),
        "env_steps_total": total * (24 * 4096 if key == "ppo" else 1),
    })

base = rows[0]["steady"]
ratio = (base / rows[1]["steady"]) if base and rows[1]["steady"] else None

tr = []
for r in rows:
    peak = "–" if r["peak"] is None else f"{r['peak'] / 1024:.1f}"
    rate = "–" if r["rate"] is None else (f"{r['rate']:.3f} it/s" if r["unit"] == "iter" else f"{r['rate']:,.0f} steps/s")
    tr.append(
        f"<tr><td>{html.escape(r['label'])}</td><td class=n>{r['env_steps_total'] / 1e6:,.0f}</td>"
        f"<td class=n>{m(r['t2'])}</td><td class=n>{m(r['t10'])}</td><td class=n>{rate}</td>"
        f"<td class=n>{h(r['naive2'])}</td><td class=n>{h(r['naive10'])}</td><td class=n><b>{h(r['steady'])}</b></td>"
        f"<td class=n>{peak}</td>"
        f"<td class=n>{'–' if r['util'] is None else r['util']}</td></tr>"
    )
tr.append(
    f"<tr class=ctx><td>context: Isaac Lab PPO full run, taco empty_cup_bowl (different sequence)</td>"
    f"<td class=n>{20000 * 24 * 4096 / 1e6:,.0f}</td><td class=n>–</td><td class=n>–</td><td class=n>0.110 it/s</td>"
    f"<td class=n>–</td><td class=n>–</td><td class=n>{TACO_FULL_H:.2f} (measured)</td><td class=n>–</td><td class=n>–</td></tr>"
)

fl = R.get("flash", {})
summary = [
    "FlashSAC vs Isaac Lab PPO — wall-clock to 100% (tissue_box_simple, Sharpa, GPU0 RTX PRO 6000)",
    f"commit {R.get('commit')}  started {R.get('started')}  finished {R.get('finished')}",
    "",
]
for r in rows:
    summary.append(
        f"{r['label']}: 2% at {m(r['t2'])} min, 10% at {m(r['t10'])} min -> 100% est {h(r['steady'])} h "
        f"(naive x50 from 2%: {h(r['naive2'])} h, x10 from 10%: {h(r['naive10'])} h); "
        f"peak GPU mem {'-' if r['peak'] is None else round(r['peak'] / 1024, 1)} GiB, util {r['util']}%"
    )
if ratio:
    summary.append(f"PPO / FlashSAC wall-clock ratio at 100% (steady-rate estimate): {ratio:.1f}x")
summary.append(f"context: earlier full PPO on taco empty_cup_bowl measured {TACO_FULL_H:.1f} h (different sequence)")
summary.append(f"FlashSAC JAX compile: {fl.get('events', {}).get('compile_s')} s")
(ART / "summary.txt").write_text("\n".join(summary) + "\n")

svl_html = ""
_svl = ART / "sim_vs_learn.json"
if _svl.exists():
    S = json.loads(_svl.read_text())
    def _row(name, d):
        return (f"<tr><td>{name}</td><td class=n>{d['sim_ms_per_step']:.1f}</td><td class=n>{d['learn_ms_per_step']:.1f}</td>"
                f"<td class=n>{d['total_ms_per_step']:.1f}</td><td class=n><b>{d['sim_share_pct']:.1f}</b></td><td>{html.escape(d['window'])}</td></tr>")
    ppo_h = 1_966_080_000 / 4096 * S["ppo"]["total_ms_per_step"] / 3.6e6
    ppo_newton_h = 1_966_080_000 / 4096 * (S["flash"]["sim_ms_per_step"] + S["ppo"]["learn_ms_per_step"]) / 3.6e6
    svl_html = (
        "<h2>Where the time goes: simulation vs learning</h2>"
        "<p class=mut>Unit: ms per env step of 4096 worlds (both lanes: 4096 worlds, 20 Hz control × 5 substeps = 100 Hz physics). "
        "PPO split from rsl_rl's own <code>collection</code>/<code>learning</code> timers (collection includes policy inference). "
        "FlashSAC split from GPU-synchronized hooks around <code>env.step</code> and the compiled learner call "
        f"(profiled on idle GPU1; unprofiled overlapped step = {S['flash']['unprofiled_ms_per_step']:.1f} ms).</p>"
        "<div class=wrap><table><thead><tr><th>run</th><th class=n>sim (ms)</th><th class=n>learn (ms)</th><th class=n>total (ms)</th>"
        "<th class=n>sim share (%)</th><th>window</th></tr></thead><tbody>"
        + _row("Isaac Lab PPO (baseline)", S["ppo"]) + _row("FlashCHORD FlashSAC (8 updates/step)", S["flash"]) +
        "</tbody></table></div>"
        f"<p>Simulation alone: MuJoCo-Warp is <b>{S['sim_speedup_newton_vs_isaaclab']:.1f}×</b> faster per step than Isaac Lab. "
        f"Decomposing the ~15×: PPO at its measured step cost over 1,966M steps ≈ {ppo_h:.1f} h; the same PPO with MuJoCo-Warp's sim cost "
        f"would be ≈ {ppo_newton_h:.1f} h (simulator ≈ {ppo_h / ppo_newton_h:.1f}×); FlashSAC's 8× fewer steps more than pay for its "
        "10× larger learning cost per step (remaining ≈ 4×).</p>")
page = f"""<!doctype html><html><head><meta charset=utf-8><title>FlashSAC vs PPO timing</title>
<meta name=viewport content="width=device-width,initial-scale=1">
<style>
:root{{--bg:#fff;--fg:#1d1d1f;--mut:#6e6e73;--line:#d2d2d7;--ctx:#f5f5f7}}
@media (prefers-color-scheme:dark){{:root{{--bg:#161617;--fg:#f5f5f7;--mut:#a1a1a6;--line:#3a3a3c;--ctx:#1f1f21}}}}
body{{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;margin:0 auto;max-width:1100px;padding:24px 16px}}
table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left}}
td.n,th.n{{text-align:right;font-variant-numeric:tabular-nums}}tr.ctx td{{background:var(--ctx);color:var(--mut)}}
.mut{{color:var(--mut)}}.wrap{{overflow-x:auto}}video{{width:100%;max-width:540px}}
</style></head><body>
<h1>FlashSAC vs Isaac Lab PPO — time to full training</h1>
<p class=mut>tissue_box_simple · Sharpa floating hands · 1× RTX PRO 6000 (GPU0) · commit {R.get('commit')} ·
{R.get('started')} → {R.get('finished')}</p>
<p>Both runs launched with their <b>full</b> configs (schedules identical to a real run). Budgets per request:
FlashSAC stopped at <b>10%</b> (2% mark read from the same run); PPO run to <b>2%</b> only (400 of 20,000 iters).
<b>Steady estimate</b> = elapsed time at the last mark + remaining budget ÷ the steady rate (FlashSAC: 2%→10%;
PPO: iter 100→400), so startup is counted once, not extrapolated. Naive columns scale elapsed time linearly.
{f'<b>PPO takes {ratio:.1f}× longer at 100%.</b>' if ratio else ''}</p>
<div class=wrap><table><thead><tr><th>run</th><th class=n>env steps at 100% (M)</th><th class=n>2% (min)</th>
<th class=n>10% (min)</th><th class=n>rate 2→10%</th><th class=n>100% naive ×50 (h)</th><th class=n>100% naive ×10 (h)</th>
<th class=n>100% steady est (h)</th><th class=n>peak GPU mem (GiB)</th><th class=n>mean util (%)</th></tr></thead>
<tbody>{''.join(tr)}</tbody></table></div>
{svl_html}
<h2>Caveats</h2><ul>
<li>Different simulators (PhysX vs MuJoCo-Warp) and different algorithms: this compares the two released lanes end to end, not the algorithm alone.</li>
<li>Only early curriculum stages are measured (FlashSAC 10%: VOC 1.0→0.75 at 19.7M; PPO 2%: VOC 1.0, first change at iter 2000). Later stages (VOC 0, frame-0 resets) may change per-step cost.</li>
<li>Time only — no policy quality was compared here. PPO ran without <code>--video</code>; FlashSAC with W&amp;B disabled.</li>
<li>PPO iteration = 24 steps × 4096 envs = 98,304 env steps; FlashSAC does 8 gradient updates per 4096-world collection.</li>
</ul>
<h2>Sanity replays (FlashCHORD, Newton)</h2>
<p class=mut>Both are <b>physics</b> replays of the retargeted reference with zero policy residual
(hands tracked by the controller; tissue box simulated). They only show that assets, scene and sim load and run.</p>
<div style="display:flex;flex-wrap:wrap;gap:16px">
<figure style="margin:0;flex:1 1 320px"><video controls src="replay_kinematic.mp4"></video>
<figcaption><b>Left — stiff object assist.</b> <code>replay=kinematic</code> only: VOC spring k=10000/d=200 pins the box
to its reference pose; contacts ON, normal hand gains. <i>Not</i> the documented kinematic mode (that also needs
<code>action=residual_hand_pose_stiff collision=kinematic embodiment=sharpa_hands_stiff</code>) and not the
no-physics "Show kinematic reference" playback.</figcaption></figure>
<figure style="margin:0;flex:1 1 320px"><video controls src="replay_kinematic_mode.mp4"></video>
<figcaption><b>Middle — documented kinematic mode.</b> <code>replay=kinematic action=residual_hand_pose_stiff
collision=kinematic embodiment=sharpa_hands_stiff</code> (<code>scripts/debug/replay.py:11-12</code>): contacts OFF,
stiff gains — clean check of how faithfully the reference is tracked; still physics-stepped.</figcaption></figure>
<figure style="margin:0;flex:1 1 320px"><video controls src="replay_physical.mp4"></video>
<figcaption><b>Right — default physical replay.</b> contacts ON, soft VOC k=400/d=40: the box moves more
through real hand contact.</figcaption></figure></div>
<p class=mut>Data: results.json · logs: flash_10pct.log, ppo_10pct.log · pytest.log (637 passed, 55 skipped)</p>
</body></html>"""
(ART / "index.html").write_text(page)
print((ART / "summary.txt").read_text())

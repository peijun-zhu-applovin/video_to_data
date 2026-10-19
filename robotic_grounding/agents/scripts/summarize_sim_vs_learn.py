#!/usr/bin/env python3
"""Sim vs learning split, both lanes in one unit: milliseconds per 4096-world env step.

PPO  : rsl_rl per-iteration `collection` (policy inference + env.step, 24 steps) and
       `learning` (PPO update) from ppo_2pct.log, steady window iters >= 100.
Flash: profile_flash_sim_vs_learn.py hooks, collections after replay warm-up (+10 compile/settle).
Writes sim_vs_learn.json (data) and prints a text summary.
"""

import json
import re
import statistics as st
from pathlib import Path

ART = Path("/root/video_to_data/robotic_grounding/artifacts/flash-vs-ppo-timing")
res = json.loads((ART / "results.json").read_text())

col, lrn = [], []
for line in (ART / "ppo_2pct.log").open():
    m = re.search(r"collection: ([\d.]+)s, learning ([\d.]+)s", line)
    if m:
        col.append(float(m.group(1)))
        lrn.append(float(m.group(2)))
col, lrn = col[100:], lrn[100:]
ppo = {"sim_ms_per_step": 1000 * st.mean(col) / 24, "learn_ms_per_step": 1000 * st.mean(lrn) / 24,
       "window": f"iters 100-{100 + len(col)}", "note": "sim includes policy inference (rsl_rl 'collection')"}

prof = json.loads((ART / "flash_profile.json").read_text())
minimum = 409_600  # sharpa_flash_sac replay.minimum_size
idx = [i for i, s in enumerate(prof["replay_size"]) if s > minimum][10:]
sim = [prof["sim"][i + 1] for i in idx if i + 1 < len(prof["sim"])]  # step i+1 follows learn call i
learn = [prof["learn"][i] for i in idx]
flash = {"sim_ms_per_step": 1000 * st.mean(sim), "learn_ms_per_step": 1000 * st.mean(learn),
         "window": f"{len(learn)} collections with updates", "updates_per_step": 8,
         "unprofiled_ms_per_step": 1000 * 4096 / res["flash"]["extrapolation"]["rate_per_s"],
         "note": "learn = replay insert + 8 SAC updates + next action (one XLA call)"}
for d in (ppo, flash):
    d["total_ms_per_step"] = d["sim_ms_per_step"] + d["learn_ms_per_step"]
    d["sim_share_pct"] = 100 * d["sim_ms_per_step"] / d["total_ms_per_step"]
out = {"unit": "ms per 4096-world env step", "ppo": ppo, "flash": flash,
       "sim_speedup_newton_vs_isaaclab": ppo["sim_ms_per_step"] / flash["sim_ms_per_step"]}
(ART / "sim_vs_learn.json").write_text(json.dumps(out, indent=1))

print("Sim vs learning, ms per 4096-world env step (both lanes use 4096 worlds)")
for name, d in (("Isaac Lab PPO (baseline)", ppo), ("FlashCHORD FlashSAC", flash)):
    print(f"{name}: sim {d['sim_ms_per_step']:.1f} ms + learn {d['learn_ms_per_step']:.1f} ms "
          f"= {d['total_ms_per_step']:.1f} ms -> sim share {d['sim_share_pct']:.1f}%  [{d['window']}]")
print(f"FlashSAC unprofiled (overlapped) step: {flash['unprofiled_ms_per_step']:.1f} ms")
print(f"Simulation alone: Newton/MuJoCo-Warp is {out['sim_speedup_newton_vs_isaaclab']:.1f}x faster per step than Isaac Lab")
print("PPO also needs 8x more env steps for its full schedule (1,966M vs 250M).")

#!/usr/bin/env python3
"""Parse rsl-rl train logs -> artifacts/repro-taco/curve/reward.json (data layer
for the live reward-curve page; rule data-then-render). Stdlib only.

Pairs each 'Learning iteration N/M' with the 'Mean reward:' that follows it,
downsamples to <=1200 points per run, and adds live meta (ETA from recent
iteration timing lines).
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ART = REPO / "artifacts/repro-taco"
OUT = ART / "curve/reward.json"

RUNS = [
    {
        "name": "empty_cup_bowl_full (official example)",
        "key": "official",
        "log": ART / "phase8_train_full.log",
        "done_marker": ART / ".train_full_done",
    },
    {
        "name": "knife_2k_v1 (first run, baseline)",
        "key": "knife",
        "log": ART / "phase6_train2k.log",
        "done_marker": None,  # finished long ago
    },
]

IT_RE = re.compile(r"Learning iteration (\d+)/(\d+)")
RW_RE = re.compile(r"Mean reward:\s*([-\d.]+)")
SP_RE = re.compile(r"collection:\s*([\d.]+)s, learning\s*([\d.]+)s")


def parse(log: Path) -> tuple[list[list[float]], int, float | None]:
    series: list[list[float]] = []
    total = 0
    it = None
    sec_per_iter: list[float] = []
    for line in log.read_text(errors="ignore").splitlines():
        m = IT_RE.search(line)
        if m:
            it, total = int(m.group(1)), int(m.group(2))
            continue
        m = SP_RE.search(line)
        if m and it is not None:
            sec_per_iter.append(float(m.group(1)) + float(m.group(2)))
            continue
        m = RW_RE.search(line)
        if m and it is not None:
            series.append([it, float(m.group(1))])
            it = None
    spi = sum(sec_per_iter[-50:]) / len(sec_per_iter[-50:]) if sec_per_iter else None
    return series, total, spi


def downsample(series: list[list[float]], max_pts: int = 1200) -> list[list[float]]:
    if len(series) <= max_pts:
        return series
    stride = len(series) // max_pts + 1
    kept = series[::stride]
    if kept[-1] != series[-1]:
        kept.append(series[-1])
    return kept


def main() -> None:
    runs = []
    for r in RUNS:
        if not r["log"].is_file():
            continue
        series, total, spi = parse(r["log"])
        if not series:
            continue
        last_it, last_rw = series[-1]
        done = r["done_marker"] is None or r["done_marker"].is_file()
        eta_h = None
        if not done and spi and total:
            eta_h = round((total - last_it) * spi / 3600, 1)
        runs.append({
            "key": r["key"],
            "name": r["name"],
            "series": downsample(series),
            "total_iters": total,
            "last_iter": int(last_it),
            "last_reward": last_rw,
            "sec_per_iter": round(spi, 2) if spi else None,
            "status": "done" if done else "running",
            "eta_hours": eta_h,
        })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "updated_unix": int(time.time()),
        "updated": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "runs": runs,
    }))
    print(f"wrote {OUT} ({', '.join(f'{r['key']}:{len(r['series'])}pts' for r in runs)})")


if __name__ == "__main__":
    main()

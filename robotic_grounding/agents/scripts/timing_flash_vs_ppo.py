#!/usr/bin/env python3
"""Timing comparison: FlashCHORD FlashSAC vs Isaac Lab PPO on tissue_box_simple (Sharpa).

Both runs use their FULL (100%) configs so learning-rate and curriculum schedules match a
real run. Budgets per Peijun 2026-09-28: FlashSAC to 10% (2% mark read from the same run),
PPO to 2% only (400 of 20000 iters; its curriculum is per-iteration so --max_iterations 400
leaves the schedule untouched).
Runs are sequential on GPU0 only. Every stdout line is timestamped; results go to
results.json (data layer), index.html renders from it, and a summary email is sent.

Usage (detached):  setsid nohup python3 agents/scripts/timing_flash_vs_ppo.py &
Resume: not resumable per stage; rerun reruns both stages (delete results.json first).
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import threading
import time
from pathlib import Path

RG = Path("/root/video_to_data/robotic_grounding")
FC = RG / "flash_chord"
ART = RG / "artifacts/flash-vs-ppo-timing"
ASSETS = RG / "source/robotic_grounding/robotic_grounding/assets"
PARQUET = ASSETS / "human_motion_data/ego_recon/processed/sequence_id=tissue_box_simple/robot_name=sharpa_wave"
RESULTS = ART / "results.json"
NOTIFY = Path.home() / "meta/bin/notify.sh"
PAGE_URL = "https://peijun-reports.axon.tools/files/video_to_data/robotic_grounding/artifacts/flash-vs-ppo-timing/index.html"

FLASH_TOTAL = 250_003_456  # configs/experiment/sharpa_flash_sac.yaml training.total_environment_steps
PPO_TOTAL_ITERS = 20_000  # rsl_rl_ppo_cfg.py:16 max_iterations
PPO_STEPS_PER_ITER = 24 * 4096  # num_steps_per_env x num_envs

PPO_OVERRIDES = [  # README.md:496-507 ego-video force-closure recipe for tissue_box_simple
    "env.rewards.force_closure.weight=5.0",
    "env.rewards.contact_wrench_support_reward.weight=0.0",
    "env.rewards.unintended_contact_penalty.weight=0.0",
    "env.rewards.missed_contact_penalty.weight=0.0",
    "env.curriculum.fixed_timestep_curriculum.params.rewards_contact_wrench_support_reward=0.0",
    "env.curriculum.fixed_timestep_curriculum.params.rewards_unintended_contact_penalty=0.0",
    "env.curriculum.fixed_timestep_curriculum.params.rewards_missed_contact_penalty=0.0",
]


def save(results: dict) -> None:
    tmp = RESULTS.with_suffix(".tmp")
    tmp.write_text(json.dumps(results, indent=1))
    tmp.replace(RESULTS)


class GpuSampler(threading.Thread):
    """Samples GPU0 memory/utilization every 5 s (nvidia-smi index 0 = physical GPU0)."""

    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.samples: list[tuple[float, int, int]] = []
        self.stop = threading.Event()

    def run(self) -> None:
        while not self.stop.is_set():
            out = subprocess.run(
                ["nvidia-smi", "-i", "0", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                capture_output=True, text=True,
            ).stdout.strip()
            try:
                mem, util = (int(x) for x in out.split(","))
                self.samples.append((time.time(), mem, util))
            except ValueError:
                pass
            self.stop.wait(5.0)

    def summary(self) -> dict:
        if not self.samples:
            return {}
        mems = [m for _, m, _ in self.samples]
        busy = [u for _, _, u in self.samples[len(self.samples) // 5:]]  # skip startup fifth
        return {"peak_mem_mib": max(mems), "mean_util_pct": round(sum(busy) / max(len(busy), 1), 1),
                "samples": len(self.samples)}


def run_stage(name: str, cmd: list[str], cwd: Path, env: dict, parse, stop_when) -> dict:
    """Run cmd, timestamp each line to <name>.log, record parsed marks, stop when asked."""
    log_path = ART / f"{name}.log"
    sampler = GpuSampler()
    sampler.start()
    t0 = time.time()
    marks: list[tuple[float, float]] = []  # (seconds since launch, progress value)
    events: dict = {}
    proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1, start_new_session=True, stdin=subprocess.DEVNULL)
    stopped_early = False
    with log_path.open("w") as log:
        for line in proc.stdout:
            now = time.time() - t0
            log.write(f"{now:10.2f} {line}")
            log.flush()
            value = parse(line, events, now)
            if value is not None:
                marks.append((now, value))
                if stop_when(value):
                    stopped_early = True
                    os.killpg(proc.pid, signal.SIGINT)
                    try:
                        proc.wait(timeout=90)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                    break
    rc = proc.wait()
    sampler.stop.set()
    sampler.join()
    return {"cmd": cmd, "returncode": rc, "stopped_at_target": stopped_early, "wall_s": time.time() - t0,
            "marks": marks, "events": events, "gpu": sampler.summary(), "log": str(log_path)}


FLASH_RE = re.compile(r"training progress: ([\d,]+)/([\d,]+) environment steps")
PPO_RE = re.compile(r"Learning iteration (\d+)/(\d+)")


def flash_parse(line: str, events: dict, now: float):
    if "compiled FlashSAC JAX executables" in line:
        events["compiled_at_s"] = now
        m = re.search(r"in ([\d.]+)s", line)
        if m:
            events["compile_s"] = float(m.group(1))
    m = FLASH_RE.search(line)
    return float(m.group(1).replace(",", "")) if m else None


def ppo_parse(line: str, events: dict, now: float):
    m = PPO_RE.search(line)
    if m:
        events.setdefault("first_iter_at_s", now)
        return float(m.group(1))  # iteration index whose log block is being printed
    return None


def time_at(marks, target):
    for t, v in marks:
        if v >= target:
            return t, v
    return None


def extrapolate(marks, target_a, target_b, total, t2_target, t10_target=None):
    """Steady rate between marks a->b; 100% estimate = time at b + remaining / rate."""
    a, b = time_at(marks, target_a), time_at(marks, target_b)
    if not a or not b:
        return None
    (ta, va), (tb, vb) = a, b
    rate = (vb - va) / (tb - ta)
    p2 = time_at(marks, t2_target)
    p10 = time_at(marks, t10_target) if t10_target else None
    return {"rate_window": [va, vb], "rate_per_s": rate, "steady_100_s": tb + (total - vb) / rate,
            "t_2pct_s": p2[0] if p2 else None, "naive_100_from_2pct_s": p2[0] * total / p2[1] if p2 else None,
            "t_10pct_s": p10[0] if p10 else None,
            "naive_100_from_10pct_s": p10[0] * total / p10[1] if p10 else None}


def main() -> None:
    ART.mkdir(parents=True, exist_ok=True)
    results = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    results.update({"started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "gpu": "GPU0 RTX PRO 6000 Blackwell",
                    "commit": subprocess.run(["git", "-C", str(RG), "rev-parse", "--short", "HEAD"],
                                             capture_output=True, text=True).stdout.strip()})
    save(results)

    # Stage 1: FlashSAC (FlashCHORD, Newton/MuJoCo-Warp + JAX). CUDA_VISIBLE_DEVICES=0 is safe here (no Kit).
    fenv = dict(os.environ, CUDA_VISIBLE_DEVICES="0", PYGLET_HEADLESS="1")
    f10 = round(FLASH_TOTAL * 0.10)
    flash_cmd = [str(FC / ".venv/bin/python"), "-u", "scripts/train_flash_sac.py", "experiment=sharpa_flash_sac",
                 f"task.parquet='{PARQUET}'", "reset.seed=42", "logging.mode=disabled",
                 f"output_dir={ART / 'flash_10pct'}"]
    if not results.get("flash", {}).get("marks"):
        results["flash"] = run_stage("flash_10pct", flash_cmd, FC, fenv, flash_parse, lambda v: v >= f10)
        save(results)
        subprocess.run([str(NOTIFY), "⏱️ FlashSAC 10% timing done — PPO stage starting", ""])
    results["flash"]["extrapolation"] = extrapolate(results["flash"]["marks"], FLASH_TOTAL * 0.02, f10, FLASH_TOTAL,
                                                    FLASH_TOTAL * 0.02, f10)
    save(results)

    # Stage 2: Isaac Lab PPO (rsl_rl). Full-schedule cfg, capped at 2000 iters = 10%; curriculum is per-iteration.
    penv = dict(os.environ, OMNI_KIT_ACCEPT_EULA="YES", HUMAN_MOTION_DATA_DIR="/root/Data/human_motion_data",
                VK_DRIVER_FILES="/etc/vulkan/icd.d/nvidia_icd.json", VK_ICD_FILENAMES="/etc/vulkan/icd.d/nvidia_icd.json",
                VIRTUAL_ENV=str(RG / ".venv"), PATH=f"{RG / '.venv/bin'}:{os.environ['PATH']}")
    penv.pop("CUDA_VISIBLE_DEVICES", None)  # Kit degrades when masked; select with --device instead
    ppo_cmd = [str(RG / ".venv/bin/python"), "-u", "scripts/rsl_rl/train.py", "--headless", "--device", "cuda:0",
               "--task", "Sharpa-V2D-v0", "--motion_file", str(PARQUET), "--num_envs", "4096",
               "--logger", "tensorboard", "--run_name", "tissue_ppo_2pct",
               "--max_iterations", str(PPO_TOTAL_ITERS // 50), *PPO_OVERRIDES]
    results["ppo"] = run_stage("ppo_2pct", ppo_cmd, RG, penv, ppo_parse, lambda v: False)
    # iteration i's block prints after it finishes; progress = i+1 iterations done
    results["ppo"]["marks"] = [(t, v + 1) for t, v in results["ppo"]["marks"]]
    # steady rate over iters 100->400 (0.5%->2%), skipping Kit startup and the first slow iterations
    results["ppo"]["extrapolation"] = extrapolate(results["ppo"]["marks"], 100, PPO_TOTAL_ITERS * 0.02,
                                                  PPO_TOTAL_ITERS, PPO_TOTAL_ITERS * 0.02)
    results["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save(results)

    subprocess.run([str(FC / ".venv/bin/python"), str(RG / "agents/scripts/render_timing_page.py")])
    body = (ART / "summary.txt").read_text() if (ART / "summary.txt").exists() else json.dumps(
        {k: results[k].get("extrapolation") for k in ("flash", "ppo")}, indent=1)
    subprocess.run([str(NOTIFY), "✅ FlashSAC vs PPO timing report (tissue_box, GPU0)", f"{body}\n\nPage: {PAGE_URL}"])
    (ART / ".done").touch()


if __name__ == "__main__":
    main()

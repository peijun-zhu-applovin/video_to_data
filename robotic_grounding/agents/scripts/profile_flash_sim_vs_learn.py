#!/usr/bin/env python3
"""Split FlashCHORD FlashSAC wall-clock into simulation vs learning.

Runs the real scripts/train_flash_sac.py entry point with two timing hooks:
  sim   = WarpRLEnv.step (Newton/MuJoCo-Warp physics substeps, rewards, obs, resets)
  learn = compiled consume_transition (replay insert, `updates_per_collection` SAC updates,
          next-action inference) — one JAX executable per env step
The GPU is synchronized at both boundaries, which removes any sim/learn overlap; the sum is
therefore compared against the unprofiled end-to-end rate in the report.

Usage (from flash_chord/, CUDA_VISIBLE_DEVICES=0):
  .venv/bin/python <this> <hydra overrides for train_flash_sac.py>   -> writes $PROFILE_OUT (json)
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import jax
import warp as wp

FC = Path("/root/video_to_data/robotic_grounding/flash_chord")
OUT = Path(os.environ["PROFILE_OUT"])

import flash_chord.training.environment as environment  # noqa: E402
import flash_chord.training.flash_sac.runner as runner  # noqa: E402
from flash_chord.training.flash_sac.learner import LearnerExecutables  # noqa: E402

records: dict[str, list[float]] = {"sim": [], "learn": [], "replay_size": []}

_orig_step = environment.WarpRLEnv.step


def timed_step(self, action):
    wp.synchronize_device()
    start = time.perf_counter()
    result = _orig_step(self, action)
    wp.synchronize_device()
    records["sim"].append(time.perf_counter() - start)
    return result


environment.WarpRLEnv.step = timed_step

_orig_compile = runner._compile_executables


def timed_compile(*args, **kwargs):
    compiled = _orig_compile(*args, **kwargs)
    consume = compiled.consume_transition

    def timed_consume(learner, *rest):
        jax.block_until_ready(learner.observation)
        start = time.perf_counter()
        out = consume(learner, *rest)
        jax.block_until_ready(out)
        records["learn"].append(time.perf_counter() - start)
        records["replay_size"].append(int(out[0].replay.size))
        return out

    return LearnerExecutables(initial_action=compiled.initial_action, consume_transition=timed_consume)


runner._compile_executables = timed_compile

import runpy  # noqa: E402

script = FC / "scripts/train_flash_sac.py"
sys.argv[0] = str(script)  # Hydra resolves config_path relative to the running script file
try:
    runpy.run_path(str(script), run_name="__main__")
finally:
    OUT.write_text(json.dumps(records))
    print(f"profile written: {OUT} ({len(records['sim'])} sim, {len(records['learn'])} learn calls)")

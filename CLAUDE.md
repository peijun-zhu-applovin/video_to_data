# video_to_data (V2D) — on this box

Clone of [nvidia-isaac/video_to_data](https://github.com/nvidia-isaac/video_to_data)
(human demo video → sim-ready assets → physics-grounded robot policies, three composable
stages). **Upstream already ships deep Claude context — do not duplicate it**: each package
has its own CLAUDE.md (`video_ingestion_agent/`, `reconstruction/`, `robotic_grounding/`)
and `.claude/skills/` carries 11 project skills (`*_onboard`, `*_doctor`, `*_run`) — reach
for those first. This file only records what upstream cannot know: this machine.

## Facts about this box that gate what runs

- **`reconstruction/` does not run here.** This box is RTX PRO 6000 Blackwell (sm_120),
  which the README explicitly lists as unsupported by the TensorRT/cuVSLAM versions in
  `v2d_cusfm`; the container build checks and refuses. Do reconstruction work on a
  supported box (A6000/L40S) or skip the stage — its outputs are cacheable inputs.
- **Docker is not installed** (k8s pod; nested containers unverified). **Decision
  (Peijun, 2026-08-26): native uv venvs**, established and working — see
  `robotic_grounding/agents/docs/` for the full recipe. Corrections to earlier notes:
  - `run_retarget_local.sh` (and its wrapper `run_pipeline_local.sh`) **is** a native
    escape hatch — it is a plain sequence of `python scripts/*.py` calls, no Docker.
    Only `run_load_local.sh` genuinely shells out to Docker (`docker/run_loader.py`);
    natively, call `python -m v2d.task_library_loader.lib.run_loader` directly instead.
  - The pixi `isaaclab==2.2.0` pin is **stale** — the block is headed "You should use
    Docker, this is not maintained. For reference only." Version authority is the Docker
    base `nvcr.io/nvidia/isaac-lab:2.3.2` (`robotic_grounding/workflow/Dockerfile:13`).
  - Two venvs mirror the two Docker images: `/root/video_to_data/.venv-loader` (GPL MANO
    stack, torch 2.8.0+cu128) and `robotic_grounding/.venv` (Dockerfile pin list; Isaac
    layer added per `/root/regrind/agents/docs/1-setup.md`). Source
    `robotic_grounding/agents/env.sh` before running anything.
  - Docker-image fallback without Docker: pull `isaac-lab:2.3.2` layers via the registry
    HTTP API (skopeo-style) and read its site-packages as dependency ground truth.
- **`<HMD>` (human-motion-data root) can be assembled from `/root/Data`**: MANO pkls are at
  `/root/Data/mano/mano/` and TACO raw data at `/root/Data/taco-raw` — symlink into an HMD
  layout rather than re-downloading. Verify each dataset's expected sub-layout against
  `robotic_grounding/docs/SETUP.md` §4 before assuming.
- GPU/torch quirks of this box (cu128 wheels, sm_120 kernels — relevant for vLLM and any
  torch install) live in the `gpu-env` skill.
- This repo retargets human hands onto **Sharpa** — the same embodiment as `/root/regrind`
  and `/root/DexPrior`. The cross-repo IK objective comparison is central at
  `/root/DexPrior/agents/docs/17_ik_objective_comparison.md`; add V2D's retargeter there
  if it gets characterized.

## Current goal (2026-08-25)

Reproduce **Robotic Grounding** (stage 3) — reconstruction is skippable (public datasets
feed stage 3 directly), so the sm_120 limitation does not block this goal.

## Long runs, sweeps, evals

Follow the `experiment-runs` skill.

Repo parameters:
- Smoke (native, this box): load via `python -m v2d.task_library_loader.lib.run_loader
  --dataset taco --max_sequences 2 --save` (Env A), then
  `HUMAN_MOTION_DATA_DIR=$HMD WITH_VIS=1 MAX_SEQUENCES=2 bash run_retarget_local.sh taco`
  (Env B). Upstream's Docker smoke (`scripts/run_pipeline_docker.py taco --hmd <HMD>
  --mano-dir <HMD>/mano --max-sequences 2`) does not run here.
- Retargeting/rollout review: GLB per the `glb-review` skill; pages per `review-page`.
- W&B project: not yet established for this repo — ask before the first logged run.

## Workspace

`agents/` per the ~/meta convention (todo / worklog / docs). Upstream docs index is
`docs/` + per-package `docs/`; put box-local findings in `agents/docs/`, never edit
upstream docs in place.

Push access to `nvidia-isaac` upstream is unverified — treat CLAUDE.md and `agents/` as
local-only commits until the human says otherwise (same posture as the cosmos repo).

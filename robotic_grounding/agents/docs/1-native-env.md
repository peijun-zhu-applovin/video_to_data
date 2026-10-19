# Native (Docker-free) environment on this box

Established 2026-08-26. This k8s pod has no Docker; the pipeline runs natively in two uv
venvs that mirror upstream's two Docker images. Activate via `source agents/env.sh`
(pipeline env) or `RG_ENV=loader source agents/env.sh` (loader env).

## Env A — loader (mirrors IMAGE 1 `v2d_task_library_loader`, GPL isolation)

`/root/video_to_data/.venv-loader`, py3.11, torch **2.8.0+cu128** (matches the loader
image base `pytorch/pytorch:2.8.0-cuda12.8`; verify `'sm_120' in torch.cuda.get_arch_list()`).

```bash
cd /root/video_to_data && uv venv --python 3.11 .venv-loader
uv pip install -p .venv-loader torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install -p .venv-loader -e reconstruction/modules/v2d_common \
  -e reconstruction/modules/v2d_task_library_loader/lib \
  -e robotic_grounding/source/robotic_grounding \
  deprecation opencv-python-headless "numpy<2" \
  "git+https://github.com/lixiny/manotorch" pip setuptools wheel
uv pip install -p .venv-loader --no-build-isolation "git+https://github.com/mattloper/chumpy"
```

- **chumpy needs `--no-build-isolation`** (its setup.py does `import pip`, absent in uv's
  isolated build env) — and pip/setuptools/wheel must already be in the venv.
- **pytorch3d is skipped on purpose** (no build on sm_120/cu128 — `gpu-env` skill); the
  loader falls back to trimesh sampling (`lib/distance_utils.py:17`).
- Native load run (bypasses the Docker bind-mount convention with explicit args):
  `python -m v2d.task_library_loader.lib.run_loader --dataset taco --dataset_root $HMD/taco/dataset
   --mesh_dir $HMD/taco/object_assets/meshes/taco --object_model_root $HMD/taco/object_assets/urdfs/taco
   --mano_model_dir $HMD/mano --output_dir $HMD/taco/taco_loaded --device cuda:0 --save`

## Env B — pipeline + Isaac (mirrors IMAGE 2 `robotic-grounding`)

`robotic_grounding/.venv`, py3.11, torch **2.7.0+cu128**.

1. torch first: `uv pip install -p .venv torch==2.7.0 torchvision==0.22.0 --index-url
   https://download.pytorch.org/whl/cu128`
2. The full `workflow/Dockerfile` pin list (lines 31–101) in ONE uv call **without**
   `numpy<2`, then `uv pip install -p .venv "numpy<2.0.0"` as a second call — uv resolves
   jointly and `cmeel-boost==1.87.0.1` declares `numpy>=2.2`; the sequential downgrade
   mirrors the Dockerfile's last RUN and works at runtime (Docker image has the same
   declared conflict).
**⚠ INSTALL ORDER (learned the hard way 2026-08-27): Isaac layer FIRST, pipeline pins
SECOND** — exactly like the Dockerfile, which layers the pin list on top of the isaac-lab
base image. The isaacsim/isaaclab install silently clobbers pipeline pins: verified
downgrades/upgrades were `pin` 3.7.0→2.7.0 (+ eigenpy/cmeel-*/pin-pink, plus a stray
`hpp-fcl` whose presence breaks the cmeel prefix), `qpsolvers` 4.9.0→4.13.0 (daqp API
mismatch: "solve() got an unexpected keyword argument 'primal_start'"), `websockets`
16→12 (viser needs websockets.asyncio). If the layers were applied in the wrong order,
re-apply the whole Dockerfile pin list, `uv pip uninstall hpp-fcl`, and if pinocchio still
imports the old build, `rm -rf .venv/.../site-packages/cmeel.prefix` and `--reinstall`
the closure (cmeel packages share that prefix; overlapping installs leave stale files).
Then numpy<2 + scipy==1.15.3 again, and re-verify: BuildFromMJCF exists, sm_120 in torch,
isaaclab imports, websockets>=15, qpsolvers==4.9.0.

3. Isaac layer (regrind recipe adapted to **v2.3.2**; scripted in the worklog):
   `isaacsim[all,extscache]==5.1.0 --extra-index-url https://pypi.nvidia.com` → **reinstall
   torch cu128** (isaacsim clobbers it with cu126; re-check sm_120) → `setuptools<81` →
   `flatdict==4.0.1 --no-build-isolation` → editable `isaaclab{,_assets,_mimic,_rl[rsl-rl],_tasks}`
   from `/root/IsaacLab-2.3.2` (tag v2.3.2, depth-1 clone; do NOT touch regrind's
   `/root/IsaacLab` v2.3.0) → re-assert `numpy<2`.
- The pinned pinocchio closure imported clean — the `liburdfdom_sensor.so.4.0` landmine
  (see `tests/test_retarget_pipeline_e2e.py:117`) did not fire with the exact Dockerfile pins.

## Version authority (corrected 2026-08-26)

- `pyproject.toml`'s pixi block (`isaaclab==2.2.0`) is headed "not maintained, reference
  only". The authority is `workflow/Dockerfile:13` → `nvcr.io/nvidia/isaac-lab:2.3.2`,
  hence IsaacLab v2.3.2 + isaacsim 5.1.0 (also `source/robotic_grounding/setup.py`
  classifier "Isaac Sim :: 5.1.0").
- `run_retarget_local.sh` runs natively as-is (plain `python scripts/*.py` calls) despite
  its "run inside the container" header; `run_load_local.sh` does NOT (it shells to docker).

## Asset routing without bind-mounts

`MESHES_DIR`/`ASSETS_DIR` are package-relative (`retarget/__init__.py`); Docker bind-mounts
over them. Natively, symlink instead (done):
- `source/robotic_grounding/robotic_grounding/assets/meshes/taco` → `$HMD/taco/object_assets/meshes/taco`
- `source/robotic_grounding/robotic_grounding/assets/urdfs/taco` → `$HMD/taco/object_assets/urdfs/taco`
`HUMAN_MOTION_DATA_DIR` is the only env-overridable path. `git lfs pull
--include='robotic_grounding/**'` is mandatory (sharpawave STLs; pinocchio BuildFromMJCF).

## Verified findings that contradict docs

- **Sharpa dual-hand parquet is legacy `ManoSharpaData`, not `motion_v1`** — both
  `taco_loaded` and `taco_processed` have no `schema_version`;
  `tasks/scene_utils/replay_data.py:270` handles this explicitly ("not yet on motion_v1").
  The repo CLAUDE.md's "everything reads/writes motion_v1" holds only for G1/Dex3 producers.
- **`hand_penetration` quality check is units-broken for taco**:
  `scripts/filter_penetrations.py::_load_hull` (~line 256) loads `*_cm.obj` without the
  ×0.01 scale, making hulls 100× too big → scores of 345 "cm" and universal rejection.
  `scripts/retarget/vis_retargeted.py` (~line 398) fixed exactly this for its renderer,
  with a comment explaining the dataset-check pitfall. Fix locally before trusting
  `taco_quality.csv`.
- taco needs BOTH tool and target meshes per sequence; `--mano-dir` layout is
  `<HMD>/mano/models/MANO_{LEFT,RIGHT}.pkl` (the `BMC/` mention in `run_load_local.sh` is
  vestigial — nothing in the loader reads it).
- **The repo CLAUDE.md's eval command is stale**: `eval.py` has no `--eval_episodes`
  (argparse error); the episode-length control is `--max_steps N`. `--checkpoint` comes
  from the appended RSL-RL cli args and works. Verified 2026-08-26.
- Real-mesh penetration measurement (correct scale, actual mesh not hull):
  `agents/scripts/analyze_penetration.py` → knife seq max 14.3 mm robot / 8.1 mm already in
  the MANO source; brush seq 20.9 mm / 11.9 mm; hand-hand 0. IK has no collision term
  (FrameTasks only), so reference penetration at this scale is expected and left to physics.

## 3D review tooling

`agents/scripts/export_processed_glb.py` exports a processed sequence to animated GLB
(pinocchio FK over the MJCF visual model + objects + green MANO fingertip targets) with a
`.metrics.json` sidecar, and self-validates FK vs the parquet's stored task frames
(agreement ~7e-8 m). Viewer page: `artifacts/repro-taco/glb/index.html` (vendored three.js
at `artifacts/vendor/`, copied from regrind).

## Penetration measurement: thin-shell blindness (learned 2026-08-31, from Peijun's review)

Depth-into-solid signed distance (trimesh signed_distance on the real mesh) CANNOT detect
a finger crossing a thin-shell object: past the ~2 mm wall the probe reads "outside the
solid" (the cavity is exterior). Measure wall crossing as **segment-vs-surface
intersections** along the kinematic chain (consecutive frame origins / capsule axes,
`bowl.ray.intersects_location`). Confirmed case: empty_cup_bowl rollout, right thumb
PP→IP segment crosses the bowl wall from t=0 (spawned interpenetrating by RSI reset with
an IK reference that presses through the wall) until ~t=4 s. Sim-side enablers:
`assets/rigid_object.py` spawns objects with collider_type='convex_decomposition',
contact_offset=0, rest_offset=0, max_depenetration_velocity=1.0 — a 1.9 mm shell
(est. 2·V/A; bowl solid 80 cm³ vs hull 1150 cm³) is below reliable PhysX contact
generation. Upstream's hand_penetration gate (convex hull) can't see this either.

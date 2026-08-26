# TODO — video_to_data

Goal: reproduce **Robotic Grounding** (Peijun, 2026-08-25).

## Human
- [ ] **Review P1–P3 pages** (load smoke, pipeline funnel + quality-gate incident, 3D GLB
      review) — dummy/train smoke is held on this per the 2-unreviewed cadence:
      http://localhost:8889/files/video_to_data/robotic_grounding/artifacts/repro-taco/index.html
- [ ] Call: fix the taco units bug in `filter_penetrations.py::_load_hull` (one-liner,
      mirrors vis stage) so `taco_quality.csv` becomes meaningful?
- [ ] W&B project for this repo — needed before the first logged training run.

## Agent (after review)
- [ ] P4: dummy_agent smoke (`Sharpa-V2D-v0-Play`, headless, record_video) on GPU0.
- [ ] P5: train 1-iter + eval smoke; stop before any real training (approval gate).
- [ ] If scaling up: extract more taco sequences from `/root/Data/taco-raw/*.zip`
      (full V1 = 2316 seqs; meshes already all linked).

## Done (2026-08-26)
- [x] Environment route decided (native uv venvs) and BUILT: Env A loader,
      Env B pipeline + isaacsim 5.1.0 / IsaacLab v2.3.2. Recipe:
      `robotic_grounding/agents/docs/1-native-env.md`.
- [x] `<HMD>` assembled from /root/Data (symlinks); taco load + retarget + support +
      vis + assess all ran natively (2 sequences).
- [x] Review pages shipped (`artifacts/repro-taco/`, incl. 3D GLB viewer).

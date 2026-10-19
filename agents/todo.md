# TODO — video_to_data

Goal: reproduce **Robotic Grounding** (Peijun, 2026-08-25). **Smoke-scope COMPLETE 2026-08-26**
(load → retarget → vis → dummy → train → eval, all native).

## In flight
- (nothing — official full run COMPLETE 2026-08-31: 20000 iters, reward ~127, 20 rollout
  videos + final-checkpoint rollout GLB on the pages)

## Human
- [ ] Review pages when convenient:
      https://peijun-reports.axon.tools/files/video_to_data/robotic_grounding/artifacts/repro-taco/index.html
- [ ] W&B key restore (netrc lost): `! wandb login --host https://cbf-wandb.alcfd.com <key>`,
      then agent syncs knife_2k_v1 + empty_cup_bowl_full to project video_to_data.
- [ ] Optional: resume knife_2k_v1 to 20000 after the official run (not official; skippable).
- [ ] Call: patch filter_penetrations.py `_cm.obj` scale (one-liner) so taco_quality.csv is
      meaningful? (Real penetration measured separately: 14–21 mm, ~half inherited from MANO
      source — agents/scripts/analyze_penetration.py.)
- [ ] Commit of today's agents/ updates (worklog, docs, analyzer script) is prepared but
      NOT made — say the word.

## Agent (after decisions)
- [ ] If real training: extract sequences, chain per-checkpoint eval at launch
      (chained-eval), live progress page, W&B logging.

## Done (2026-08-26)
- [x] Native env route built and validated end-to-end (two uv venvs; recipe in
      robotic_grounding/agents/docs/1-native-env.md).
- [x] taco: load 2/2 → 206 URDFs → IK 2/2 → support → vis (mp4+viser) → assess.
- [x] 3D GLB review shipped; penetration adjudicated by Peijun (acceptable, proceed).
- [x] dummy_agent smoke (first native Kit boot in pod) + train 1-iter + eval + policy export.

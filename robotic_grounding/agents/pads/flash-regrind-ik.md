# flash-regrind-ik — FlashCHORD with V2D IK vs regrind high-quality IK, 5 TACO bimanual episodes

## Goal
Peijun 2026-10-02: redo regrind's report 4 episodes
(`/root/regrind/artifacts/reports-1001/bimanual-tolerance/`) with FlashCHORD, each episode twice:
**A = original FlashCHORD** (V2D IK, default reset) and **B = FlashCHORD on regrind's high-quality IK**
(+ `reset=reference_high_quality_ik`). "用两个 gpu 吧. 五个 episode 都跑, 不过先跑完那两个重点的."
Priority: brush_brush_toy_038, scrape-off_spoon_plate_031; then kettle_106, pan_044, helmet_003.

## Approach
- Scenes (regrind demo meta `source_scene`): toy_038 `(brush, brush, toy)/20230919_038`, plate_031
  `(scrape off, spoon, plate)/20230926_031`, kettle_106 `(brush, eraser, kettle)/20231103_106`, pan_044
  `(brush, brush, pan)/20230926_044`, helmet_003 `(brush, brush, helmet)/20231027_003`.
- A data: V2D loader on `/root/Data/taco-raw/v1` (--sequence_pattern) → taco_loaded; `run_retarget_local.sh taco`
  → taco_processed + reconstructed_stage. Train like flash_full_taco.sh (CWD = taco dataset dir).
- B data: regrind package `/root/Data/regrind-bimanual-ik-1002/` (made by regrind session "bimanual",
  recipe mink_init_tol0_dextip_ladder_halfspace_1001, all gate-pass; README there). Transplant its
  right/left (true world, 120 fps, wxyz, root link `<side>_hand_C_MC`, joints by name) into a copy of
  the A parquet. Use `joint_scene.use` files (pan/kettle → `_sep`). Frame relation to V2D world: TO VERIFY
  by comparing mano_joint_coords vs parquet mano_*_joints. Field mapping: research agent report (pending).
- Reset B: `reset=reference_high_quality_ik` (finger [0.75, 0.93], settle 10) — Peijun: only with high-quality IK.

## Progress
- 2026-10-02: V2D load of 5 scenes OK (16 s). Retarget running.
- Frame check (toy_038, plate_031): V2D loaded mano_*_joints == regrind mano_joint_coords[::4] (0.0 mm,
  both hands); object_body_position/wxyz == regrind object_pos/quat[::4] (0.0 mm, |dot|-1 < 1e-7).
  => same world frame, V2D frame k = regrind frame 4k; transplant touches robot_* fields only.
- Retarget (V2D IK) of 5 scenes OK in 1.6 min; support USDAs written (kettle has 2 surfaces).
  `hand_penetration` "fail" on 4 = known units bug (docs/1-native-env.md), sequences stay in taco_processed.
- Support: V2D support top vs regrind joint-scene table: toy 0.5424 vs 0.5411 m (+1.3 mm), plate 0.5428 vs
  0.5462 m (−3.4 mm). Decision (mine, reversible): keep V2D support for BOTH A and B so A/B differ only in
  IK + reset config.

- Support clearance (agents/scripts/check_support_clearance.py -> artifacts/flash-regrind-ik/support_clearance.json;
  only geometry over a support disc, vs that disc's top): objects frame 0 never below support (0.0..+7.4 mm);
  over the episode ≥ -1.96 mm. Hands, V2D IK (A): frame 0 -3..-40 mm (inside the table), episode min to -45 mm.
  Hands, regrind IK (B): frame 0 -1.0..+7.9 mm, episode min ≥ -1.93 mm. Training config has
  robot_support_collision: false, so A's hands pass through the table without force (no reset explosion).
- IK transplant facts (research agent): wrist cols = world pose of <side>_hand_C_MC (no offset); FlashCHORD URDF
  == V2D MJCF == regrind URDF kinematics (DP FK 0.000 mm), joints by name, limits identical; regrind LEFT quat
  needs q ⊗ Rz(pi) (wxyz (w,x,y,z) -> (-z, y, -x, w)); overwrite robot_{side}_wrist_position/_wxyz/_finger_joints,
  recompute robot_{side}_frames by FK (unchecked by FlashCHORD); keep mano_*, object_*, fps=30, take h5[::4].

- B built (agents/scripts/transplant_regrind_ik.py -> taco/taco_processed_regrind_ik/, transplant.json per seq):
  FK self-test vs V2D frames ≤ 7e-8 m; FK of transplanted fields vs regrind keypoints ≤ 0.001 mm (all 5, L incl. Rz(pi)).
  pan/kettle use _sep files. FlashCHORD loads + replays B; zero-residual physical replays A/B for toy/plate in
  artifacts/flash-regrind-ik/replays/. B smoke (plate, 491,520 steps, reset=reference_high_quality_ik) OK.
- Peijun 2026-10-02 ~07:40 UTC: "let's finish these. i will check back tomorrow morning." -> overnight queue.

## In flight (overnight)
- A toy (GPU0) / A plate (GPU1): `pgrep -af "^bash agents/scripts/[f]lash_full_taco"`.
- Queues `pgrep -af "^bash agents/scripts/[q]ueue_regrind5"` (agents/scripts/queue_regrind5.sh), each waits for its GPU's A run:
  GPU0: B toy -> A kettle -> B kettle -> A helmet; GPU1: B plate -> A pan -> B pan -> B helmet.
- Page refresher loop every 10 min; final aggregator `pgrep -af "[r]ender_regrind5_page.py --wait"` emails when queues exit.
- Per-run start/done/fail emails from flash_full_taco.sh. ETA all done ≈ 16:30 UTC (10 runs × ~1.9 h over 2 GPUs).
- Done proof per run: artifacts/flash-regrind-ik/<A_v2d_ik|B_regrind_ik>/<seq>/.done.
- Resume (safe, skips finished stages): rerun queue_regrind5.sh <gpu> - <items...> with the unfinished items.

## State
Unattended overnight: the 10 runs + evals + policy videos + page + email. Decided alone (reversible): V2D support for
both A and B; support collision stays off (option "B + robot_support_collision=true" offered, not queued).
Morning: page https://peijun-reports.axon.tools/files/video_to_data/robotic_grounding/artifacts/flash-regrind-ik/index.html
Next human decision: judge videos A vs B; whether to run B with hand–table collision on; whether to split the two B knobs.

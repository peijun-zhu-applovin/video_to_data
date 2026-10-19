# 2026-10-02 — FlashCHORD on regrind report-4 episodes: V2D IK (A) vs regrind IK (B)

- regrind session "bimanual" built /root/Data/regrind-bimanual-ik-1002 (10 fresh solves, halfspace_1001, all gate-pass).
- V2D load of 5 scenes from /root/Data/taco-raw/v1 (16 s) + run_retarget_local.sh (1.6 min). hand_penetration "fail" x4 =
  known units bug. Frame check: V2D == regrind[::4] for MANO joints and object poses (0.0 mm).
- Reset config `reset=reference_high_quality_ik` (+ ReferenceResetConfig.reset_finger_openness_min, test added).
- Research agent: field mapping; left wrist needs q ⊗ Rz(pi); FlashCHORD URDF == V2D MJCF == regrind kinematics.
- Support clearance (check_support_clearance.py): objects frame0 ≥ 0 mm; V2D-IK hands -3..-40 mm inside the table at
  frame 0 (hand-support collision off in training so no force); regrind hands ≥ -1.9 mm over the episode.
- Transplant (transplant_regrind_ik.py): checks ≤ 0.001 mm for all 5. Replays A/B rendered for toy/plate. B smoke OK.
- Launched: A toy/plate (07:10 UTC), then queue_regrind5.sh on both GPUs (8 more runs), page refresher, final email.

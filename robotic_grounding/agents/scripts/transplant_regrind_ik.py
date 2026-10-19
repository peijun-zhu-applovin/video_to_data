#!/usr/bin/env python3
"""IK transplant: V2D ManoSharpaData parquet with its robot_* fields replaced by regrind's IK solution.

Inputs: V2D <taco>/taco_processed/sequence_id=<seq>/robot_name=sharpa_wave/*.parquet and the regrind package
/root/Data/regrind-bimanual-ik-1002 (README there; files chosen by index.json joint_scene.use).
Output: <taco>/taco_processed_regrind_ik/sequence_id=<seq>/robot_name=sharpa_wave/<same name>.parquet
(same dataset dir, so dataset-relative object paths and reconstructed_stage/<seq>_support.usda still resolve)
+ transplant.json beside it (provenance and checks, data layer).

Facts used (verified, see agents/pads/flash-regrind-ik.md):
- same world frame; V2D 30 fps frame k == regrind 120 fps frame 4k -> take h5[::4]
- wrist columns = world pose of <side>_hand_C_MC; regrind LEFT root frame = FlashCHORD/V2D left root ⊗ Rz(pi)^-1,
  so q_v2d = q_regrind ⊗ Rz(pi); joints mapped by name
- robot_<side>_frames recomputed with V2D's own FK (pinocchio MJCF + free flyer, hand_kinematics.py:459-464),
  self-tested by reproducing V2D's stored frames from V2D's own fields first.
Overwritten: robot_<side>_{wrist_position,wrist_wxyz,finger_joints,frames}; robot_<side>_frame_task_errors and
_num_optimization_iterations set to NaN/-1 (unread by FlashCHORD, stale otherwise). Everything else is kept.

Run with the robotic_grounding venv (pinocchio): .venv/bin/python agents/scripts/transplant_regrind_ik.py <key> [...]
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pinocchio as pin
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

TACO = Path("/root/Data/human_motion_data/taco")
PKG = Path("/root/Data/regrind-bimanual-ik-1002")
XML = Path("/root/video_to_data/robotic_grounding/source/robotic_grounding/robotic_grounding/assets/xmls/sharpawave")
SEQ_OF = {
    "brush_brush_toy_038": "taco_brush__brush__toy_20230919_038",
    "scrape-off_spoon_plate_031": "taco_scrape_off__spoon__plate_20230926_031",
    "brush_eraser_kettle_106": "taco_brush__eraser__kettle_20231103_106",
    "brush_brush_pan_044": "taco_brush__brush__pan_20230926_044",
    "brush_brush_helmet_003": "taco_brush__brush__helmet_20231027_003",
}
RZ_PI = Rotation.from_euler("z", np.pi)


class FK:
    """V2D's hand model: pinocchio MJCF with a free-flyer root (hand_kinematics.py:459-464)."""

    def __init__(self, side: str):
        self.robot = pin.RobotWrapper.BuildFromMJCF(str(XML / f"{side}_sharpawave.xml"), root_joint=pin.JointModelFreeFlyer())
        self.model, self.data = self.robot.model, self.robot.model.createData()
        self.joint_names = [self.model.names[i] for i in range(2, self.model.njoints)]
        self.frame_names = [f.name for f in self.model.frames]

    def frames(self, pos, wxyz, joints: dict) -> np.ndarray:
        q = np.zeros(self.model.nq)
        q[:3] = pos
        q[3:7] = [wxyz[1], wxyz[2], wxyz[3], wxyz[0]]
        for name in self.joint_names:
            q[self.model.idx_qs[self.model.getJointId(name)]] = joints[name]
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        out = []
        for i in range(len(self.model.frames)):
            M = self.data.oMf[i]
            out.append(np.hstack([M.translation, Rotation.from_matrix(M.rotation).as_quat(scalar_first=True)]))
        return np.asarray(out)


def frames_err(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Max position error (m) and max rotation error (rad) between (T,F,7) frame arrays."""
    dp = np.linalg.norm(a[..., :3] - b[..., :3], axis=-1).max()
    dot = np.abs(np.sum(a[..., 3:] * b[..., 3:], axis=-1)).clip(0, 1)
    return float(dp), float((2 * np.arccos(dot)).max())


def main(keys: list[str]) -> None:
    idx = json.loads((PKG / "index.json").read_text())
    fk = {s: FK(s) for s in ("right", "left")}
    for key in keys:
        seq = SEQ_OF[key]
        src = Path(glob.glob(str(TACO / "taco_processed" / f"sequence_id={seq}" / "robot_name=sharpa_wave" / "*.parquet"))[0])
        table = pq.read_table(src)
        row = {c: table.column(c)[0].as_py() for c in table.column_names}
        T = len(row["object_body_position"])
        ep = idx["episodes"][key]
        use = dict(zip(("right", "left"), ep["joint_scene"]["use"]))
        report = {"key": key, "sequence_id": seq, "source_parquet": str(src), "recipe": idx["recipe"],
                  "files": {s: ep["files"][use[s]] for s in use}, "frames": T, "checks": {}}
        for side in ("right", "left"):
            f = fk[side]
            names = row[f"{side}_robot_frame_names"]
            if names != f.frame_names:
                raise SystemExit(f"{key} {side}: parquet frame names differ from the MJCF frame list")
            fingers = row[f"{side}_robot_finger_joint_names"]
            # Self-test: V2D's own fields must reproduce V2D's stored frames with this FK.
            own = np.array([f.frames(row[f"robot_{side}_wrist_position"][k], row[f"robot_{side}_wrist_wxyz"][k],
                                     dict(zip(fingers, row[f"robot_{side}_finger_joints"][k]))) for k in range(T)])
            report["checks"][f"{side}_fk_selftest_vs_v2d_frames"] = dict(zip(("pos_m", "rot_rad"),
                                                                             frames_err(own, np.array(row[f"robot_{side}_frames"]))))
            with h5py.File(PKG / ep["files"][use[side]]) as h:
                pos, quat, jnt, kp = (h[n][::4] for n in ("robot_pos", "robot_quat", "robot_joints", "robot_keypoints"))
            if len(pos) != T:
                raise SystemExit(f"{key} {side}: h5[::4] has {len(pos)} frames, parquet {T}")
            if side == "left":
                quat = np.array([(Rotation.from_quat(q, scalar_first=True) * RZ_PI).as_quat(scalar_first=True) for q in quat])
            jmap = [dict(zip(idx[f"joint_names_{side}"], j)) for j in jnt]
            if set(jmap[0]) != set(fingers):
                raise SystemExit(f"{key} {side}: joint name sets differ")
            new_frames = np.array([f.frames(pos[k], quat[k], jmap[k]) for k in range(T)])
            # Check: our FK of the transplanted fields reproduces regrind's own keypoints (frames named in index.json).
            kp_frames = [n.replace("right_", f"{side}_") for n in idx["keypoint_robot_frames"]]
            cols = [(i, names.index(n)) for i, n in enumerate(kp_frames) if n in names]
            err = np.linalg.norm(new_frames[:, [c for _, c in cols], :3] - kp[:, [i for i, _ in cols]], axis=-1)
            report["checks"][f"{side}_fk_vs_regrind_keypoints"] = {"frames_compared": len(cols),
                                                                   "max_mm": float(err.max() * 1e3), "mean_mm": float(err.mean() * 1e3)}
            row[f"robot_{side}_wrist_position"] = pos.tolist()
            row[f"robot_{side}_wrist_wxyz"] = quat.tolist()
            row[f"robot_{side}_finger_joints"] = [[m[n] for n in fingers] for m in jmap]
            row[f"robot_{side}_frames"] = new_frames.tolist()
            for c, fill in ((f"robot_{side}_frame_task_errors", np.nan), (f"robot_{side}_num_optimization_iterations", -1)):
                if c in row and row[c] is not None:
                    row[c] = (np.full(np.shape(row[c]), fill)).tolist()
        out_dir = TACO / "taco_processed_regrind_ik" / f"sequence_id={seq}" / "robot_name=sharpa_wave"
        out_dir.mkdir(parents=True, exist_ok=True)
        new = pa.Table.from_pylist([row], schema=table.schema)
        pq.write_table(new, out_dir / src.name)
        (out_dir / "transplant.json").write_text(json.dumps(report, indent=1))
        print(json.dumps(report["checks"], indent=1))
        print("wrote", out_dir / src.name)


if __name__ == "__main__":
    main(sys.argv[1:] or list(SEQ_OF))

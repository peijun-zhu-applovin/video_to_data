#!/usr/bin/env python3
"""Clearance of hands and objects above the V2D support surface, frame 0 and whole episode.

For each episode: lowest world z of (a) each object mesh (V2D's OBJ, cm -> m), (b) each hand's collision
meshes posed by V2D's IK (variant A, processed parquet) and by regrind's IK (variant B, package h5 [::4],
left wrist quaternion ⊗ Rz(pi) to FlashCHORD's left root frame), against the V2D support cylinder top
(reconstructed_stage/<seq>_support.usda). clearance = min over vertices that lie over a support disc (xy within its radius) of (z - that disc's top);
negative = penetration. Geometry outside every disc is ignored (it hangs past the table edge).
Hand geometry: FlashCHORD's own URDF (flash_chord/src/flash_chord/assets/urdfs/sharpawave/<side>_sharpawave.urdf).
Writes artifacts/flash-regrind-ik/support_clearance.json (data) and prints a table.

Run: uv run --no-project --with yourdfpy --with trimesh --with h5py --with pyarrow --with numpy --with scipy python <this>
"""

from __future__ import annotations

import glob
import json
import re
from pathlib import Path

import h5py
import numpy as np
import pyarrow.parquet as pq
import trimesh
import yourdfpy
from scipy.spatial.transform import Rotation

TACO = Path("/root/Data/human_motion_data/taco")
PKG = Path("/root/Data/regrind-bimanual-ik-1002")
URDF = Path("/root/video_to_data/robotic_grounding/flash_chord/src/flash_chord/assets/urdfs/sharpawave")
OUT = Path("/root/video_to_data/robotic_grounding/artifacts/flash-regrind-ik/support_clearance.json")
EPISODES = {  # regrind key -> V2D sequence id
    "brush_brush_toy_038": "taco_brush__brush__toy_20230919_038",
    "scrape-off_spoon_plate_031": "taco_scrape_off__spoon__plate_20230926_031",
    "brush_eraser_kettle_106": "taco_brush__eraser__kettle_20231103_106",
    "brush_brush_pan_044": "taco_brush__brush__pan_20230926_044",
    "brush_brush_helmet_003": "taco_brush__brush__helmet_20231027_003",
}
RZ_PI = Rotation.from_euler("z", np.pi)


def support_tops(seq: str) -> list[dict]:
    text = (TACO / "reconstructed_stage" / f"{seq}_support.usda").read_text()
    out = []
    for block in re.findall(r'def Cylinder "[^"]+".*?\{(.*?)\n    \}', text, re.S):
        h = float(re.search(r"double height = ([\d.eE+-]+)", block).group(1))
        r = float(re.search(r"double radius = ([\d.eE+-]+)", block).group(1))
        t = [float(v) for v in re.search(r"xformOp:translate = \(([^)]*)\)", block).group(1).split(",")]
        out.append({"top_z": t[2] + h / 2, "center_xy": t[:2], "radius": r})
    return out


class Hand:
    def __init__(self, side: str):
        self.robot = yourdfpy.URDF.load(str(URDF / f"{side}_sharpawave.urdf"), build_collision_scene_graph=True,
                                        load_collision_meshes=True, build_scene_graph=False, load_meshes=False)
        self.names = set(self.robot.actuated_joint_names)

    def vertices(self, pos, wxyz, joints: dict) -> np.ndarray:
        self.robot.update_cfg({k: v for k, v in joints.items() if k in self.names})
        mesh = self.robot.collision_scene.to_mesh() if hasattr(self.robot.collision_scene, "to_mesh") \
            else self.robot.collision_scene.dump(concatenate=True)
        R = Rotation.from_quat([wxyz[1], wxyz[2], wxyz[3], wxyz[0]]).as_matrix()
        return mesh.vertices @ R.T + np.asarray(pos)


def clearance(v: np.ndarray, tops: list[dict]) -> float:
    """Min (z - top) over vertices above any support disc; +inf if none is over a disc."""
    best = np.inf
    for t in tops:
        inside = np.linalg.norm(v[:, :2] - np.asarray(t["center_xy"]), axis=1) <= t["radius"]
        if inside.any():
            best = min(best, float((v[inside, 2] - t["top_z"]).min()))
    return best


def main() -> None:
    hands = {s: Hand(s) for s in ("right", "left")}
    idx = json.loads((PKG / "index.json").read_text())
    report = {}
    for key, seq in EPISODES.items():
        tops = support_tops(seq)
        top = 0.0  # clearances below are already relative to each disc's own top
        f = glob.glob(str(TACO / "taco_processed" / f"sequence_id={seq}" / "**" / "*.parquet"), recursive=True)[0]
        t = pq.read_table(f)
        col = lambda c: np.array(t.column(c)[0].as_py())  # noqa: E731
        T = col("object_body_position").shape[0]
        ep = {"support": tops, "frames": T}
        # objects: lowest mesh vertex over time
        paths, P, Q = t.column("object_mesh_paths")[0].as_py(), col("object_body_position"), col("object_body_wxyz")
        for b, (name, mp) in enumerate(zip(t.column("object_body_names")[0].as_py(), paths)):
            verts = trimesh.load(str(TACO / mp), force="mesh").vertices * 0.01
            z = np.array([clearance(verts @ Rotation.from_quat(Q[k, b][[1, 2, 3, 0]]).as_matrix().T + P[k, b], tops)
                          for k in range(T)])
            ep[f"object_{name}"] = {"frame0_clearance_mm": (z[0] - top) * 1e3, "min_clearance_mm": (z.min() - top) * 1e3,
                                    "min_frame": int(z.argmin())}
        ep_idx = idx["episodes"][key]
        use = dict(zip(("right", "left"), ep_idx["joint_scene"]["use"]))
        for side in ("right", "left"):
            names = t.column(f"{side}_robot_finger_joint_names")[0].as_py()
            A = (col(f"robot_{side}_wrist_position"), col(f"robot_{side}_wrist_wxyz"), col(f"robot_{side}_finger_joints"))
            with h5py.File(PKG / ep_idx["files"][use[side]]) as h:
                pos, quat, jnt = h["robot_pos"][::4], h["robot_quat"][::4], h["robot_joints"][::4]
            bnames = idx[f"joint_names_{side}"]
            if side == "left":  # regrind left root frame -> FlashCHORD left root frame
                quat = np.array([(Rotation.from_quat(q[[1, 2, 3, 0]]) * RZ_PI).as_quat()[[3, 0, 1, 2]] for q in quat])
            for variant, (p, q, j, jn) in {"A_v2d_ik": (*A, names), "B_regrind_ik": (pos, quat, jnt, bnames)}.items():
                zs = [clearance(hands[side].vertices(p[k], q[k], dict(zip(jn, j[k]))), tops) for k in range(min(T, len(p)))]
                zs = np.array(zs)
                ep[f"hand_{side}_{variant}"] = {"frame0_clearance_mm": (zs[0] - top) * 1e3,
                                                "min_clearance_mm": (zs.min() - top) * 1e3, "min_frame": int(zs.argmin())}
        report[key] = ep
        print(key, "supports", [(round(x["top_z"], 4), round(x["radius"], 3)) for x in tops])
        for k, v in ep.items():
            if isinstance(v, dict) and "frame0_clearance_mm" in v:
                print(f"   {k:28s} frame0 {v['frame0_clearance_mm']:+8.2f} mm   min {v['min_clearance_mm']:+8.2f} mm @f{v['min_frame']}")
    OUT.write_text(json.dumps(report, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()

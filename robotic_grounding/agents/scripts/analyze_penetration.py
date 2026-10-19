#!/usr/bin/env python
"""Measure REAL hand-object penetration for retargeted taco sequences.

Unlike scripts/filter_penetrations.py (convex hull + missing *_cm scale), this
uses the actual object meshes, correctly scaled, and reports per-frame series:
  - robot hand capsules (same 21-capsule primitive URDF as upstream) vs mesh
  - MANO source fingertips vs the same mesh (was the demo already penetrating?)
  - robot hand-hand capsule penetration (upstream's own routine)

Depth convention: positive = inside the object (m); sphere depth = sd + radius.
Writes artifacts/repro-taco/glb/<sid>.penetration.json for the review page.

Run with Env B: .venv/bin/python agents/scripts/analyze_penetration.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import trimesh

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("fp", REPO / "scripts/filter_penetrations.py")
fp = importlib.util.module_from_spec(spec)
sys.modules["fp"] = fp
spec.loader.exec_module(fp)

from robotic_grounding.retarget.data_logger import ManoSharpaData  # noqa: E402
from robotic_grounding.retarget.params import MANO_FINGERTIP_INDICES  # noqa: E402

PROCESSED = Path("/root/Data/human_motion_data/taco/taco_processed")
MESH_DIR = Path("/root/Data/human_motion_data/taco/object_assets/meshes/taco")
OUT_DIR = REPO / "artifacts/repro-taco/glb"
SEQS = sys.argv[1:] or [
    "taco_cut__knife__bowl_20231020_254",
    "taco_dust__brush__bowl_20230927_028",
]


def load_mesh(path: str) -> trimesh.Trimesh:
    p = Path(path)
    if not p.is_file():
        p = MESH_DIR / p.name
    m = trimesh.load(str(p), force="mesh")
    if p.name.endswith("_cm.obj"):
        m.vertices = m.vertices * 0.01
    return m


def max_depth_points(mesh: trimesh.Trimesh, pts_world, radii, obj_pos, obj_R):
    """Max penetration depth (m) of spheres at pts_world into posed mesh."""
    pts_local = (np.asarray(pts_world) - obj_pos) @ obj_R  # R^T applied row-wise
    sd = trimesh.proximity.signed_distance(mesh, pts_local)  # positive = inside
    depths = sd + np.asarray(radii)
    i = int(np.argmax(depths))
    return float(depths[i]), i


def analyze(sid: str) -> dict:
    data = ManoSharpaData.from_parquet(
        root_path=str(PROCESSED),
        filters=[("sequence_id", "=", sid), ("robot_name", "=", "sharpa_wave")],
        trajectory_id=0,
    )
    F = len(data.robot_right_wrist_position)
    fps = float(data.fps)
    right_shapes, left_shapes = fp._get_shapes()
    right_hsc = fp._HandShapeCache(right_shapes, list(data.right_robot_frame_names))
    left_hsc = fp._HandShapeCache(left_shapes, list(data.left_robot_frame_names))

    bodies = list(data.object_body_names)
    meshes = [load_mesh(p) for p in data.object_mesh_paths]
    obj_pos = np.asarray(data.object_body_position)  # (F, n, 3)
    obj_wxyz = np.asarray(data.object_body_wxyz)

    mano = {
        "right": np.asarray(data.mano_right_joints)[:, MANO_FINGERTIP_INDICES, :],
        "left": np.asarray(data.mano_left_joints)[:, MANO_FINGERTIP_INDICES, :],
    }

    series = {k: np.zeros(F) for k in
              ("robot_right", "robot_left", "mano_right", "mano_left", "hand_hand")}
    per_body_max = {f"{h}:{b}": 0.0 for h in ("robot_right", "robot_left") for b in bodies}
    worst = {"depth": -1e9, "frame": 0, "link": "", "body": "", "hand": ""}

    rf = np.asarray(data.robot_right_frames)
    lf = np.asarray(data.robot_left_frames)
    for t in range(F):
        caps = {"right": right_hsc.world_spheres(rf[t]), "left": left_hsc.world_spheres(lf[t])}
        cap_meta = {
            side: ([ep for c in cl for ep in (c[0], c[1])], [c[2] for c in cl for _ in range(2)])
            for side, cl in caps.items()
        }
        link_names = {
            side: [s.link_name for _, s in hsc.entries for _ in range(2)]
            for side, hsc in (("right", right_hsc), ("left", left_hsc))
        }
        for bi, (bname, mesh) in enumerate(zip(bodies, meshes)):
            R = fp._quat_wxyz_to_matrix(obj_wxyz[t, bi])
            p0 = obj_pos[t, bi]
            for side in ("right", "left"):
                pts, radii = cap_meta[side]
                d, i = max_depth_points(mesh, pts, radii, p0, R)
                key = f"robot_{side}"
                series[key][t] = max(series[key][t], d)
                per_body_max[f"{key}:{bname}"] = max(per_body_max[f"{key}:{bname}"], d)
                if d > worst["depth"]:
                    worst = {"depth": d, "frame": t, "link": link_names[side][i],
                             "body": bname, "hand": side}
                md, _ = max_depth_points(mesh, mano[side][t], np.zeros(5), p0, R)
                series[f"mano_{side}"][t] = max(series[f"mano_{side}"][t], md)
        series["hand_hand"][t] = fp._max_hand_hand_penetration(caps["right"], caps["left"])

    out = {
        "sequence_id": sid,
        "frames": F,
        "fps": fps,
        "depth_convention": "m, positive = inside object; robot = capsule surface, mano = fingertip point",
        "bodies": [
            {"name": b, "mesh": Path(p).name, "watertight": bool(m.is_watertight),
             "hull_volume_ratio": round(float(m.convex_hull.volume / m.volume), 2)}
            for b, p, m in zip(bodies, data.object_mesh_paths, meshes)
        ],
        "series_m": {k: np.round(v, 5).tolist() for k, v in series.items()},
        "per_body_max_m": {k: round(v, 5) for k, v in per_body_max.items()},
        "summary": {
            "robot_max_m": round(float(max(series["robot_right"].max(), series["robot_left"].max())), 5),
            "worst": {k: (round(v, 5) if isinstance(v, float) else v) for k, v in worst.items()},
            "robot_frames_over_5mm_right": int((series["robot_right"] > 0.005).sum()),
            "robot_frames_over_5mm_left": int((series["robot_left"] > 0.005).sum()),
            "mano_max_m": round(float(max(series["mano_right"].max(), series["mano_left"].max())), 5),
            "hand_hand_max_m": round(float(series["hand_hand"].max()), 5),
        },
    }
    (OUT_DIR / f"{sid}.penetration.json").write_text(json.dumps(out, indent=1))
    s = out["summary"]
    print(f"{sid}:")
    print(f"  robot max depth {s['robot_max_m']*1000:.1f} mm at frame {worst['frame']} "
          f"({worst['hand']} {worst['link']} -> {worst['body']})")
    print(f"  frames >5mm: right {s['robot_frames_over_5mm_right']}/{F}, left {s['robot_frames_over_5mm_left']}/{F}")
    print(f"  MANO source fingertip max depth {s['mano_max_m']*1000:.1f} mm; hand-hand max {s['hand_hand_max_m']*1000:.1f} mm")
    return out


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for sid in SEQS:
        analyze(sid)

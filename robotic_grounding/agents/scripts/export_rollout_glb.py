#!/usr/bin/env python
"""Build an animated GLB from a recorded policy rollout (record_rollout_states.py NPZ).

Scene: Sharpa hands via pinocchio FK of recorded wrist pose + joint positions
(joints mapped sim-name -> pin-q-index), objects at recorded sim poses (solid),
reference object as a translucent ghost, reference fingertip targets as green
markers. Writes <out>.metrics.json in the same pass (data-then-render), including
an FK-vs-recorded-sim-fingertip consistency check.

Usage: .venv/bin/python agents/scripts/export_rollout_glb.py --npz <file> \
         --processed_dir <seq dir> --out <file.glb>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pinocchio as pin
import trimesh
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, os.path.expanduser("~/.claude/skills/glb-review"))
from glb_builder import GLBBuilder  # noqa: E402

from robotic_grounding.retarget.data_logger import ManoSharpaData  # noqa: E402
from robotic_grounding.retarget.retarget_utils import setup_sharpa_kinematics  # noqa: E402

MESH_DIR = Path("/root/Data/human_motion_data/taco/object_assets/meshes/taco")


def wxyz_to_xyzw(q):
    return np.asarray(q)[..., [1, 2, 3, 0]]


def load_mesh(path: str) -> trimesh.Trimesh:
    p = Path(path)
    if not p.is_file():
        p = MESH_DIR / p.name
    m = trimesh.load(str(p), force="mesh")
    if p.name.endswith("_cm.obj"):
        m.vertices = m.vertices * 0.01
    return m


def add_hand_from_rollout(b, side, d, meta, metrics):
    kin = setup_sharpa_kinematics(side)
    robot = kin.robot
    model, rdata = robot.model, robot.data
    gmodel = robot.visual_model
    gdata = gmodel.createData()

    # sim joint name -> pin q index (floating base occupies q[0:7])
    pin_qidx = {}
    for jid in range(1, model.njoints):  # 0 = universe
        name = model.names[jid]
        if name == "root_joint":
            continue
        pin_qidx[name] = model.joints[jid].idx_q
    sim_names = meta[f"{side}_joint_names"]
    unmatched = [n for n in sim_names if n not in pin_qidx]
    assert not unmatched, f"sim joints missing in pin model: {unmatched}"

    wrist_pos = d[f"{side}_wrist_pos"]
    wrist_wxyz = d[f"{side}_wrist_wxyz"]
    joint_pos = d[f"{side}_joint_pos"]
    F = len(wrist_pos)
    ngeom = len(gmodel.geometryObjects)
    trans = np.zeros((ngeom, F, 3), np.float32)
    quat = np.zeros((ngeom, F, 4), np.float32)

    # FK-vs-sim fingertip validation
    tip_names = meta[f"{side}_fingertip_body_names"]
    tip_fids = [model.getFrameId(n) for n in tip_names]
    tip_err = np.zeros((F, len(tip_fids)), np.float32)
    sim_tips = d[f"{side}_fingertip_pos"]

    for f in range(F):
        q = robot.q0.copy()
        q[:3] = wrist_pos[f]
        q[3:7] = wxyz_to_xyzw(wrist_wxyz[f])
        for n, v in zip(sim_names, joint_pos[f]):
            q[pin_qidx[n]] = v
        pin.forwardKinematics(model, rdata, q)
        pin.updateFramePlacements(model, rdata)
        pin.updateGeometryPlacements(model, rdata, gmodel, gdata)
        for i in range(ngeom):
            om = gdata.oMg[i]
            trans[i, f] = om.translation
            quat[i, f] = R.from_matrix(om.rotation).as_quat()
        for j, fid in enumerate(tip_fids):
            tip_err[f, j] = np.linalg.norm(rdata.oMf[fid].translation - sim_tips[f, j])

    tint = (0.72, 0.74, 0.78) if side == "right" else (0.55, 0.62, 0.72)
    for i, g in enumerate(gmodel.geometryObjects):
        mesh = trimesh.load(g.meshPath, force="mesh")
        scale = np.asarray(g.meshScale).reshape(-1)
        if not np.allclose(scale, 1.0):
            mesh.vertices = mesh.vertices * scale
        col = np.asarray(g.meshColor).reshape(-1)
        color = tuple(col[:3]) if col[:3].max() > 0 else tint
        b.add_node(mesh, color, f"{side}__{g.name}", trans[i], quat[i])

    metrics[f"{side}_fk_vs_sim_fingertips_m"] = {
        "mean": float(tip_err.mean()), "max": float(tip_err.max())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--processed_dir", required=True,
                    help="processed sequence partition dir (for object mesh paths)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    z = np.load(args.npz)
    meta = json.loads(str(z["meta"][0]))
    d = {k: z[k] for k in z.files if k != "meta"}
    # Keep the LONGEST clean episode: recordings contain settle-resets (contact
    # impulses on the freshly placed grasp) and post-time-out re-grasp churn;
    # episodes are the monotonic timestep_counter segments between drops.
    tc = d["ref_timestep"]
    drops = np.where(np.diff(tc) < 0)[0]
    bounds = [0] + (drops + 1).tolist() + [len(tc)]
    s0, s1 = max(((bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)),
                 key=lambda s: s[1] - s[0])
    if (s0, s1) != (0, len(tc)):
        d = {k: v[s0:s1] for k, v in d.items()}
        print(f"[glb] kept longest episode: steps {s0}..{s1-1} (tc {tc[s0]}..{tc[s1-1]})")
    F = d["right_wrist_pos"].shape[0]
    fps = 1.0 / meta["step_dt"]
    times = np.arange(F, dtype=np.float32) * meta["step_dt"]

    pdir = Path(args.processed_dir)
    pq = ManoSharpaData.from_parquet(
        root_path=str(pdir.parent.parent),
        filters=[("sequence_id", "=", pdir.parent.name.replace("sequence_id=", "")),
                 ("robot_name", "=", pdir.name.replace("robot_name=", ""))],
        trajectory_id=0)
    mesh_paths = list(pq.object_mesh_paths)
    body_names = meta["object_body_names"]

    metrics = {"steps": F, "fps": round(fps, 2), "npz": args.npz,
               "checkpoint": meta["checkpoint"],
               "note": "policy rollout (deterministic, reset at frame 0); ghost = reference object; green = reference fingertips"}
    b = GLBBuilder(times)
    for side in ("right", "left"):
        add_hand_from_rollout(b, side, d, meta, metrics)
        sphere = trimesh.creation.icosphere(subdivisions=2, radius=0.006)
        ref = d[f"ref_{side}_fingertip_pos"]  # (F, 5, 3)
        for k in range(ref.shape[1]):
            b.add_node(sphere, (0.15, 0.75, 0.3), f"ref_{side}__tip{k}",
                       ref[:, k, :].astype(np.float32), alpha=0.9)
    palette = [(0.85, 0.62, 0.25), (0.62, 0.4, 0.72)]
    for i, name in enumerate(body_names):
        mesh = load_mesh(mesh_paths[i])
        b.add_node(mesh, palette[i % 2], f"obj__{name}",
                   d["object_pos"][:, i, :].astype(np.float32),
                   wxyz_to_xyzw(d["object_wxyz"][:, i, :]).astype(np.float32))
        b.add_node(mesh, (0.5, 0.55, 0.6), f"ref__{name}",
                   d["ref_object_pos"][:, i, :].astype(np.float32),
                   wxyz_to_xyzw(d["ref_object_wxyz"][:, i, :]).astype(np.float32),
                   alpha=0.3)
    b.save(args.out)
    Path(args.out).with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=1))

    from pygltflib import GLTF2
    g = GLTF2().load(args.out)
    for m in g.meshes:
        for p in m.primitives:
            assert p.attributes.NORMAL is not None
    print(f"[glb] {args.out}: {Path(args.out).stat().st_size/1e6:.1f} MB, {len(g.nodes)} nodes, "
          f"{F} steps | fk_vs_sim tips max r={metrics['right_fk_vs_sim_fingertips_m']['max']:.4f} "
          f"l={metrics['left_fk_vs_sim_fingertips_m']['max']:.4f} m")


if __name__ == "__main__":
    main()

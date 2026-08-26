#!/usr/bin/env python
"""Export retargeted ManoSharpaData (<ds>_processed parquet) to an animated GLB.

One GLB per sequence: Sharpa Wave hand meshes (pinocchio FK over the MJCF visual
model), object meshes with their pose channels, and green MANO-fingertip target
markers. Writes a `<stem>.metrics.json` sidecar in the same pass (rule
data-then-render) including an FK-vs-stored-frames consistency check.

Run with the robotic_grounding pipeline venv (Env B):
    .venv/bin/python agents/scripts/export_processed_glb.py \
        --input_dir $HMD/taco/taco_processed --sequence_id <sid> --out_dir artifacts/repro-taco/glb
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

from robotic_grounding.retarget import MESHES_DIR  # noqa: E402
from robotic_grounding.retarget.data_logger import ManoSharpaData  # noqa: E402
from robotic_grounding.retarget.params import MANO_FINGERTIP_INDICES  # noqa: E402
from robotic_grounding.retarget.retarget_utils import setup_sharpa_kinematics  # noqa: E402

FINGERTIP_NAMES = ["thumb", "index", "middle", "ring", "pinky"]


def wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    return np.asarray(q)[..., [1, 2, 3, 0]]


def resolve_mesh(path: str) -> str | None:
    if path and Path(path).is_file():
        return path
    # Fall back to the package meshes dir (paths baked in another environment).
    cand = MESHES_DIR / "taco" / Path(path).name
    return str(cand) if cand.is_file() else None


def add_hand(b: GLBBuilder, side: str, data: ManoSharpaData, metrics: dict) -> None:
    kin = setup_sharpa_kinematics(side)
    robot = kin.robot
    model, rdata = robot.model, robot.data
    gmodel = robot.visual_model
    gdata = gmodel.createData()

    wrist_pos = np.asarray(getattr(data, f"robot_{side}_wrist_position"))
    wrist_wxyz = np.asarray(getattr(data, f"robot_{side}_wrist_wxyz"))
    fingers = np.asarray(getattr(data, f"robot_{side}_finger_joints"))
    frame_names = list(getattr(data, f"{side}_robot_frame_names"))
    stored_frames = np.asarray(getattr(data, f"robot_{side}_frames"))
    F = len(wrist_pos)

    ngeom = len(gmodel.geometryObjects)
    trans = np.zeros((ngeom, F, 3), np.float32)
    quat = np.zeros((ngeom, F, 4), np.float32)
    frame_ids = [model.getFrameId(n) for n in frame_names]
    fk_err = np.zeros((F, len(frame_ids)), np.float32)

    for f in range(F):
        q = robot.q0.copy()
        q[:3] = wrist_pos[f]
        q[3:7] = wxyz_to_xyzw(wrist_wxyz[f])
        q[7:] = fingers[f]
        pin.forwardKinematics(model, rdata, q)
        pin.updateFramePlacements(model, rdata)
        pin.updateGeometryPlacements(model, rdata, gmodel, gdata)
        for i in range(ngeom):
            om = gdata.oMg[i]
            trans[i, f] = om.translation
            quat[i, f] = R.from_matrix(om.rotation).as_quat()
        for j, fid in enumerate(frame_ids):
            fk_err[f, j] = np.linalg.norm(
                rdata.oMf[fid].translation - stored_frames[f, j, :3]
            )

    tint = (0.72, 0.74, 0.78) if side == "right" else (0.55, 0.62, 0.72)
    for i, g in enumerate(gmodel.geometryObjects):
        mesh = trimesh.load(g.meshPath, force="mesh")
        scale = np.asarray(g.meshScale).reshape(-1)
        if not np.allclose(scale, 1.0):
            mesh.vertices = mesh.vertices * scale
        col = np.asarray(g.meshColor).reshape(-1)
        color = tuple(col[:3]) if col[:3].max() > 0 else tint
        b.add_node(mesh, color, f"{side}__{g.name}", trans[i], quat[i])

    metrics[f"{side}_fk_vs_stored_frames_m"] = {
        "mean": float(fk_err.mean()),
        "max": float(fk_err.max()),
    }
    metrics[f"{side}_ik_frame_task_err_per_frame_mean_m"] = (
        np.asarray(getattr(data, f"robot_{side}_frame_task_errors"))
        .reshape(F, -1)
        .mean(axis=1)
        .round(6)
        .tolist()
    )


def add_mano_targets(b: GLBBuilder, side: str, data: ManoSharpaData) -> None:
    joints = np.asarray(getattr(data, f"mano_{side}_joints"))  # (F, 21, 3)
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=0.006)
    for k, idx in enumerate(MANO_FINGERTIP_INDICES):
        b.add_node(
            sphere,
            (0.15, 0.75, 0.3),
            f"mano_{side}__{FINGERTIP_NAMES[k]}_target",
            joints[:, idx, :].astype(np.float32),
            alpha=0.9,
        )


def add_objects(b: GLBBuilder, data: ManoSharpaData, metrics: dict) -> None:
    names = list(data.object_body_names)
    mesh_paths = list(data.object_mesh_paths or [])
    pos = np.asarray(data.object_body_position)  # (F, n, 3)
    wxyz = np.asarray(data.object_body_wxyz)  # (F, n, 4)
    palette = [(0.85, 0.62, 0.25), (0.62, 0.4, 0.72), (0.4, 0.6, 0.5)]
    loaded = []
    for i, name in enumerate(names):
        resolved = resolve_mesh(mesh_paths[i]) if i < len(mesh_paths) else None
        if resolved is None:
            print(f"  [warn] no mesh for object {name!r}; skipping")
            continue
        mesh = trimesh.load(resolved, force="mesh")
        if resolved.endswith("_cm.obj"):
            mesh.vertices = mesh.vertices * 0.01  # TACO meshes are in cm
        b.add_node(
            mesh,
            palette[i % len(palette)],
            f"obj__{name}",
            pos[:, i, :].astype(np.float32),
            wxyz_to_xyzw(wxyz[:, i, :]).astype(np.float32),
        )
        loaded.append(name)
    metrics["objects"] = loaded


def export_one(input_dir: Path, sequence_id: str, out_dir: Path) -> dict:
    data = ManoSharpaData.from_parquet(
        root_path=str(input_dir),
        filters=[("sequence_id", "=", sequence_id), ("robot_name", "=", "sharpa_wave")],
        trajectory_id=0,
    )
    F = len(data.robot_right_wrist_position)
    fps = float(data.fps)
    times = np.arange(F, dtype=np.float32) / fps
    metrics: dict = {
        "sequence_id": sequence_id,
        "frames": F,
        "fps": fps,
        "source": str(input_dir),
    }
    b = GLBBuilder(times)
    for side in ("right", "left"):
        add_hand(b, side, data, metrics)
        add_mano_targets(b, side, data)
    add_objects(b, data, metrics)

    out = out_dir / f"{sequence_id}.glb"
    b.save(str(out))
    (out_dir / f"{sequence_id}.metrics.json").write_text(json.dumps(metrics, indent=1))

    # Structural validation (SKILL.md fallback checks).
    from pygltflib import GLTF2

    g = GLTF2().load(str(out))
    for m in g.meshes:
        for p in m.primitives:
            assert p.attributes.NORMAL is not None, "primitive missing NORMAL"
    mesh_nodes = [n for n in g.nodes if n.mesh is not None]
    assert all(n.translation is not None for n in mesh_nodes), "frame-0 TRS not baked"
    root = g.nodes[g.scenes[0].nodes[0]]
    assert root.rotation is not None, "root missing Z-up->Y-up rotation"
    print(
        f"  {out.name}: {out.stat().st_size/1e6:.1f} MB, {len(g.nodes)} nodes, "
        f"{len(g.meshes)} meshes, {F} frames | fk_vs_stored max "
        f"r={metrics['right_fk_vs_stored_frames_m']['max']:.2e} "
        f"l={metrics['left_fk_vs_stored_frames_m']['max']:.2e} m"
    )
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", type=Path, required=True)
    ap.add_argument("--sequence_id", action="append", required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for sid in args.sequence_id:
        print(f"Exporting {sid}")
        export_one(args.input_dir, sid, args.out_dir)


if __name__ == "__main__":
    main()
